import os
import json
import copy
import random
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from openai import (
    OpenAI,
    APIError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    PermissionDeniedError
)
from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type
)

from .base import TaskRunner
from ..utils import (
    create_logger,
    log_exception_with_traceback,
    NonRetriableHTTPError
)


class APIPlatform(Enum):
    AIML = "aiml"


@dataclass
class APITaskRunner(TaskRunner):
    r"""API task runner via either OpenAI-style or Bearer-style API
    **Notes**: While vLLM local server also supports Bearer-style API,
    for stability reasons always use the offline engine as implemented
    in the ``VLLMTaskRunner`` class."""

    max_workers: int = 30
    retries: int = 10
    wait_time: int = 5
    timeout: float = 120
    platform: APIPlatform = APIPlatform.AIML
    # While
    use_openai_client: bool = False

    def _update_aiml_payload(self, payload: Dict[str, Any]):
        r"""Update payload with thinking configurations,
        useful in Bearer-style API."""
        model = payload["model"].lower()
        # Case-by-case with models
        if "qwen" in model:
            # payload["enable_thinking"] = True  # TODO: Make a reasonable warning
            payload["thinking_budget"] = self.thinking_budget
        if "grok" in model:
            payload["thinking"] = {
                "max_tokens": self.thinking_budget,
                "exclude": not self.enable_thinking,
            }
        if "openai" in model or "gpt" in model:
            # TODO: maybe auto adjust reasoning effort
            payload["reasoning_effort"] = "low"
        if "glm" in model:
            payload["thinking"] = {
                "type": "enabled" \
                    if self.enable_thinking else "disabled",
            }
        if "claude" in model:
            payload["thinking"] = {
                "type": "enabled", # Cannot be disabled...
                "budget_tokens": max(1024, self.thinking_budget),
            }
        return payload

    def _update_bearer_payload(self, payload: Dict[str, Any]):
        rev_payload = copy.deepcopy(payload)
        # No raise for alternative platforms
        if self.platform is APIPlatform.AIML:
            self._update_aiml_payload(rev_payload)
        return rev_payload

    def __post_init__(self):
        super(APITaskRunner, self).__post_init__()
        if self.use_openai_client:
            self.api_key = os.environ.get("API_KEY", "")
            self.base_url = os.environ.get("OAI_API_URL", "")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            self.headers = None  # No need for headers using oai client
        else:
            api_key = os.environ.get("API_KEY", "")
            self.api_key = f"Bearer {api_key}"
            self.base_url = os.environ.get("BEARER_API_URL", "")
            self.client = None
            self.headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            }
        self.logger = create_logger()

    def _execute_bearer_api_call(self, **kwargs) -> Dict[str, Any]:
        r"""Executes the actual HTTP POST request and handles response codes."""
        kwargs = self._update_bearer_payload(kwargs)
        try:
            with requests.post(
                self.base_url,
                headers=self.headers,
                json=kwargs,
                timeout=self.timeout
            ) as response:
                # Check for HTTP errors and decide if we should retry
                status_code = response.status_code
                json_response = response.json()
                status_code = max(status_code, json_response.get("statusCode", 200))
                if status_code >= 400:
                    error_message = f"HTTP Error {status_code}: {response.text}"
                    self.logger.error(error_message)
                    # For client-side errors (4xx), we usually don't want to retry,
                    # except for 429 (Rate Limit). [1]
                    if status_code == 400:
                        self.logger.warning("Request timeout. Waiting for retry...")
                        response.raise_for_status()
                    elif status_code == 429:
                        self.logger.warning("Rate limit exceeded. Waiting for retry...")
                        # We can raise a generic RequestException to trigger tenacity's retry
                        response.raise_for_status()
                    # For server-side errors (5xx), we want to retry. [1]
                    elif 500 <= status_code < 600:
                        self.logger.warning(f"Server error ({status_code}). Waiting for retry...")
                        response.raise_for_status()
                    # For other 4xx errors, raise a non-retriable error. [7]
                    else:
                        raise NonRetriableHTTPError(error_message, status_code)
                return json_response
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"A network-level error occurred: {e}. Waiting for retry...")
            raise

    def _execute_oai_api_call(self, **kwargs):
        r"""Executes the actual API call. This method is wrapped by the retry logic."""
        try:
            # self.logger.info(f"Attempting API call with parameters: {kwargs}")
            response = self.client.chat.completions.create(**kwargs)
            return response.model_dump_json()
        except RateLimitError as e:
            self.logger.warning(f"Rate limit exceeded. Waiting for retry... Error: {e}")
            raise
        except APIConnectionError as e:
            self.logger.warning(f"API connection error. Waiting for retry... Error: {e}")
            raise
        except APITimeoutError as e:
            self.logger.warning(f"API timeout error. Waiting for retry... Error: {e}")
            raise
        except PermissionDeniedError as e:
            self.logger.warning(f"Permission denied error. Waiting for retry... Error: {e}")
            raise
        except APIError as e:
            # Check if the error is a server-side issue (5xx) that might resolve on retry
            if 500 <= int(e.code) < 600:
                self.logger.warning(f"API server error ({int(e.code)}). Waiting for retry... Error: {e}")
                raise
            else:
                # For client-side errors (4xx), we don't want to retry.
                self.logger.error(f"A non-retriable API error occurred: {e}")
                raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise

    def _make_api_call_with_retry(self, **kwargs) -> Dict[str, Any]:
        r"""Makes a single API call to the OpenAI ChatCompletion endpoint with dynamic retry logic."""
        # Create a Retrying object with instance-specific configurations
        retryer = Retrying(
            stop=stop_after_attempt(self.retries),
            wait=wait_fixed(self.wait_time),
            retry=retry_if_exception_type(  # TODO: Maybe catch some more
                (
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    APIError,
                    PermissionDeniedError,
                    requests.exceptions.HTTPError,
                    requests.exceptions.RequestException,
                )
            ),
            reraise=True  # Reraise the exception if all retries fail
        )

        # Before calling, adjust payload
        key = kwargs.pop("key")  # SHALL NOT be emtpy, let it raise KeyError
        payload = kwargs.pop("payload", {})
        model = kwargs["model"].lower()
        max_tokens = self.max_tokens
        if any(
            (
                "claude-3" in model,
                "gemma-2" in model,
                "gpt-4o-2024-05-13" in model,
            )
        ):
            max_tokens = 4096
        if any(
            (
                "qwen-max" in model,
                "gemini-2.0-flash" in model,
                "deepseek" in model,
            )
        ):
            max_tokens = 8192
        kwargs["max_tokens"] = max_tokens
        # Use the '__call__' method of the Retrying object to wrap the API call
        executor = self._execute_oai_api_call if self.use_openai_client else self._execute_bearer_api_call
        raw_result = retryer(executor, **kwargs)
        parsed_result = self._parse_response(
            {"request": kwargs, "response": raw_result}
        )
        parsed_result.update(
            {
                "key": key,
                "payload": payload,
            }
        )
        return parsed_result

    @staticmethod
    def _parse_oai_response(result):
        response = result["response"]
        request_model = result["request"]["model"]  # The configured model
        enable_thinking = result["request"].get("enable_thinking", False)
        # TODO: Make it more flexible if we allow n > 1
        choice = response.choices[0]
        query_model = response.model  # The "real" model queried by the api
        message = choice.message
        output = message.content
        reasoning_content = getattr(message, "reasoning_content", "")
        # Token summaries
        usage = response.usage
        finish_reason = choice.finish_reason
        n_prompt_tokens = usage.prompt_tokens
        n_output_tokens = usage.completion_tokens
        return {
            "infer_output": output,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "n_prompt_tokens": n_prompt_tokens,
            "n_output_tokens": n_output_tokens,
            "request_model": request_model,
            "query_model": query_model,
            "enable_thinking": enable_thinking,
        }

    @staticmethod
    def _parse_bearer_response(result):
        response = result["response"]
        request_model = result["request"]["model"]
        enable_thinking = result["request"].get("enable_thinking", False)
        choice = response["choices"][0]
        query_model = response["model"]
        message = choice["message"]
        output = message["content"]
        reasoning_content = message.get("reasoning_content", "")
        usage = response["usage"]
        finish_reason = choice["finish_reason"]
        n_prompt_tokens = usage["prompt_tokens"]
        n_output_tokens = usage["completion_tokens"]
        return {
            "infer_output": output,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "n_prompt_tokens": n_prompt_tokens,
            "n_output_tokens": n_output_tokens,
            "request_model": request_model,
            "query_model": query_model,
            "enable_thinking": enable_thinking,
        }

    def _parse_response(self, result):
        if self.use_openai_client:
            return self._parse_oai_response(result)
        else:
            return self._parse_bearer_response(result)

    def process_requests(self, reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r"""Processes a list of API requests concurrently using a thread pool.

        Args:
            reqs(List[Dict[str, Any]]): A list of dictionaries,
                where each dictionary contains the parameters for a single API call.
        Returns:
            A list of dictionaries containing the API responses.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a future for each request
            future_to_request = {
                executor.submit(self._make_api_call_with_retry, **req): req
                for req in reqs
            }

            for future in tqdm(
                as_completed(future_to_request),
                total=len(future_to_request),
                desc="Processing requests"
            ):
                request_params = future_to_request[future]
                try:
                    result = future.result()
                    results.append(result)
                    # self.logger.info(f"Successfully processed request: {request_params}")
                    self.json_writer.append(
                        obj=result,
                    )
                except Exception as exc:
                    results.append({"request": request_params, "error": str(exc)})
                    self.logger.error(f'Request {request_params} generated an exception: {exc}')
                    log_exception_with_traceback(logger=self.logger)
        return results

    def run_task(self):
        done_instances = set()
        output_path = self.json_writer.fpath
        if os.path.exists(output_path):
            with open(output_path, "r") as output_file:
                for line in output_file:
                    record = json.loads(line)
                    done_instances.add((record["key"], record["request_model"]))
        self.logger.info(f"Found existing output with {len(done_instances)} records")
        all_reqs = [
            s for s in self.dataset.samples \
            if (s["key"], s["model"]) not in done_instances
        ]  # Simply use all the samples
        random.shuffle(all_reqs)
        # Check format
        assert all("model" in req for req in all_reqs)
        return self.process_requests(all_reqs)
