import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import (
    OpenAI,
    APIError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)
from swift.utils import JsonlWriter
from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type
)

from .reasoning_parser import CompositeOutputParser
from ..utils import (
    create_logger,
    log_exception_with_traceback,
    NLGTaskType,
    normalize,
    compute_rouge_batch,
    compute_meteor_batch,
    compute_bleu_batch,
    compute_bertscore_batch
)


class MetricCalculator(object):

    @staticmethod
    def _llm_as_a_judge_prompt(candidate, ground_truth):
        text_prompt = f"""
The following are a candidate answer generated from some LLM as well as the ground truth:
- candidate: {candidate}
- ground_truth: {ground_truth}

Please determine whether the candidate matches the ground truth. Your output SHALL BE simply yes/no.

Your decision: """
        return text_prompt

    @staticmethod
    def _initialize_llm_as_a_judge_client(**kwargs):
        r"""Initialize an OpenAI client to use for doing (Optional) LLM-as-a-Judge"""
        api_key = os.environ.get("API_KEY", "")
        base_url = os.environ.get("OAI_API_URL", "")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        return client

    def __init__(
        self,
        answer_support,
        output_path: str,
        llm_as_a_judge_model,
        retries,
        wait_time,
        max_workers
    ):
        reasoning_parser_model_path = os.environ.get("REASONING_PARSER_MODEL_PATH")
        self.reasoning_parser = CompositeOutputParser(
            model_path=reasoning_parser_model_path,
            answer_support=answer_support,
        )
        self.json_writer = JsonlWriter(
            fpath=output_path,
        )
        # For LLM-as-a-Judge
        self.llm_as_a_judge_model = llm_as_a_judge_model
        self.retries = retries
        self.wait_time = wait_time
        self.max_workers = max_workers
        self.retryer = Retrying(
            stop=stop_after_attempt(self.retries),
            wait=wait_fixed(self.wait_time),
            retry=retry_if_exception_type(  # TODO: Maybe catch some more
                (
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    APIError,
                )
            ),
            reraise=True  # Reraise the exception if all retries fail
        )
        self._llm_as_a_judge_client = self._initialize_llm_as_a_judge_client()
        self.logger = create_logger()

    def batch_process_mcq(
        self,
        batch: List[Dict[str, Any]],
        global_skips_reasoning_parser=False,
        candidate_key="candidate",
        ground_truth_key="ground_truth",
    ):
        llm_as_a_judge_collection = []
        for item in batch:
            requires_llm_as_a_judge, answer = self.reasoning_parser.process_mcq(
                item[candidate_key],
                global_skips_reasoning_parser,
            )
            if requires_llm_as_a_judge:
                llm_as_a_judge_collection.append(item)
            else:
                # Record judgement in-place
                item["metric"] = {"is_correct": int(answer == item[ground_truth_key])}
                item["judgement_method"] = "direct_match"
                self.json_writer.append(item)
        if llm_as_a_judge_collection:
            self.logger.info(
                f"Found {len(llm_as_a_judge_collection)} records that requires LLM-as-a-Judge,"
                f"running LLM-as-a-Judge with judge model {self.llm_as_a_judge_model}..."
            )
            self._run_llm_as_a_judge(
                batch=llm_as_a_judge_collection,
                candidate_key=candidate_key,
                ground_truth_key=ground_truth_key,
            )
        return batch

    def _run_one_llm_as_a_judge(
        self,
        item: Dict[str, Any],
        candidate_key,
        ground_truth_key,
    ):
        text_prompt = self._llm_as_a_judge_prompt(
            candidate=item[candidate_key],
            ground_truth=item[ground_truth_key],
        )
        messages = [{"role": "user", "content": text_prompt}]
        try:
            # self.logger.info(f"Attempting API call with parameters: {kwargs}")
            response = self._llm_as_a_judge_client.chat.completions.create(
                messages=messages,
                model=self.llm_as_a_judge_model,
                max_tokens=4096,
            )
            judgement = response.choices[0].message.content
            item["metric"] = {"is_correct": int("yes" in judgement.lower())}
            item["judgement_method"] = f"<{self.llm_as_a_judge_model}>_as_a_judge"
            self.json_writer.append(item)
            return item
        except RateLimitError as e:
            self.logger.warning(f"Rate limit exceeded. Waiting for retry... Error: {e}")
            raise
        except APIConnectionError as e:
            self.logger.warning(f"API connection error. Waiting for retry... Error: {e}")
            raise
        except APITimeoutError as e:
            self.logger.warning(f"API timeout error. Waiting for retry... Error: {e}")
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

    def run_one_llm_as_a_judge(self, item: Dict[str, Any], candidate_key, ground_truth_key):
        return self.retryer(
            self._run_one_llm_as_a_judge,
            item=item,
            candidate_key=candidate_key,
            ground_truth_key=ground_truth_key,
        )

    def _run_llm_as_a_judge(
        self,
        batch,
        candidate_key="candidate",
        ground_truth_key="ground_truth",
    ):
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a future for each request
            future_to_request = {
                executor.submit(
                    self.run_one_llm_as_a_judge,
                    item=item,
                    candidate_key=candidate_key,
                    ground_truth_key=ground_truth_key,
                ): item
                for item in batch
            }

            for future in tqdm(
                as_completed(future_to_request),
                total=len(future_to_request),
                desc="Processing LLM-as-a-Judge requests"
            ):
                request_params = future_to_request[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    results.append({"request": request_params, "error": str(exc)})
                    self.logger.error(f'Request {request_params} generated an exception: {exc}')
                    log_exception_with_traceback(logger=self.logger)
        return results

    def _batch_process_nlg(self, candidates, references, task_type: NLGTaskType):
        candidates = normalize(
            candidates,
            strip_task_prefix=True,
            task_type=task_type
        )
        references = normalize(
            references,
            strip_task_prefix=False,
            task_type=task_type
        )
        # ROUGE
        r1, r2, rL = compute_rouge_batch(candidates, references)
        # BERTScore
        bP, bR, bF1 = compute_bertscore_batch(
            candidates,
            references,
        )
        # METEOR
        mtr = compute_meteor_batch(candidates, references)
        # BLEU
        bleu = compute_bleu_batch(
            candidates,
            references,
            tokenize="13a",
            smooth_method="exp"
        )
        n = len(candidates)
        output = []
        for i in range(n):
            metric = {
                "bleu": bleu[i],
                "meteor": mtr[i],
                "rouge1": r1[i],
                "rouge2": r2[i],
                "rougeL": rL[i],
                "bertscore_P": bP[i],
                "bertscore_R": bR[i],
                "bertscore_F1": bF1[i],
            }
            output.append(metric)
        return output

    def batch_process_nlg(
        self,
        batch: List[Dict[str, Any]],
        candidate_key="candidate",
        ground_truth_key="ground_truth",
        task_type: NLGTaskType = NLGTaskType.SUMMARY,
    ):
        r"""**Notes** General NLG never requires LLM-as-a-Judge."""
        df = pd.DataFrame(batch)

        def _strip_reasoning(text):
            _, result_content = self.reasoning_parser.reasoning_parser.extract_reasoning_content(
                model_output=text,
                request=None,  # TODO: perhaps create a dummy request
            )
            return result_content

        metrics = self._batch_process_nlg(
            candidates=df[candidate_key].apply(_strip_reasoning).tolist(),
            references=df[ground_truth_key].tolist(),
            task_type=task_type,
        )
        df["metric"] = metrics
        output = df.to_dict(orient="records")
        self.json_writer.append(output)
        return output
