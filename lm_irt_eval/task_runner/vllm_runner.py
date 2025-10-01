import os
import random
from typing import List, Dict, Any
from dataclasses import dataclass

from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor
from vllm import (
    LLM,
    SamplingParams,
    RequestOutput,
    CompletionOutput,
)
from torch.utils.data import DataLoader
from openai_harmony import Role

from .base import TaskRunner
from .dataset import Collator
from ..utils import OPENAI_ENCODING, create_logger, log_exception_with_traceback


@dataclass
class VLLMParamConfigurer(object):
    r"""For handling model-specific logics when calling VLLM"""

    model_name: str
    max_model_len: int
    trust_remote_code: bool = False

    def get_vllm_init_params(self):
        extra_vllm_kwargs = {
            "max_model_len": self.max_model_len,
            "trust_remote_code": self.trust_remote_code,
        }
        if "hunyuan" in self.model_name.lower():
            extra_vllm_kwargs["trust_remote_code"] = True
        if "gemma" in self.model_name.lower():
            extra_vllm_kwargs["dtype"] = "bfloat16"
            extra_vllm_kwargs["max_model_len"] = 8192
        return extra_vllm_kwargs

    def get_vllm_sampling_params(self):
        extra_kwargs = {
            "temperature": 0.,
        }
        if "phi-4-reasoning-plus" in self.model_name:
            # As suggested in
            # https://huggingface.co/microsoft/Phi-4-reasoning-plus
            extra_kwargs.update(
                {
                    "top_k": 50,
                    "top_p": 0.9,
                    "temperature": 0.8,
                }
            )
        if "MiMo-7B-SFT" in self.model_name:
            # As suggested in
            # https://huggingface.co/XiaomiMiMo/MiMo-7B-SFT
            extra_kwargs.update(
                {
                    "temperature": 0.6,
                }
            )
        if "NVIDIA-Nemotron-Nano" in self.model_name:
            # As suggested in
            # https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2
            extra_kwargs.update(
                {
                    "temperature": 0.6,
                    "top_p": 0.95,
                }
            )
        if "Mistral" in self.model_name:
            extra_kwargs.update(
                {
                    "temperature": 0.6,
                }
            )
        if "gpt-oss" in self.model_name:
            stop_token_ids = OPENAI_ENCODING.stop_tokens_for_assistant_actions()
            extra_kwargs.update(
                {
                    "stop_token_ids": stop_token_ids,
                    # As suggested by https://huggingface.co/openai/gpt-oss-20b/discussions/28
                    "temperature": 0.5,
                }
            )
        return extra_kwargs

    # def get_tokenizer_params(self):
    #     extra_kwargs = {
    #         "trust_remote_code": self.trust_remote_code,
    #     }
    #     if "hunyuan" in self.model_name.lower():
    #         extra_kwargs["trust_remote_code"] = True
    #     if "gemma" in self.model_name.lower():
    #         extra_kwargs["trust_remote_code"] = True
    #     return extra_kwargs


@dataclass
class VLLMTaskRunner(TaskRunner):
    r"""Local task runner via VLLM
    In VLLM Runner, the model_name SHALL be explicitly specified as the local path"""

    max_model_len: int = 32768
    trust_remote_code: bool = False
    batch_size: int = 16
    tensor_parallel_size: int = 2
    # For parallel sampling treatments
    sampling_n: int = 1

    def __post_init__(self):
        super(VLLMTaskRunner, self).__post_init__()
        assert os.path.exists(self.model_name_or_path)
        self.model_name = self._extract_model_name()
        self.kwargs_configurer = VLLMParamConfigurer(
            model_name=self.model_name,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
        )
        extra_vllm_kwargs = self.kwargs_configurer.get_vllm_init_params()
        self.llm = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=2 * self.batch_size * self.sampling_n,
            **extra_vllm_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.logger = create_logger()

    @property
    def requires_harmony(self):
        return "gpt-oss" in self.model_name.lower()

    @staticmethod
    def _parse_harmony_output(output: CompletionOutput):
        r"""Parse OpenAI GPT-OSS model results using the harmony recipe
        documented in https://cookbook.openai.com/articles/gpt-oss/run-vllm"""
        output_tokens = output.token_ids
        entries = OPENAI_ENCODING.parse_messages_from_completion_tokens(
            output_tokens,
            Role.ASSISTANT
        )
        infer_output, reasoning_content = "", ""
        for message in entries:
            contents = message.content
            channel = message.channel
            if channel == "final":
                for content in contents:
                    infer_output += content.text
            else:
                for content in contents:
                    reasoning_content += content.text
        return infer_output, reasoning_content

    @staticmethod
    def _filter_outputs_by_length(req_output: RequestOutput):
        r"""Pick one random outputs that is finished by ``stop`` status
        Otherwise warns and randomly select one truncated answer"""
        if len(req_output.outputs) == 1:
            return req_output.outputs[0]
        random.shuffle(req_output.outputs)
        for output in req_output.outputs:
            if output.finish_reason == "stop":
                return output
        return random.choice(req_output.outputs)

    def _parse_output(
        self,
        response: List[RequestOutput],
        requests: Dict[str, List[Any]],
    ) -> List[Dict[str, Any]]:
        parsed_outputs = []
        batch_size = len(response)
        for i in range(batch_size):
            req_output = response[i]
            parsed_output = {
                "request_model": self.model_name,
                "query_model": self.model_name,
                "enable_thinking": self.enable_thinking,
            }
            output = self._filter_outputs_by_length(req_output)
            if self.requires_harmony:
                infer_output, reasoning_content = self._parse_harmony_output(
                    output
                )
            else:
                # TODO: maybe use reasoning parser here
                infer_output, reasoning_content = output.text, ""
            finish_reason = output.finish_reason
            n_prompt_tokens = len(req_output.prompt_token_ids)
            n_output_tokens = len(output.token_ids)
            parsed_output["infer_output"] = infer_output
            parsed_output["reasoning_content"] = reasoning_content
            parsed_output["finish_reason"] = finish_reason
            parsed_output["n_prompt_tokens"] = n_prompt_tokens
            parsed_output["n_output_tokens"] = n_output_tokens
            parsed_output["key"] = requests["key"][i]
            parsed_output["payload"] = requests["payload"][i]
            parsed_outputs.append(parsed_output)
        return parsed_outputs

    def run_task(self):
        collator = Collator(
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            enable_thinking=self.enable_thinking,
        )
        loader = DataLoader(
            self.dataset,
            self.batch_size,
            num_workers=16,
            collate_fn=collator,
        )
        extra_sampling_params_kwargs = self.kwargs_configurer.get_vllm_sampling_params()
        sampling_params = SamplingParams(
            n=self.sampling_n,
            max_tokens=self.max_tokens,
            **extra_sampling_params_kwargs,
        )
        results = []
        for batch in tqdm(loader):
            try:
                prompts = batch["prompt"]
                req_outputs = self.llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                parsed_outputs = self._parse_output(
                    req_outputs,
                    batch,
                )
                self.json_writer.append(parsed_outputs)
                results.extend(parsed_outputs)
            except Exception as e:
                self.logger.error(e)
                log_exception_with_traceback(self.logger)
        return results
