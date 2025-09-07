import os
from typing import List, Dict, Any
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams, RequestOutput
from torch.utils.data import DataLoader

from .base import TaskRunner
from .dataset import Collator
from .utils import create_logger, log_exception_with_traceback


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

    def __post_init__(self):
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
            max_num_seqs=2 * self.batch_size,
            **extra_vllm_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.logger = create_logger()

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
            infer_output = req_output.outputs[0].text
            reasoning_content = ""  # TODO: maybe use reasoning parser here
            finish_reason = req_output.outputs[0].finish_reason
            n_prompt_tokens = len(req_output.prompt_token_ids)
            n_output_tokens = len(req_output.outputs[0].token_ids)
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
            max_tokens=self.max_tokens,
            **extra_sampling_params_kwargs,
        )
        results = []
        for batch in loader:
            try:
                req_outputs = self.llm.generate(
                    batch["prompt"],
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
