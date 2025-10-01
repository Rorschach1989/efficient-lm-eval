import os
import re
import enum
import json
from typing import List, Union, Optional
from dataclasses import dataclass

from transformers import AutoTokenizer
from vllm.reasoning import ReasoningParserManager

from ..utils import create_logger


class AnswerType(enum.Enum):
    MCQ = 1
    YN = 2
    LIST = 3
    GENERAL = 4


@dataclass
class DecomposedOutputs(object):

    thinking_content: str
    result_content: str
    answer_content: str

    def to_json(self):
        return json.dumps(
            self.__dict__,
            sort_keys=True,
            indent=4,
            ensure_asci=False,
        )


class CompositeOutputParser(object):

    @staticmethod
    def _get_default_tokenizer():
        model_root = os.environ.get("MODEL_ROOT", "")
        if model_root:
            qwen3_model = os.path.join(
                model_root,
                "Qwen",
                "Qwen3-0.6B",
            )
            return AutoTokenizer.from_pretrained(qwen3_model)
        else:
            return None

    @staticmethod
    def get_reasoning_parser_name(model_path):
        model_path = model_path.lower()
        if "qwen3" in model_path:
            parser = "qwen3"
        elif "gpt" in model_path:
            parser = "GptOss"
        elif "glm" in model_path:
            parser = "glm45"
        else:
            # Fallback to qwen3
            parser = "qwen3"
        return parser

    def __init__(
        self,
        model_path: str = None,
        answer_support: Union[str, List[str]] = None,
        strict_pattern: bool = True,
    ):
        if model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            tokenizer = self._get_default_tokenizer()
        assert tokenizer is not None
        reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser(
            self.get_reasoning_parser_name(model_path)
        )
        self.reasoning_parser = reasoning_parser_cls(tokenizer)
        self.potential_answers: Optional[set] = None
        self.answer_pattern = r".+?"  # As the default choice
        if isinstance(answer_support, str) and answer_support.startswith("mcq"):
            # mcq@{number_of_choices}
            number_of_choices = int(answer_support.split("@")[-1])
            last_choice = chr(ord("A") + number_of_choices - 1)
            self.potential_answers = {
                chr(i) for i in range(65, 65 + number_of_choices)
            }
            if strict_pattern:
                self.answer_pattern = rf"[A-{last_choice}]"
            self.answer_type = AnswerType.MCQ
        elif answer_support == "yn":
            if strict_pattern:
                self.answer_pattern = r"yes|no"
            self.answer_type = AnswerType.YN
        elif isinstance(answer_support, list):
            if strict_pattern:
                self.answer_pattern = r"|".join(
                    map(str, answer_support)
                )
            self.answer_type = AnswerType.LIST
        else:
            # Matching anything non-empty
            self.answer_type = AnswerType.GENERAL
        self.logger = create_logger()

    def _extract_boxed_answers(self, text):
        r"""Extract answers placed in boxes like \boxed{A}, TeX-like grammar
        Often encountered in Qwen QVQ, MiMo or InternLM"""
        matches = re.findall(
            rf"\\boxed\{{({self.answer_pattern})\}}",
            text,
            re.DOTALL
        )
        if matches:
            return True, matches[-1]  # Always pick the last occurrence
        else:
            return False, text

    def _extract_glm_boxed_answers(self, text):
        r"""Extract answers produced by GLM-4.1/4.5 types of models which are
        enclosed in the <|begin_of_box|><|end_of_box|> tag"""
        matches = re.findall(
            rf"<\|begin_of_box\|>\s*:?({self.answer_pattern})\s*:?<\|end_of_box\|>",
            text,
            re.DOTALL,
        )
        if matches:
            return True, matches[-1]  # Always pick the last occurrence
        else:
            return False, text

    def _extract_tagged_answers(self, text):
        r"""Extract answers enclosed by <answer></answer>
        Seen in Tencent-Hunyuan or GLM-style models. Note that sometimes the right
        tag is not closed, under such cases we directly match till the end"""
        matches = re.findall(
            rf"<answer>\s*:?({self.answer_pattern})\s*:?(?:</answer>|$)",
            text,
            re.DOTALL,
        )
        if matches:
            return True, matches[-1]
        else:
            return False, text

    def _extract_general_answers(self, text):
        r"""Extract answers following some explicit answer hint
        The regex is produced by Gemini-2.5-pro after inspecting 745 cases produced
        by MiMo-7B-SFT"""
        matches = re.findall(
            rf"(?i)\b(?:answer|final answer)\s*:?\.?\s*(?:is)?\s*\**\s*({self.answer_pattern})(?=\b|\.|$|,|\s)",
            text,
            re.DOTALL,
        )
        if matches:
            return True, matches[-1]  # Always pick the last occurrence
        else:
            return False, text

    def _extract_markdown_bold_answers(self, text):
        r"""A relatively loose pattern that extract stuffs like
        **Answer: B**, yet we do NOT test whether the boldfaced content
        indeed seems like an answer."""
        matches = re.findall(
            rf"\*\*({self.answer_pattern})\*\*",
            text,
            re.DOTALL,
        )
        if matches:
            return True, matches[-1]
        else:
            return False, text

    def maybe_extract_enclosed_answers(self, text):
        stop_search = False
        answer = text
        for extract_fn in [
            self._extract_boxed_answers,
            self._extract_glm_boxed_answers,
            self._extract_tagged_answers,
            self._extract_general_answers,
            self._extract_markdown_bold_answers,
        ]:
            stop_search, answer = extract_fn(answer)
            if stop_search:
                break
        return answer

    def _process_one(self, text, skip_reasoning_parser: bool):
        thinking_content, result_content = self.reasoning_parser.extract_reasoning_content(
            model_output=text,
            request=None,  # TODO: perhaps create a dummy request
        )
        if result_content is None:
            # In this case, mostly the <think> left tag is NOT closed by the right </think>
            # tag, indicating that the output is truncated by length, do nothing
            # TODO: use logger to add warnings
            return DecomposedOutputs(text, "", "")
        if not skip_reasoning_parser:
            answer = self.maybe_extract_enclosed_answers(result_content)
        else:
            answer = result_content
        return DecomposedOutputs(thinking_content, result_content, answer)

    def process_mcq(self, text, skip_reasoning_parser: bool = False):
        r"""Answer processing method that determines whether a result is directly
        parsable, or might require LLM-as-a-Judge"""
        assert self.answer_type is AnswerType.MCQ
        decomposed_text = self._process_one(text, skip_reasoning_parser)
        answer = decomposed_text.answer_content.strip()
        requires_llm_as_a_judge = False
        if not answer:
            requires_llm_as_a_judge = True
            answer = decomposed_text.result_content.strip()
        else:
            answer = answer[0]
            if not answer in self.potential_answers:
                requires_llm_as_a_judge = True
        return requires_llm_as_a_judge, answer

    def process_nlg(self, text, skip_reasoning_parser: bool = True):
        decomposed_text = self._process_one(text, skip_reasoning_parser)
        answer = decomposed_text.result_content.strip()
        stop_search = False
        for extract_fn in [
            self._extract_glm_boxed_answers,
            self._extract_tagged_answers,
        ]:
            stop_search, answer = extract_fn(answer)
            if stop_search:
                break
        return False, answer.strip()

    def __call__(self, batch_text: List[str], verbose=False, skip_reasoning_parser: bool = False):
        decoded_texts = []
        for text in batch_text:
            decomposed_text = self._process_one(text, skip_reasoning_parser)
            answer = decomposed_text.answer_content.strip()
            if self.answer_type == AnswerType.MCQ:
                answer = answer[0]  # Ensure no redundant explanations during MCQ
            elif self.answer_type == AnswerType.YN:
                answer = answer[:3]  # Either yes or no
            if verbose:
                decoded_texts.append((answer, decomposed_text.to_json()))
            else:
                decoded_texts.append(answer)
        return decoded_texts
