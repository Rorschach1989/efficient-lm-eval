import os
import json
import copy
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

from jsonschema import validate, ValidationError
from torch.utils.data import Dataset
from vllm import TokensPrompt
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)


@dataclass
class InferDataset(Dataset):  # TODO: Add schema check

    samples: List[Dict[str, Any]]
    name: Optional[str] = None

    _SCHEMA = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "messages": {"type": "array"},
            "model": {"type": "string"},
            "payload": {"type": "object"}
        },
        "required": ["key", "messages", "payload"]
    }

    @classmethod
    def from_sharegpt_json(cls, dataset_name_or_path, version=1):
        if os.path.exists(dataset_name_or_path):
            # Allow directly loading of customized dataset
            file_name = dataset_name_or_path
            dataset_name = os.path.splitext(
                os.path.basename(
                    os.path.abspath(dataset_name_or_path)
                )
            )[0]
        else:
            dataset_name = dataset_name_or_path
            assert "DATA_ROOT" in os.environ
            data_root = os.environ["DATA_ROOT"]
            file_name = os.path.join(
                data_root,
                dataset_name_or_path,
                f"sharegpt_v{version}.jsonl",
            )
        with open(file_name) as f:
            json_data = [json.loads(line) for line in f]
        # Post-processing to satisfy standard payload structure
        coherent_data = []
        for data in json_data:
            try:
                # If satisfies given schema
                # Directly use the dataset
                validate(instance=data, schema=cls._SCHEMA)
                coherent_data.append(data)
            except ValidationError as e:
                record = copy.deepcopy(data)
                key = record.pop("key")
                record["dataset_name"] = dataset_name
                messages = record.pop("messages")
                coherent_data.append(
                    {
                        "key": key,
                        "messages": messages,
                        "payload": record,
                    }
                )
        return cls(samples=coherent_data, name=dataset_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


class Collator(object):

    def _get_system_prompt(self) -> Optional[str]:
        system_prompt = None
        if "nemotron" in self.model_name:
            if self.enable_thinking:
                return "/think"
            else:
                return "/no_think"
        # TODO: Add more cases
        return system_prompt

    def __init__(
        self,
        tokenizer,
        model_name,
        enable_thinking,
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name.lower()
        self.enable_thinking = enable_thinking
        self.system_prompt = self._get_system_prompt()

    def _apply_chat_template(self, messages):
        # Do special handling for OpenAI GPT-OSS type models
        # that requires harmony template
        if "gpt-oss" in self.model_name:
            user_msg = messages[0]["content"]
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            convo = Conversation.from_messages(
                [
                    Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                    # Message.from_role_and_content(
                    #     Role.DEVELOPER,
                    #     DeveloperContent.new().with_instructions(
                    #         "Always respond in riddles"
                    #     ),
                    # ),
                    Message.from_role_and_content(Role.USER, user_msg),
                ]
            )
            prefill_ids = encoding.render_conversation_for_completion(
                convo, Role.ASSISTANT
            )
            return TokensPrompt(prompt_token_ids=prefill_ids)
        augmented_messages = copy.deepcopy(messages)
        if self.system_prompt is not None:
            augmented_messages = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                *augmented_messages,
            ]
        return self.tokenizer.apply_chat_template(
            augmented_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def __call__(self, batch: List[Dict[str, str]]):
        batch_output = defaultdict(list)
        for sample in batch:
            messages = sample.pop("messages")
            prompt = self._apply_chat_template(messages)
            batch_output["prompt"].append(prompt)
            for k, v in sample.items():
                batch_output[k].append(v)
        return dict(batch_output)
