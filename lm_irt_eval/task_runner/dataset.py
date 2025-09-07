import os
import json
import copy
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

from torch.utils.data import Dataset


@dataclass
class InferDataset(Dataset):  # TODO: Add schema check

    samples: List[Dict[str, Any]]
    name: Optional[str] = None

    @classmethod
    def from_sharegpt_json(cls, dataset_name, version=1):
        assert "DATA_ROOT" in os.environ
        data_root = os.environ["DATA_ROOT"]
        file_name = os.path.join(
            data_root,
            dataset_name,
            f"sharegpt_v{version}.jsonl",
        )
        with open(file_name) as f:
            json_data = [json.loads(line) for line in f]
        return cls(samples=json_data, name=dataset_name)

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
