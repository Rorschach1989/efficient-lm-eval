import os
from dataclasses import dataclass

from swift.utils import JsonlWriter

from .dataset import InferDataset


@dataclass
class TaskRunner(object):
    r"""Baseclass for task-specific runners."""

    dataset: InferDataset
    model_name_or_path: str = None
    enable_thinking: bool = False
    thinking_budget: int = 512
    max_tokens: int = 16384
    _MIXED: str = "MIXED"
    version: int = 1

    def _extract_model_name(self):
        if self.model_name_or_path is None:
            return None
        return self.model_name_or_path.split("/")[-1]

    def _get_output_file_path(self):
        assert "IRT_INFER_ROOT" in os.environ
        irt_infer_root = os.environ["IRT_INFER_ROOT"]
        dataset_name = self.dataset.name or self._MIXED
        os.makedirs(
            os.path.join(irt_infer_root, dataset_name),
            exist_ok=True
        )
        model_name = self._extract_model_name() or self._MIXED
        thinking_tag = "_think" if self.enable_thinking else ""
        model_name += thinking_tag
        return os.path.join(
            irt_infer_root,
            dataset_name,
            f"{model_name}_v{self.version}.jsonl",
        )

    def __post_init__(self):
        self.json_writer = JsonlWriter(
            fpath=self._get_output_file_path(),
        )

    def run_task(self):
        raise NotImplementedError