import copy
from typing import Dict, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IRTInput(object):
    y: np.ndarray
    meta_data: Dict[str, Any]


@dataclass
class IrtInputHandler(object):

    raw_data: pd.DataFrame
    _STUDENT: str = "student"
    _ITEM: str = "item"
    _METRIC: str = "metric"

    @staticmethod
    def _get_default_column_schema():
        return [
            "key",
            "model_id",
            "metric",
        ]

    def __post_init__(self):
        # Do schema check
        for column in self._get_default_column_schema():
            assert column in self.raw_data.columns
        # Do sorting
        self.raw_data = self.raw_data.sort_values(
            ["model_id", "key"],
            ascending=True,
        ).reset_index(drop=True)
        self.model_ids = self.raw_data["model_id"].unique().tolist()
        self.keys = self.raw_data["key"].unique().tolist()
        self.metric_list = set(self.raw_data["metric"].values[0].keys())
        for metric in self.raw_data["metric"]:
            assert set(metric.keys()) == self.metric_list

    @property
    def n_models(self):
        return len(self.model_ids)

    @property
    def n_items(self):
        return len(self.keys)

    def extract_metric(self, metric_id) -> np.ndarray:
        assert metric_id in self.metric_list
        metric_values_1d = (
            self.raw_data["metric"].apply(lambda x: x.get(metric_id)).values
        )
        metric_values_2d = metric_values_1d.reshape((self.n_models, self.n_items))
        return metric_values_2d

    def extract_metric_to_irt_inputs(self, metric_id) -> IRTInput:
        return IRTInput(
            y=self.extract_metric(metric_id),
            meta_data={
                self._STUDENT: copy.deepcopy(self.model_ids),
                self._ITEM: copy.deepcopy(self.keys),
            },
        )

    def extract_all_metrics_to_irt_inputs(self):
        metric_list = sorted(self.metric_list)
        meta_data = {
            self._STUDENT: copy.deepcopy(self.model_ids),
            self._ITEM: copy.deepcopy(self.keys),
            self._METRIC: copy.deepcopy(metric_list),
        }
        metrics = [self.extract_metric(metric_id) for metric_id in metric_list]
        metrics = np.stack(metrics, axis=-1)
        return IRTInput(
            y=metrics,
            meta_data=meta_data,
        )


@dataclass
class MultiBenchmarkIRTInputHandler(IrtInputHandler):

    _BENCHMARK: str = "benchmark"

    @staticmethod
    def _get_default_column_schema():
        return [
            "key",
            "model_id",
            "metric",
            "benchmark",
        ]

    def __post_init__(self):
        super(MultiBenchmarkIRTInputHandler, self).__post_init__()
        # TODO: Maybe we should check inter-benchmark key uniqueness
        self.benchmarks = sorted(self.raw_data["benchmark"].unique().tolist())
        self.benchmark_indices = (
            self.raw_data.drop_duplicates(["key", "benchmark"])["benchmark"]
            .apply(lambda x: self.benchmarks.index(x))
            .values
        )  # Should be a numpy array for compatibility with pymc

    def extract_metric_to_irt_inputs(self, metric_id) -> IRTInput:
        return IRTInput(
            y=self.extract_metric(metric_id),
            meta_data={
                self._STUDENT: copy.deepcopy(self.model_ids),
                self._ITEM: copy.deepcopy(self.keys),
                self._BENCHMARK: copy.deepcopy(self.benchmarks),
                "benchmark_indices": copy.deepcopy(self.benchmark_indices),
            },
        )

    def extract_all_metrics_to_irt_inputs(self):
        metric_list = sorted(self.metric_list)
        meta_data = {
            self._STUDENT: copy.deepcopy(self.model_ids),
            self._ITEM: copy.deepcopy(self.keys),
            self._METRIC: copy.deepcopy(metric_list),
            self._BENCHMARK: copy.deepcopy(self.benchmarks),
            "benchmark_indices": copy.deepcopy(self.benchmark_indices),
        }
        metrics = [self.extract_metric(metric_id) for metric_id in metric_list]
        metrics = np.stack(metrics, axis=-1)
        return IRTInput(
            y=metrics,
            meta_data=meta_data,
        )
