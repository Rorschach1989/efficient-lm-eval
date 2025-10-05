from enum import Enum
from typing import List

import pandas as pd
import pymc as pm
import arviz as az

from ...utils.irt_input_transform import IRTInput


class IRTMode(Enum):
    r"""Two canonical modeling mode for IRT tasks"""

    CALIBRATION = "calibration"
    SCORING = "scoring"


class BaseIRTModel(object):

    _OBS_KEY = "obs_score"

    _STUDENT = "student"
    _ITEM = "item"
    r""""""

    def __init__(self, *args, **kwargs):
        self._theta_std_prior = kwargs.pop(
            "theta_std_prior", "heterogeneous"
        )  # {homogenous, heterogeneous}
        self._b_std_prior = kwargs.pop(
            "b_std_prior", "heterogeneous"
        )  # {homogenous, heterogeneous}
        self._item_bank = {}

    def clear_item_bank(self):
        self._item_bank = {}

    def register_item_bank(self, az_summary: pd.DataFrame):
        for item_id, row in az_summary.iterrows():
            if item_id not in self._item_bank:
                self._item_bank[item_id] = row.to_dict()

    def retrieve_calibrated_items(self, item_keys):
        calibration_map = {f"{key}_calibrated": [] for key in self._get_item_nodes()}
        for key in item_keys:
            for prefix in self._get_item_nodes():
                item_id = f"{prefix}[{key}]"
                if item_id not in self._item_bank:
                    raise KeyError(f"item {key} not found in item bank")
                calibration_map[f"{prefix}_calibrated"].append(self._item_bank[item_id])
        return calibration_map

    def _get_item_nodes(self) -> List[str]:
        r"""Could be over-ridden in subclasses"""
        return ["b"]

    def _build_model(self, meta_data, mode: IRTMode, *args, **kwargs) -> pm.Model:
        r"""Build the IRT model"""
        raise NotImplementedError

    def _transform_inputs(self, y):
        r"""By default, do nothing"""
        return y

    def sample(self, inputs: IRTInput, mode: IRTMode = IRTMode.CALIBRATION, **kwargs):
        with pm.observe(
            self._build_model(
                inputs.meta_data,
                mode,
                **kwargs,  # TODO: bad design here
            ),
            {
                self._OBS_KEY: self._transform_inputs(inputs.y),
            },
        ) as irt_model:
            return pm.sample(**kwargs)

    def calibrate(self, inputs: IRTInput, **sampling_kwargs):
        idata = self.sample(inputs, mode=IRTMode.CALIBRATION, **sampling_kwargs)
        item_nodes = self._get_item_nodes()
        az_summary = az.summary(
            idata, var_names=item_nodes
        )  # Records everything produced by az
        self.register_item_bank(az_summary)
        return idata

    def score(self, inputs: IRTInput, **sampling_kwargs):
        item_keys = inputs.meta_data[self._ITEM]
        calibration_map = self.retrieve_calibrated_items(item_keys)
        sampling_kwargs = {
            **calibration_map,
            **sampling_kwargs,
        }
        idata = self.sample(inputs, mode=IRTMode.SCORING, **sampling_kwargs)
        return idata


def _get_theta_default(
    theta_std_prior: str,
    theta_dim: str = BaseIRTModel._STUDENT,
    **kwargs,
):
    theta_offset = pm.Normal(
        "theta_offset",
        mu=0,
        sigma=1.0,
        dims=theta_dim,
    )
    if theta_std_prior == "heterogeneous":
        theta_std = pm.HalfNormal(name="theta_std")
        theta = pm.Deterministic(
            "theta",
            theta_offset * theta_std,
            dims=theta_dim,
        )
    else:
        theta = pm.Deterministic(
            "theta",
            theta_offset,
            dims=theta_dim,
        )
    return theta


def _get_b_default(
    b_std_prior: str,
    b_dim: str = BaseIRTModel._ITEM,
    mode: IRTMode = IRTMode.CALIBRATION,
    **kwargs,
):
    if mode is IRTMode.CALIBRATION:
        b_offset = pm.Normal(
            "b_offset",
            mu=0,
            sigma=1.0,
            dims=b_dim,
        )
        if b_std_prior == "heterogeneous":
            b_std = pm.HalfNormal(name="b_std")
            b = pm.Deterministic(
                "b",
                b_offset * b_std,
                dims=b_dim,
            )
        else:
            b = pm.Deterministic(
                "b",
                b_offset,
                dims=b_dim,
            )
    else:
        assert "b_calibrated" in kwargs
        b_calibrated = kwargs.pop("b_calibrated")
        b = pm.Deterministic(
            "b",
            b_calibrated,
            dims=b_dim,
        )
    return b


def _get_a_default(
    a_dim: str = BaseIRTModel._ITEM,
    mode: IRTMode = IRTMode.CALIBRATION,
    **kwargs,
):
    if mode is IRTMode.CALIBRATION:
        return pm.LogNormal(
            "a",
            mu=0,
            sigma=0.5,
            dims=a_dim,
        )
    else:
        assert "a_calibrated" in kwargs
        a_calibrated = kwargs.pop("a_calibrated")
        a = pm.Deterministic(
            "a",
            a_calibrated,
            dims=a_dim,
        )
    return a
