from typing import List

import pymc as pm
import pytensor.tensor as pt

from .base import (
    IRTMode,
    BaseIRTModel,
    _get_theta_default,
    _get_b_default,
    _get_a_default
)


class ContinuousMetricIRTModel(BaseIRTModel):
    r"""LEGOIRT-CM"""

    def _transform_inputs(self, y):
        return pm.logit(pm.math.clip(y, 1e-8, 1.0 - 1e-8))

    def _build_model(self, meta_data, mode: IRTMode, *args, **kwargs) -> pm.Model:
        coords = {
            self._STUDENT: meta_data[self._STUDENT],
            self._ITEM: meta_data[self._ITEM],
        }
        with pm.Model(coords=coords) as irt_model:
            theta = _get_theta_default(
                self._theta_std_prior,
            )
            b = _get_b_default(self._b_std_prior)
            a = _get_a_default()
            logit_mean = a[None, :] * theta[:, None] - b[None, :]
            logit_std = pm.HalfNormal(
                "logit_std",
            )
            logit_y = pm.Normal(
                self._OBS_KEY,
                mu=logit_mean,
                sigma=logit_std,
                dims=(self._STUDENT, self._ITEM),
            )
            return irt_model


class MultiMetricIRTModel(ContinuousMetricIRTModel):
    r"""LEGOIRT-MM"""

    _METRIC = "metric"

    def _get_item_nodes(self) -> List[str]:
        r"""Could be over-ridden in subclasses"""
        return ["a", "b"]

    def _build_model(self, meta_data, mode: IRTMode, *args, **kwargs) -> pm.Model:
        coords = {
            self._STUDENT: meta_data[self._STUDENT],
            self._ITEM: meta_data[self._ITEM],
            self._METRIC: meta_data[self._METRIC],
        }
        n = len(coords[self._METRIC])
        with pm.Model(coords=coords) as irt_model:
            psi = pm.Normal("psi", mu=0, sigma=1, dims=self._STUDENT)
            chol, corr, stds = pm.LKJCholeskyCov(
                'chol_zeta',
                eta=2.0,
                n=n,
                sd_dist=pm.Exponential.dist(1.0),
                compute_corr=True
            )
            zeta_raw = pm.Normal("zeta_raw", 0.0, 1.0, dims=(self._STUDENT, self._METRIC))
            zeta = pm.Deterministic("zeta", pt.dot(zeta_raw, chol.T))
            theta = pm.Deterministic("theta", psi[:, None, None] + zeta[:, None, :])
            b = _get_b_default(self._b_std_prior)
            a = _get_a_default()
            logit_mean = a[None, :, None] * theta - b[None, :, None]
            logit_std = pm.HalfNormal(
                "logit_std",
            )
            logit_y = pm.Normal(
                self._OBS_KEY,
                mu=logit_mean,
                sigma=logit_std,
                dims=(self._STUDENT, self._ITEM, self._METRIC),
            )
            return irt_model


class MultiBenchmarkIRTModel(BaseIRTModel):
    r"""LEGOIRT-MB"""

    _BENCHMARK = "benchmark"

    def _get_item_nodes(self) -> List[str]:
        r"""Could be over-ridden in subclasses"""
        return ["a", "b"]

    def _build_model(self, meta_data, mode: IRTMode, *args, **kwargs) -> pm.Model:
        coords = {
            self._STUDENT: meta_data[self._STUDENT],
            self._ITEM: meta_data[self._ITEM],
            self._BENCHMARK: meta_data[self._BENCHMARK],
        }
        n = len(coords[self._BENCHMARK])
        benchmark_indices = meta_data["benchmark_indices"]
        with pm.Model(coords=coords) as irt_model:
            psi = pm.Normal("psi", mu=0, sigma=1, dims=self._STUDENT)
            chol, corr, stds = pm.LKJCholeskyCov(
                'chol_zeta',
                eta=2.0,
                n=n,
                sd_dist=pm.Exponential.dist(1.0),
                compute_corr=True
            )
            delta_raw = pm.Normal(
                "delta_raw",
                0.0,
                1.0,
                dims=(self._STUDENT, self._BENCHMARK)
            )
            delta = pm.Deterministic("zeta", pt.dot(delta_raw, chol.T))
            theta = pm.Deterministic("theta", psi[:, None] + delta)
            theta_items = theta[:, benchmark_indices]
            a = _get_a_default()
            b = _get_b_default(self._b_std_prior)
            prob = pm.math.sigmoid(a[None, :] * theta_items - b[None, :])
            y = pm.Bernoulli(
                self._OBS_KEY,
                p=prob,
                dims=(self._STUDENT, self._ITEM),
            )
            return irt_model
