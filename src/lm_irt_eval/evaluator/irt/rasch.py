import pymc as pm

from .base import (
    IRTMode,
    BaseIRTModel,
    _get_theta_default,
    _get_b_default
)


class Rasch1PLModel(BaseIRTModel):
    r"""TODO: improve the design of priors"""

    # def _get_theta(self):
    #     theta_offset = pm.Normal(
    #         "theta_offset",
    #         mu=0,
    #         sigma=1.,
    #         dims=self._STUDENT,
    #     )
    #     if self._theta_std_prior == "heterogeneous":
    #         theta_std = pm.HalfNormal(name="theta_std")
    #         theta = pm.Deterministic(
    #             "theta",
    #             theta_offset * theta_std,
    #             dims=self._STUDENT,
    #         )
    #     else:
    #         theta = pm.Deterministic(
    #             "theta",
    #             theta_offset,
    #             dims=self._STUDENT,
    #         )
    #     return theta

    # def _get_b(self):
    #     b_offset = pm.Normal(
    #         "b_offset",
    #         mu=0,
    #         sigma=1.,
    #         dims=self._ITEM,
    #     )
    #     if self._b_std_prior == "heterogeneous":
    #         b_std = pm.HalfNormal(name="b_std")
    #         b = pm.Deterministic(
    #             "b",
    #             b_offset * b_std,
    #             dims=self._ITEM,
    #         )
    #     else:
    #         b = pm.Deterministic(
    #             "b",
    #             b_offset,
    #             dims=self._ITEM,
    #         )
    #     return b

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
            prob = pm.math.sigmoid(
                theta[:, None] - b[None, :]
            )
            y = pm.Bernoulli(
                self._OBS_KEY,
                p=prob,
                dims=(self._STUDENT, self._ITEM),
            )
            return irt_model
