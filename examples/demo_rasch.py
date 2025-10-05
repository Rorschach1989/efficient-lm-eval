import os

import pandas as pd
import arviz as az

from lm_irt_eval.utils.logging import create_logger
from lm_irt_eval.evaluator.irt import Rasch1PLModel
from lm_irt_eval.utils.irt_input_transform import IrtInputHandler


def get_example_data_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        os.path.dirname(current_dir),
        "data"
    )


def main():
    logger = create_logger()
    data_root = get_example_data_root()
    # mmlu_calibration_file = "/Users/zekkarorschach/Projects/irt_eval/efficient-lm-eval/data/examples/lego_irt/mmlu_calibration.jsonl"
    # mmlu_scoring_file = "/Users/zekkarorschach/Projects/irt_eval/efficient-lm-eval/data/examples/lego_irt/mmlu_scoring.jsonl"
    mmlu_calibration_file = os.path.join(
        data_root,
        "examples",
        "lego_irt",
        "mmlu_calibration.jsonl",
    )
    mmlu_scoring_file = os.path.join(
        data_root,
        "examples",
        "lego_irt",
        "mmlu_scoring.jsonl",
    )
    calibration_df = pd.read_json(mmlu_calibration_file, lines=True).rename(
        columns={"request_model": "model_id", "dataset": "benchmark"}
    )
    scoring_df = pd.read_json(mmlu_scoring_file, lines=True).rename(
        columns={"request_model": "model_id", "dataset": "benchmark"}
    )
    calibration_inputs_handler = IrtInputHandler(calibration_df)
    scoring_inputs_handler = IrtInputHandler(scoring_df)
    model = Rasch1PLModel()
    calibration_outputs = model.calibrate(
        inputs=calibration_inputs_handler.extract_metric_to_irt_inputs("is_correct"),
        draws=2000,
        tune=1500,
        chains=4,
        target_accept=0.9,
        nuts_sampler="blackjax",
    )
    # Do some viz over calibration outputs
    theta_est_calibration = az.summary(calibration_outputs, var_names=['theta'])['mean']
    logger.info(theta_est_calibration.sort_values())
    scoring_outputs = model.score(
        inputs=scoring_inputs_handler.extract_metric_to_irt_inputs("is_correct"),
        draws=2000,
        tune=1500,
        chains=4,
        target_accept=0.9,
        nuts_sampler="blackjax",
    )
    theta_est_scoring = az.summary(scoring_outputs, var_names=['theta'])['mean']
    logger.info(theta_est_scoring.sort_values())


if __name__ == "__main__":
    main()
