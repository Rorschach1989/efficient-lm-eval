import argparse

import numpy as np
import pandas as pd

from .evaluator.metric_calculator import MetricCalculator


def get_args():
    parser = argparse.ArgumentParser()
    # In API mode, explicitly assemble a mixed dataset before running
    # As this would be much more efficient
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--wait_time", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument(
        "--llm_as_a_judge_model",
        type=str,
        default="openai/gpt-5-2025-08-07"
    )
    args = parser.parse_args()
    return args


gt_map = {
    "MMLU": lambda x: x["true_answer"],
    "CSQA": lambda x: x["GT"],
    "XSUM": lambda x: x["true_answer"],
    "Ceval": lambda x: x["true_answer"],
}


def main():
    args = get_args()
    df = pd.read_json(args.input_file, lines=True)
    df["dataset_name"] = df["payload"].apply(lambda x: x["dataset_name"])
    df["groud_truth"] = df.apply(
        lambda row: gt_map[row["dataset_name"]](row["payload"]),
        axis=1
    )
    metric_calculator = MetricCalculator(
        answer_support="mcq@8",
        output_path=args.output_file,
        llm_as_a_judge_model=args.llm_as_a_judge_model,
        retries=args.retries,
        wait_time=args.wait_time,
        max_workers=args.max_workers,
    )
    for (request_model, dataset_name), sub_df in df.groupby(["request_model", "dataset_name"]):
        batch = []
        for record in sub_df.to_dict(orient="records"):
            batch.append(
                {
                    "key": record["key"],
                    "candidate": record["infer_output"],
                    "ground_truth": record["groud_truth"],
                    "payload": record["payload"],
                    "request_model": record["request_model"],
                    "query_model": record["query_model"],
                }
            )
        if dataset_name in {"XSUM"}:
            batch = metric_calculator.batch_process_nlg(batch)
            for m in batch[0]["metric"]:
                ms = [b["metric"][m] for b in batch]
                print(f"Request model {request_model}@{dataset_name} under metric {m}: {np.mean(ms)}")
        else:
            batch = metric_calculator.batch_process_mcq(batch)
            batch_size = len(batch)
            has_judgement = sum("metric" in item for item in batch)
            correct = sum(item["metric"]["is_correct"] if "metric" in item else 0 for item in batch)
            acc = correct / has_judgement
            print(f"Request model {request_model}@{dataset_name}: "
                  f"Batch size = {batch_size},"
                  f" has_judgement = {has_judgement}, acc = {acc}")


if __name__ == "__main__":
    main()
