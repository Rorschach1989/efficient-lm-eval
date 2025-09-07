import json
import argparse

from .task_runner import InferDataset, APITaskRunner


def get_args():
    parser = argparse.ArgumentParser()
    # In API mode, explicitly assemble a mixed dataset before running
    # As this would be much more efficient
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--use_openai_client", action="store_true" )
    parser.add_argument("--max_workers", type=int, default=50)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.input_file) as fr:
        all_samples = [json.loads(line) for line in fr]
    infer_dataset = InferDataset(all_samples)
    runner = APITaskRunner(
        infer_dataset,
        use_openai_client=args.use_openai_client,
        max_workers=args.max_workers,
    )
    runner.run_task()


if __name__ == "__main__":
    main()
