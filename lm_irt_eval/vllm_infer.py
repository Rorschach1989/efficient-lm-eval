import argparse

from .task_runner import InferDataset, VLLMTaskRunner


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sampling_n", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, required=False, default=16384)
    parser.add_argument("--max_tokens", type=int, required=False, default=1024)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # Configure dataset
    infer_dataset = InferDataset.from_sharegpt_json(args.dataset)
    # Configure runner
    task_runner = VLLMTaskRunner(
        dataset=infer_dataset,
        model_name_or_path=args.model,
        enable_thinking=args.enable_thinking,
        sampling_n=args.sampling_n,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        trust_remote_code=args.trust_remote_code,
    )
    task_runner.run_task()


if __name__ == "__main__":
    main()
