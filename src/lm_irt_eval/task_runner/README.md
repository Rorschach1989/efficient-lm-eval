# A simplified task runner that do LLM inference based on either vLLM or direct api call

## vLLM mode
The entrypoint is defined at [vllm_infer](../vllm_infer.py)  
Alternatively, when the task is standard, we recommend using better data-parallel modules such as ``swift infer`` in ``ms-swift``.
The following solution better handles task-specific flexibility.

### Data preparation

Organize data as in those in the [data](../../../data) directory, in ``ShareGPT`` format  
The json schema should be defined as follows (in terms of ``jsonschema`` verification format):
```python
_SCHEMA = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "messages": {"type": "array"},
        "model": {"type": "string"},
        "payload": {"type": "object"},
    },
    "required": ["key", "messages", "payload"],
}
```

### Run inference
Refer to the following shell script example, replacing those ``path_to_xxx`` arguments into concrete paths.
```shell
# For IO
export IRT_INFER_ROOT=path_to_your_infer_path_root
export DATA_ROOT=path_to_your_data_path_root
# Optional configs
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ./logs

dataset=wmt20  # Or other datasets that are organized into suitable formats

model_ids=(
  "paths_to_your_models"
)

for model_id in "${model_ids[@]}"
do
    model_name=$(basename "$model_id")
    echo "Evaluating model: $model_name over dataset $dataset"
    
    # Sometimes specific models might required tailored environments
    python_exec=path_to_your_python_interpreter
    max_model_len=32768
    max_tokens=16384

    $python_exec -m lm_irt_eval.vllm_infer \
      --dataset "$dataset" \
      --model "$model_id" \
      --max_model_len "$max_model_len" \
      --max_tokens "$max_tokens" \
      --trust_remote_code \
      2>&1 | tee -a "./logs/${dataset}_${model_name}.log"
    echo "Finished evaluating $model_name"
    echo "---------------------------------"
done
```

## API mode
The entrypoint is defined at [api_infer](../api_infer.py)

### Data preparation
In addition to the schema defined above, to allow better concurrency over models, add one extra ``request_model`` field when using
API mode.

### Run inference
Refer to the following shell script. Both ``openai``-style and ``bearer``-style are supported. Pass base urls through environmental variables
```shell
export API_KEY=
export OAI_API_URL=
export BEARER_API_URL=
export IRT_INFER_ROOT=path_to_your_data_path_root

ulimit -n 4096
python=path_to_your_python_interpreter

$python -m lm_irt_eval.api_infer \
	--input_file path_to_your_input_jsonl_file \
	--max_workers 50 
```
