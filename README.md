## Installation

We use ``uv`` as the package management tool. To build an applicable environment:
```shell
git clone https://github.com/Rorschach1989/efficient-lm-eval.git
cd efficient-lm-eval
uv venv --python 3.12 --seed
uv pip install -e .
```

## Examples

Examples are in the [examples](examples) directory, which includes demos for the following models:
- [Rasch model](examples/demo_rasch.py): This serves as the vanilla method in IRT
- [LEGO-IRT-CM](examples/demo_lego_irt_cm.py): This method handles continuous metrics that extends binary Rasch model
- [LEGO-IRT-MM](examples/demo_lego_irt_mm.py): This method goes beyond ``LEGO-IRT-CM`` through integrating multiple metrics
- [LEGO-IRT-MB](examples/demo_lego_irt_mb.py): This method efficiently combines multiple benchmarks