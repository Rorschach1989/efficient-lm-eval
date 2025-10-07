<div align="center">
<h1> LLM Evaluations with Item Response Theory(IRT)</h1>
<h3>Toward a unified framework for data-efficient evaluation of large language models
</h3>

Lele Liao<sup>1</sup>, Qile Zhang<sup>2</sup>, Ruofan Wu<sup>1</sup>, Guanhua Fang<sup>1</sup>

<sup>1</sup> Fudan University  
<sup>2</sup> Shanghai Jiao Tong University

</div>

[![arXiv](https://img.shields.io/badge/arXiv-2510.04051-04051.svg)](https://arxiv.org/abs/2510.04051)

## Installation

We use ``uv`` as the package management tool. To build an applicable environment:
```shell
git clone https://github.com/Rorschach1989/efficient-lm-eval.git
cd efficient-lm-eval
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e .
```

## Examples

Examples are in the [examples](examples) directory, which includes demos for the following models:
- [Rasch model](examples/demo_rasch.py): This serves as the vanilla method in IRT
- [LEGO-IRT-CM](examples/demo_lego_irt_cm.py): This method handles continuous metrics that extends binary Rasch model
- [LEGO-IRT-MM](examples/demo_lego_irt_mm.py): This method goes beyond ``LEGO-IRT-CM`` through integrating multiple metrics
- [LEGO-IRT-MB](examples/demo_lego_irt_mb.py): This method efficiently combines multiple benchmarks

## Citation
If you find this repository helpful, please consider giving a star ⭐ and a citation

```bib
@misc{liao2025unifiedframeworkdataefficientevaluation,
      title={Toward a unified framework for data-efficient evaluation of large language models}, 
      author={Lele Liao and Qile Zhang and Ruofan Wu and Guanhua Fang},
      year={2025},
      eprint={2510.04051},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.04051}, 
}
```