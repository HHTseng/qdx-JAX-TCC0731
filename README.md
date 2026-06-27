<table>
  <tr>
    <td>
      <img src="images/qdx_logo_wordart.png" alt="overview" width="200"/>
    </td>
    <td>
      <h1>QEC AI-discovery with JAX ⚛️🤖🚀</h1>
    </td>
  </tr>
</table>


[![Paper](https://img.shields.io/badge/npj_qi-10_126_(2024)-b31b1b.svg)](https://www.nature.com/articles/s41534-024-00920-y)  <a href="https://colab.research.google.com/drive/1nU9Xivfms_wXrJmv0F6uFz4_DOWoryhg?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 

Code repository for the paper "Simultaneous Discovery of Quantum Error Correction Codes and Encoders with a Noise-Aware Reinforcement Learning Agent" by *Jan Olle, Remmy Zen, Matteo Puviani and Florian Marquardt*.

## Description
This library can be used to train Reinforcement Learning (RL) agents to codiscover quantum error correction (QEC) codes and their encoding circuits *from scratch, without any additional domain knowledge* except how many errors are not detected given the quantum circuit it has built.

The RL agent can be made *noise-aware*, meaning that it learns to produce encoding strategies simultaneously for a range of noise models, making it applicable in very broad situations. 

<img src="images/overview.png" alt="overview" width="800"/>

The whole RL algorithm, including the Clifford simulations of the quantum circuits, are implemented in Jax. This enables parallelized training of multiple agents on vectorized environments in a single GPU. Our implementation builds upon [PureJaxRL](https://github.com/luchris429/purejaxrl?tab=readme-ov-file).

## Installation

QDX can be installed by:

1. Cloning the repository

``` bash
git clone https://github.com/jolle-ag/qdx.git
cd qdx
```

2. Installing requirements
``` bash
pip install -r requirements.txt
```

### GPU compatibility:

To install [JAX with NVIDIA GPU support](https://github.com/jax-ml/jax), use:

```
# CUDA 12 installation
pip install -U "jax[cuda12]"
```



## Usage Example

We include a [demo](https://github.com/jolle-ag/qdx/blob/main/notebooks/demo.ipynb) jupyter notebook for two different situations: [[7,1,3]] code discovery in a fixed symmetric depolarizing noise channel and noise-aware [[6,1]] code discovery in a biased noise channel.

 <a href="https://colab.research.google.com/drive/1nU9Xivfms_wXrJmv0F6uFz4_DOWoryhg?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 

## GNN-QDX v1

The variable-size policy is enabled with `"MODEL": "GNN"` on the
`STANDARD` environment. `CodeFinder` then uses `GraphCodeDiscovery`,
while PPO, KL reward calculation, and tableau transitions stay on the
existing code path.

The bucket capacity is configured independently of the current task:

```python
config.update(
    {
        "MODEL": "GNN",
        "GNN_N_MAX": 7,
        "GNN_STABILIZERS_MAX": 6,
        "GNN_HARDWARE_EDGES_MAX": 42,
        "GNN_HIDDEN_DIM": 64,
        "GNN_RELATION_DIM": 8,
        "GNN_GATE_DIM": 8,
        "GNN_NUM_LAYERS": 3,
    }
)
```

Tasks in the same bucket and with the same ordered gate set share model
parameters. The multi-task demo trains sequentially on
`N=(6,7,8,9), K=(1,2)`, validates on `N=(5,...,10), K=(1,2)`, and
uses `MAX_STEPS=50`. Its default 2,000,000-step budget is divided
across all training task visits.

Run the demo, its shape-only dry run, and the executable tests with:

```bash
conda run -n qdx python examples/demo_multitask_nk.py
conda run -n qdx python examples/demo_multitask_nk.py --dry-run
conda run -n qdx python -m unittest tests.test_gnn_qdx -v
```

Training writes the checkpoint, training history, run configuration, and
validation results under `results/demo_multitask_nk/`.

 ## License

The code in this repository is released under the MIT License.

## Citation
``` bib
@article{olle_simultaneous_2024,
  title={Simultaneous Discovery of Quantum Error Correction Codes and Encoders with a Noise-Aware Reinforcement Learning Agent},
  author={Olle, Jan and Zen, Remmy and Puviani, Matteo and Marquardt, Florian},
  url = {https://www.nature.com/articles/s41534-024-00920-y},
  journal={npj Quantum Information 10, Article number: 126 (2024)},
  urldate = {2024-12-03},
  publisher = {npj Quantum Information},
  month = dec,
  year = {2024},
  note = {arXiv:2311.04750 [quant-ph]},
}
```
