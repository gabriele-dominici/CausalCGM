# Causal Concept Graph Models: Beyond Causal Opacity in Deep Learning

Official implementation of the paper [**"Causal Concept Graph Models: Beyond Causal Opacity in Deep Learning"**](https://arxiv.org/abs/2405.16507) accepted at **ICLR 2025**.


## Setup

### Prerequisites

- Python 3.9 or higher

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CausalCGM.git
cd CausalCGM
```

2. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Place the entire dataset directory in `./datasets/celeba/`
3. Preprocess the images and extract embeddings:
```bash
python3 preprocess_data.py
```

## Running Experiments

```bash
python3 run.py
```


## Repository Structure

### Core Components

- **CausalCGM** (`causalcgm/causalcgm.py`):
  - `CausalConceptGraphLayer`: Core implementation of the CausalCGM layer
  - `CausalCGM`: Complete model including encoder and training procedures

- **DAGMA** (`causalcgm/dagma.py`):
  - Modified implementation of [DAGMA](https://arxiv.org/abs/2209.08037) for DAG learning
  - Can be replaced with other DAG-enforcing losses
  - Future support planned for [COSMO loss](https://arxiv.org/abs/2309.08406)

- **Baselines** (`causalcgm/baselines.py`):
  - Implementation of Concept Bottleneck Model
  - Implementation of Concept Embedding Model

## Key Configuration Parameters

### Model Parameters
- `embedding_size` (in `run.py`):
  - Higher values increase model expressivity
  - Lower values improve intervention effectiveness

### Intervention Parameters
- `index_perturb`: Parent nodes for intervention
- `index_block`: Child nodes of perturbed parents (must be label ancestors)

### Graph Learning Parameters
- `random_start`:
  - `True`: Random graph initialization
  - `False`: Uses data conditional entropy
- `to_check`: List of edge pairs with unknown directions (e.g., `[(1,2)]`)

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{dominici2025causalconceptgraphmodels,
      title={Causal Concept Graph Models: Beyond Causal Opacity in Deep Learning}, 
      author={Gabriele Dominici and Pietro Barbiero and Mateo Espinosa Zarlenga and Alberto Termine and Martin Gjoreski and Giuseppe Marra and Marc Langheinrich},
      year={2025},
      eprint={2405.16507},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.16507}, 
}
```