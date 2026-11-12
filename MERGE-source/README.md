# ECDT-R1

Gene expression prediction from frozen image features and a hierarchical tissue graph. This package contains the final ECDT-R1 architecture developed in the MERGE workspace.

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Inputs

The model predicts 250 expression values per real spot from frozen 1024-dimensional Phikon-v2 features. Raw whole-slide images are not direct model inputs. Encoder weights, graph caches, expression targets and model checkpoints are supplied separately.

Each graph is stored as an NPZ file with:

| Array | Shape |
|---|---|
| `x` | nodes × 1024, raw frozen features |
| `coords` | nodes × 2 |
| `is_real` | nodes, Boolean |
| `spatial_membership` | real spots × 5 |
| `morphology_membership` | real spots × 5 |
| Seven relation arrays | 2 × edges, integer node indices |

Relation names are `spatial_local`, `spatial_cluster_star`, `morphology_cluster_star`, `spatial_representative_shortcut`, `morphology_representative_shortcut`, `cross_representative_shortcut`, and `pseudo_context`. Optional arrays include `barcodes`, `source_row_indices`, and `raw_panel_gene_names`.

Existing graph-cache NPZ files from this model can be read directly. To build a graph from extracted features, provide `x`, `coords`, `is_real`, and `barcodes`, with real nodes first and pseudo nodes afterwards:

```bash
python prepare_graph.py --features features.npz --output graph.npz
```

## Prediction

```bash
python predict.py \
  --checkpoint /path/to/best.pt \
  --graph /path/to/slide.npz \
  --normalization /path/to/normalization.npz \
  --output prediction.npz \
  --device cpu
```

Both raw tensor checkpoints and existing wrappers containing a `model` state dictionary are supported, with strict loading. The normalization file must match the model's training data; it contains `mean` and `std`, each with 1024 entries. Features are normalized for the node stem, while relation evidence is calculated from raw features and original coordinates.

The output NPZ contains `prediction` (real spots × 250), real-spot coordinates and, when supplied, barcodes and gene names. Input gene order and spot order must be preserved.

```python
from ecdt import load_model
from ecdt.data import load_graph, load_normalization, prepare_graph
import torch

model = load_model('/path/to/best.pt', device='cpu')
graph = load_graph('/path/to/slide.npz')
data = prepare_graph(graph, load_normalization('/path/to/normalization.npz'))
with torch.inference_mode():
    expression = model(**data)['prediction']
```

## Training

Provide explicit, disjoint training and validation graph files. Expression targets are NPY arrays named after each graph, with 250 columns in the same gene order. Targets containing original full-slide rows use the graph's `source_row_indices` to select real spots.

```bash
python train.py \
  --train-graphs data/train_a.npz data/train_b.npz \
  --val-graphs data/val_a.npz \
  --target-root data/expression \
  --output runs/model
```

Default training runs 400 epochs with AdamW, learning rate 2e-4, weight decay 1e-4, gradient clipping at 1, and real-spot MSE. Normalization is fitted only on training graphs. CUDA training uses BF16 autocast with the model's FP32-sensitive operations; CPU runs use FP32.

The best validation-MSE model is saved as `model.ckpt`, the final model as `final.ckpt`, and feature normalization as `normalization.npz`. New checkpoints contain model tensors only. This entry point provides a fixed-length single-run workflow; supplied historical weights are supported for inference.

## Acknowledgement

The project was developed using [MERGE](https://github.com/ags3927/MERGE). See `LICENSE` and `NOTICE`.
