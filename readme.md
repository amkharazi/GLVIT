# How to Run

## Step 1: Download Datasets
Before running any experiments, download all required datasets:

```bash
python download_dataset.py
```

## Step 2: Run Tests
To run a specific experiment, use one of the provided scripts from the `test` directory. A typical example:

```bash
python TEST_basic.py \
    --TEST_ID TEST_ID001 \
    --dataset cifar10 \
    --batch_size 32 --n_epoch 10 --image_size 32 \
    --train_size 40000 --patch_size 4 --num_classes 10 \
    --dim 64 --depth 6 --heads 8 --mlp_dim 128
```

> Make sure you use a unique `--TEST_ID` each time to avoid overwriting results.

## Model Variants

### Baseline Models
- **TEST_basic.py** and **TEST_basic_val.py** run ViT models trained from scratch.
- Pretrained ViT models (based on the `timm` library) are also evaluated for comparison.

### Our Models: GLViT Variants
The scripts named `TEST_glvit_vX.py` correspond to our proposed Dynamic Hybrid Attention Transformer (DHAT) variants:

| Version | Architecture Summary |
|--------|------------------------|
| **v1** | `2n x 2n` with 1 CLS token |
| **v2** | `2n x 2n` with 2 CLS tokens |
| **v3** | Selection routing, shared Q/K, unique V, summed CLS |
| **v4** | Selection routing, shared Q/K, unique V, weighted summed CLS |
| **v5** | Selection routing, unique Q/K/V, summed CLS |
| **v6** | Selection routing, unique Q/K/V, weighted summed CLS |
| **v7** | Selection routing, unique Q/K/V, fused MLP CLS |
| **v8** | Selection routing, unique Q/K/V, fused 2-layer MLP CLS |
| **v9** | Fourier-transformed patch embeddings + v8 architecture |
| **v10** | Combination routing (not selection), unique Q/K/V, fused MLP CLS |

All variants are executed with scripts like:

```bash
python TEST_glvit_v7.py --TEST_ID TEST_FINAL_ID022 --dataset cifar10 --batch_size 32 --n_epoch 200 --image_size 32 --patch_size 4 --num_classes 10 --dim 64 --depth 6 --heads 8 --mlp_dim 128

python TEST_glvit_v7_val.py --TEST_ID TEST_FINAL_ID022 --dataset cifar10 --batch_size 32 --n_epoch 200 --image_size 32 --patch_size 4 --num_classes 10 --dim 64 --depth 6 --heads 8 --mlp_dim 128 --print_w True > ../results/TEST_FINAL_ID022/accuracy_stats/output.txt
```

## Notes
- Datasets include MNIST, CIFAR10/100, STL10, FashionMNIST, Oxford Pets, Flowers-102, and Tiny ImageNet.
- Results are saved in the `../results/TEST_ID/` directory.
- For pretrained ViT, we use the [timm](https://github.com/huggingface/pytorch-image-models) library.
- All scripts are modular and can be extended for ablation studies or testing additional variants.
