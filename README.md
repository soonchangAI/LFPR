# Efficient label-free pruning and retraining for Text-VQA Transformers

[Soon Chang Poh](https://github.com/soonchangAI), [Chee Seng Chan](http://cs-chan.com), [Chee Kau Lim](https://umexpert.um.edu.my/limck.html)


The official implementation for [Efficient label-free pruning and retraining for Text-VQA Transformers](https://www.sciencedirect.com/science/article/abs/pii/S0167865524001338)


* A label-free importance score for structured pruning of autoregressive Transformers for TextVQA.

*  An adaptive retraining approach for pruned Transformer models of varying sizes.

* Achieve up to 60% reduction in size with only <2.4% drop in accuracy.

## Results

**Comparison of pruning between $L_1$ and our proposed method in terms of different model size for TAP(TextVQA) architecture.**

| Method                              | Hardware      | \#Params | Val acc | GPU hours | File Size (MB) | Cloud computing cost |
|-------------------------------------|---------------|----------|---------|------------|----------------|-----------------------|
| $L_1$        | V100×2        | 11.8M    | 47.73   | 33.9       | 305            | \$207.5               |
| Absolute Loss Gradient              | TitanXP×2     | 11.0M    | 44.99   | 1.58       | 301.5          | \$0.316               |
| OWO                                 | TitanXP×2     | 13.4M    | 47.56   | 0.91       | 310.9          | \$0.182               |
| Fisher Information                  | TitanXP×2     | 10.6M    | 25.29   | 0.14       | 299.8          | \$0.028               |
| **Ours**                            | TitanXP×2     | 11.4M    | 47.57   | 0.58       | 43.9           | \$0.116               |

| Method                              | Hardware      | \#Params | Val acc | GPU hours | File Size (MB) | Cloud computing cost |
|-------------------------------------|---------------|----------|---------|------------|----------------|-----------------------|
| $L_1$       | V100×2        | 21.3M    | 48.23   | 33.8       | 343            | \$206.9               |
| Absolute Loss Gradient              | TitanXP×2     | 22.6M    | 47.53   | 1.76       | 348            | \$0.352               |
| OWO                                 | TitanXP×2     | 22.7M    | 49.79   | 0.71       | 355.7          | \$0.142               |
| Fisher Information                  | TitanXP×2     | 22.4M    | 48.89   | 0.16       | 347.1          | \$0.032               |
| **Ours**                            | TitanXP×2     | 22.7M    | 49.92   | 0.30       | 20.1           | \$0.06                |



## Installation 
```
git clone https://github.com/soonchangAI/LFPR
cd LFPR
conda create -n lfpr_tap python=3.6.13
pip install -r TAP/requirements.txt

conda activate lfpr_tap
cd TAP

python setup.py develop

```

## Data Setup

For TextVQA and ST-VQA dataset, [see](TAP/data/README.md)

For sample set and retraining set, [download here](https://drive.google.com/drive/folders/1ls7UOG7eg6gP8gXEnijrkTUvDMkIGxIN?usp=sharing) and structure the directory as follows:
```
imdb/
├── m4c_textvqa/
│   ├── calculate_score/
│   └── TAP_predicted_labels/
│   └── TAP12_predicted_labels/

original_dl/
│   ├── m4c_stvqa/
│   │   ├── calculate_score/
│   │   ├── TAP_predicted_labels/
│   │   └── TAP12_predicted_labels/
```

## Quickstart

The pruning and retraining scripts are located in <code>[scripts](scripts)</code>

1. Setup the paths in the scripts:
```
# General config

code_dir= # directory of repo /TAP
output_dir= # output directory to save pruned models
data_dir= # data directory
org_model=$checkpoint/save/finetuned/textvqa_tap_base_best.ckpt # checkpoint directory

# Pruning config
prune_code_dir= # directory of repo

# retrain config
num_gpu= # number of GPUs
```

2. Run experiment using the script. For example, run experiment for TAP(TextVQA)

```
cd scripts/tap_pruning/tap_textvqa
chmod +x prune_tap_textvqa.sh
./prune_tap_textvqa.sh
```

## Citation
```
@article{POH20241,
title = {Efficient label-free pruning and retraining for Text-VQA Transformers},
journal = {Pattern Recognition Letters},
volume = {183},
pages = {1-8},
year = {2024},
issn = {0167-8655},
doi = {https://doi.org/10.1016/j.patrec.2024.04.024},
url = {https://www.sciencedirect.com/science/article/pii/S0167865524001338},
author = {Soon Chang Poh and Chee Seng Chan and Chee Kau Lim},
}
```

## Acknowledgement
The TAP implementation is based on [TAP: Text-Aware Pre-training](https://github.com/microsoft/TAP)

The pruning heuristic <code>sum</code> is based on [A Fast Post-Training Pruning Framework for Transformers](https://github.com/WoosukKwon/retraining-free-pruning)
