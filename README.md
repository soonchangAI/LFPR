## Efficient label-free pruning and retraining for Text-VQA Transformers

The code implementation for [Efficient label-free pruning and retraining for Text-VQA Transformers](https://www.sciencedirect.com/science/article/abs/pii/S0167865524001338)
## Installation 
```
conda create -n lfpr_tap python=3.6.13
pip install -r TAP/requirements.txt

conda activate lfpr_tap
cd TAP

python setup.py develop

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
│   │   │   ├── january.csv
│   │   │   └── TAP_predicted_labels/
            └── TAP12_predicted_labels/

```

## Quickstart

The pruning and retraining scripts are located in <code>[scripts](scripts)</code>

1. Setup the paths:

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

2. Run experiments for TAP(TextVQA)

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

## Credits

The TAP implementation is based on following repo:
* [TAP: Text-Aware Pre-training](https://github.com/microsoft/TAP)

The pruning heuristic <code>sum</code> is based on:
* [A Fast Post-Training Pruning Framework for Transformers](https://github.com/WoosukKwon/retraining-free-pruning)