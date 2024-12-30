# README

## About this Project

This repository contains the code implementation of the model described in the manuscript **"Reconstructing the 2D Material Structure from Diffraction Pattern with Physical-Sensitive Deep Learning"**.

## Installation

To set up the environment for this project, you can use one of the following methods:

### Using Conda:
```bash
conda env create -f environment_requests.yaml
conda activate dd2d
```

### Using Pip:
```bash
pip install -r requirements.txt
```

## Usage

### Pre-trained Models
Pre-trained models are stored in the `SavedModels` folder as `.pt` files, which include the models showcased in the manuscript.

### Example Usage
Detailed usage examples for the model can be found in the `example` folder, specifically in the `ablation.ipynb` notebook. This notebook demonstrates step-by-step how to use the model for reconstruction tasks and perform ablation studies.

### Training a Custom Model
If you wish to train your own model from scratch, you can use the `train.py` script. The training process is configurable through the `TrainerConfig`, which includes the following parameters:
- `max_epochs`: Number of training epochs.
- `batch_size`: Batch size.
- `learning_rate`: Set to `4e-4` by default.
- `lr_decay`: Enable learning rate decay.
- `warmup_tokens`: Set to `512*20`.
- `final_tokens`: Set to `2 * len(train_dataset) * blockSize`.
- `num_workers`: Number of data loader workers (set to `0` by default).
- `ckpt_path`: Path to save the checkpoint.

For detailed experiments on the model structure and ablation studies, refer to the `example` folder's `ablation.ipynb` notebook.

## Acknowledgement

If you utilize the data or code from this repository, please reference our paper **"Reconstructing the 2D Material Structure from Diffraction Pattern with Physical-Sensitive Deep Learning"**. 
