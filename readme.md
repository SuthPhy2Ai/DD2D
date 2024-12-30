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
```bash
python pred.py
```
### Example Usage
Detailed usage examples for the model can be found in the `example` folder, specifically in the `ablation.ipynb` notebook. This notebook demonstrates step-by-step how to use the model for reconstruction tasks and perform ablation studies.

```python
# create the model
mconf = TransformerConfig(train_dataset.block_size,
                  n_layer=n_layer, n_head=n_head, n_embd=embeddingSize)
model = DD2D(mconf)
# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=numEpochs, batch_size=batchSize, 
                      learning_rate=4e-4,
                      lr_decay=True, warmup_tokens=512*20, 
                      final_tokens=2*len(train_dataset)*blockSize,
                      num_workers=0, ckpt_path=ckptPath)
trainer = Trainer(model, train_dataset, val_dataset, tconf, bestLoss, device=device)
print('The following model {} has been loaded!'.format(ckptPath))
```


```python
checkpoint = torch.load(ckptPath)
model.load_state_dict(checkpoint)
model = model.eval().to(trainer.device)
loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    pin_memory=True,
    batch_size=len(test_dataset),
    num_workers=0)
for i, (x, y) in enumerate(loader):
    topN = 1
    acc = sample_from_model(model, y, x, topN=topN)
    print('Top {} accuracy: {}'.format(topN, acc))
```

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
