# Mugs

This is my personal copy of the [Mugs](https://github.com/sail-sg/mugs) repository for image-based SSL customized for my own purposes. The code here can be used to train and evaluate Mugs models.

## Usage examples

### Training 
To train a Mugs model with a ViT-B/16 architecture from scratch on your data, use [`train_mugs.py`](https://github.com/eminorhan/mugs/blob/master/train_mugs.py): 
```python
python -u train_mugs.py \
	--use_fp16 false \
	--arch "vit_base" \
	--batch_size_per_gpu 64 \
	--num_workers 8 \
	--freeze_last_layer 0 \
	--lr 0.0005 \
	--min_lr 0.0005 \
	--output_dir OUTPUT_DIR \
	--data_path DATA_PATH \
	--save_prefix INFORMATIVE_SAVE_PREFIX
```

Note that this uses the [`webdataset`](https://github.com/webdataset/webdataset) interface to feed the data into the model. 

### Linear evaluation 
To evaluate a model with the linear probing approach, use [`eval_linear.py`](https://github.com/eminorhan/mugs/blob/master/eval_linear.py):
```python
python -u eval_linear.py \
	--arch "vit_base" \
	--pretrained_weights MODEL_PATH \
	--save_prefix INFORMATIVE_SAVE_PREFIX \
	--batch_size 1024 \
	--epochs 50 \
	--num_workers 16 \
	--lr 0.0005 \
	--output_dir OUTPUT_DIR \
	--train_data_path TRAIN_DATA_PATH \
	--val_data_path VAL_DATA_PATH \
	--num_labels 1000
```