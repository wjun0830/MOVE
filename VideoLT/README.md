# (MOVE) VideoLT dataset.
 
## Data Preparation
For the data preparation, send an e-mail to [zxwu at fudan.edu.cn](zxwu@fudan.edu.cn) and agree to the license, then they will send back the download links to you. 
Raw videos(~1.7TB) and extracted features(~900GB in total, ~295GB for each feature set from ResNet50, ResNet101, TSM) will be provided.

Then, to decompress the downloaded `.tar.gz` files, please use commands:
```
cat TSM-R50-feature.tar.gz.part* | tar zx 
cat ResNet50-feature.tar.gz.part* | tar zx
cat ResNet101-feature.tar.gz.part* | tar zx
```

For using extracted features, modify `dataset/dutils.py` and set the correct path to features.

## Usage

### Prepare Data Path

1. Modify `FEATURE_NAME`, `PATH_TO_FEATURE` and `FEATURE_DIM` in `dataset/dutils.py`.

2. Set `ROOT` in `dataset/dutils.py` to `labels` folder. The directory structure is:

```
    labels
    |-- count-labels-train.lst
    |-- test.lst
    |-- test_videofolder.txt
    |-- train.lst
    |-- train_videofolder.txt
    |-- val_videofolder.txt
    `-- validate.lst
```

### Train
We provide scripts for training. Please refer to scripts directory.

- `base_main.py` is for training our baselines.
- `base_main_Agg.py` is for training our baselines with our learnable feature aggregators.
- `MOVE.py` is for training our proposed method with all components.


Example training scripts:

```
FEATURE_NAME='ResNet101'

export CUDA_VISIBLE_DEVICES='0'
python MOVE.py    \
       --augment "None" \
       --feature_name $FEATURE_NAME \
       --lr 0.0001 \
       --lr_steps 30 60 \
       --epochs 100  \
       --batch-size 128  -j 16 --eval-freq 5 --print-freq 8000 \
       --root_log=$FEATURE_NAME-log      --root_model=$FEATURE_NAME'-checkpoints' \
       --store_name=$FEATURE_NAME'_MOVE'      --num_class=1004      --model_name=NonlinearClassifier  \
       --train_num_frames=60      --val_num_frames=150      --loss_func=BCELoss \
       --lb 3.0 \ 
       --mixupbias 0.5 --sampling_prob samplenum
```

### Test

We also provide scripts for testing in `scripts`. 

Example testing scripts:

```
FEATURE_NAME='ResNet101'
CKPT='/project/ResNet-101/ResNet101_MOVE/ckpt.best.pth.tar'

export CUDA_VISIBLE_DEVICES='0'
python MOVE.py \
     --resume $CKPT \
     --evaluate \
     --feature_name $FEATURE_NAME \
     --batch-size 128 -j 16 \
	 --print-freq 20 \
     --num_class=1004 \
     --model_name=NonlinearClassifier \
     --train_num_frames=60 \
     --val_num_frames=150 \
     --loss_func=BCELoss
```

