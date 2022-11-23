# (MOVE) Imbalanced-MiniKinetics200 dataset.

## Data Preparation
Imbalanced-MiniKinetics200 can be downloaded [here](http://115.145.172.53:1215/AAAI_2023_MOVE_longtailed_minikinetics/index.html).
For using extracted features, modify the details in `dataset/dutils_kinetics.py` and set the correct path to features.
If one wants to download dataset by him/herself, please refer to instructions [pretrained_extract_kinetics/README.md](pretrained_extract_kinetics/README.md).

## Usage

Modify `FEATURE_NAME`, `PATH_TO_FEATURE` and `FEATURE_DIM` in `dataset/dutils_kinetics.py`.


## Train
We provide scripts for training. Please refer to scripts directory.  
```
sh scripts/baseline_kinetics.sh
sh scripts/MOVE_kinetics.sh
```

- `base_kinetics.py` is for training our baselines.
- `base_Agg_kinetics.py` is for training our baselines with our learnable feature aggregators.
- `MOVE_kinetics.py` is for training our proposed method with all components.

To run the code, it requires dataset/dutils_kinetics.py, dataset/imbalance_minikinetics.py.

Example training scripts:

```
FEATURE_NAME='ResNet50'
python MOVE_kinetics.py    \
       --augment "None" \
       --feature_name $FEATURE_NAME \
       --lr 0.0001 \
       --lr_steps 30 60 \
       --epochs 100  \
       --batch-size 128  -j 16 --eval-freq 5 --print-freq 8000 \
       --root_log='minikinetics-'$FEATURE_NAME-log      --root_model='minikinetics-'$FEATURE_NAME'-checkpoints' \
       --store_name=$FEATURE_NAME'_MOVE'      --num_class=200      --model_name=NonlinearClassifier  \
       --train_num_frames=60      --val_num_frames=150      --loss_func=BCELoss \
       --lb 3.0 --calib_bias 0.5 --imb_factor 0.01
```

