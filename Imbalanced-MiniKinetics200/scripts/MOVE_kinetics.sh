FEATURE_NAME='ResNet50'
export CUDA_VISIBLE_DEVICES='0'

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