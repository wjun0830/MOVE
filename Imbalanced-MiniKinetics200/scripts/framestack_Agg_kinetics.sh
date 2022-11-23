FEATURE_NAME='ResNet50'

export CUDA_VISIBLE_DEVICES='0'
python Framestack_Agg_kinetics.py  \
     --feature_name $FEATURE_NAME \
     --lr 0.0001 \
     --gd 20 --lr_steps 30 60 --epochs 100 \
     --batch-size 128 -j 16 \
     --eval-freq 5 \
	 --print-freq 7000 \
     --root_log='minikinetics-'$FEATURE_NAME-log \
     --root_model='minikinetics-'$FEATURE_NAME'-checkpoints' \
     --store_name=$FEATURE_NAME'_framestack_Agg' \
     --gamma 1 \
     --num_class=200 \
     --model_name=NonlinearClassifier \
     --train_num_frames=60 \
     --val_num_frames=150 \
     --loss_func=FocalLoss \
     --clip_length=60 \
     --warm_epoch=0 \
     --ratio=0.5 \
     --imb_factor 0.02