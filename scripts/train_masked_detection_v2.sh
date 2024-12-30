DATASET=DRCT-2M

TRAIN_PATH=/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-inpainting/train2017,/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-v1-4/train2017
VAL_PATH=/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-2-inpainting/val2017,/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-v1-4/val2017


TRAIN_REAL_PATH=/root/autodl-tmp/AIGC_data/MSCOCO/train2017
# TRAIN_REAL_PATH='/no/path
VAL_REAL_PATH=/root/autodl-tmp/AIGC_data/MSCOCO/val2017


SAVE_PATH=/root/autodl-tmp/code/DeCLIP/checkpoint/V2/

EXP_NAME=$(date +"%Y%m%d")

python ../train.py --name $EXP_NAME --train_dataset $DATASET --feature_layer layer20 --fix_backbone --train_path $TRAIN_PATH --valid_path $VAL_PATH \
                --mask_plus_label \
                --train_real_list_path $TRAIN_REAL_PATH --valid_real_list_path $VAL_REAL_PATH \
                --checkpoints_dir $SAVE_PATH \
                --gpu_ids 0,1,2 \
                --batch_size 32 --lr 0.0005 --lovasz_weight 0.1 --data_aug drct \
| tee ../checkpoint/log.txt