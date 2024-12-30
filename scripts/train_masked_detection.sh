DATASET=DRCT-2M

TRAIN_PATH=/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-2-inpainting/train2017
VAL_PATH=/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-2-inpainting/val2017

TRAIN_MASK_PATH=/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-2-inpainting/masks/train2017
VAL_MASK_PATH=/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-2-inpainting/masks/val2017

TRAIN_REAL_PATH=/root/autodl-tmp/AIGC_data/MSCOCO/train2017
# TRAIN_REAL_PATH='/no/path'
VAL_REAL_PATH=/root/autodl-tmp/AIGC_data/MSCOCO/val2017

TRAIN_REAL_MASK_PATH=/root/autodl-tmp/AIGC_data/MSCOCO/masks/train2017
# TRAIN_REAL_MASK_PATH='/no/path'
VAL_REAL_MASK_PATH=/root/autodl-tmp/AIGC_data/MSCOCO/masks/val2017

SAVE_PATH=/root/autodl-tmp/code/DeCLIP/checkpoint

EXP_NAME=$(date +"%Y%m%d_%H%M%S")

python ../train.py --name $EXP_NAME --train_dataset $DATASET --feature_layer layer20 --fix_backbone --train_path $TRAIN_PATH --valid_path $VAL_PATH \
                --train_masks_ground_truth_path $TRAIN_MASK_PATH --valid_masks_ground_truth_path $VAL_MASK_PATH \
                --mask_plus_label --train_masks_real_ground_truth_path $TRAIN_REAL_MASK_PATH --valid_masks_real_ground_truth_path $VAL_REAL_MASK_PATH \
                --train_real_list_path $TRAIN_REAL_PATH --valid_real_list_path $VAL_REAL_PATH \
                --checkpoints_dir $SAVE_PATH \
                --gpu_ids 0\
                --batch_size 16 --lr 0.0005 \