# nproc_per_node
# 
base_dir="./output_dir_mesorch2"
mkdir -p ${base_dir}

Prob_aug=1
P_cutmixup_real_fake=0
P_cutmixup_real_rec=0
P_cutmixup_real_real=1

echo "Prob_aug: $Prob_aug"
echo "P_cutmixup_real_fake: $P_cutmixup_real_fake"
echo "P_cutmixup_real_rec: $P_cutmixup_real_rec"
echo "P_cutmixup_real_real: $P_cutmixup_real_real"

CUDA_VISIBLE_DEVICES=0,1 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
./train_DRCT.py \
    --model Mesorch \
    --conv_pretrain True \
    --seg_pretrain_path "./segformer/mit_b3.pth" \
    --world_size 2 \
    --find_unused_parameters \
    --batch_size 4 \
    --epochs 150 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --weight_decay 0.05 \
    --is_DRCT \
    --prob_aug ${Prob_aug} \
    --prob_cutmixup_real_fake ${P_cutmixup_real_fake} \
    --prob_cutmixup_real_rec  ${P_cutmixup_real_rec} \
    --prob_cutmixup_real_real ${P_cutmixup_real_real} \
    --root_path /root/autodl-tmp/AIGC_data/MSCOCO/train2017 \
    --fake_root_path /root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-inpainting/train2017,/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-v1-4/train2017 \
    --fake_indexes 2 \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 2 \
    --seed 42 \
    --test_period 2 \
    --num_workers 12 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log