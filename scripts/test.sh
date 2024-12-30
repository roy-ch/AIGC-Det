 CHECK_POINT_PATH=/root/autodl-tmp/code/DeCLIP/checkpoint/V2/20241229/model_last_epoch_6_acc_94.35567954462198.pth
 RESULT_PATH=V2/20241229
 
 python ../validate.py --arch=CLIP:ViT-L/14 --ckpt=$CHECK_POINT_PATH \
                    --result_folder=/root/autodl-tmp/code/DeCLIP/results/$RESULT_PATH --gpu_ids 0 \
                    --mask_plus_label \
                    --batch_size 16  --visualize_masks\