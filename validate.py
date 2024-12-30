import argparse
import os
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from models import get_model
from PIL import Image, ImageOps
from dataset_paths import DETECTION_DATASET_PATHS, LOCALISATION_DATASET_PATHS, MASKED_DETECTION_DATASET_PATHS, MASKED_DETECTION_DATASET_PATHS_V2
import random
import shutil
from utils.utils import compute_batch_iou, compute_batch_localization_f1, compute_batch_ap, generate_outputs, find_best_threshold, compute_accuracy_detection, compute_average_precision_detection
from data.datasets import RealFakeDataset, RealFakeDetectionDataset, RealFakeMaskedDetectionDataset, RealFakeMaskedDetectionDataset_V2
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
from options.test_options import TestOptions
from tqdm import tqdm

from utils.visualize import *


SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}
STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def validate(model, loader):
    
    with torch.no_grad():
        y_true, y_pred = [], []
        all_img_paths = []
        print ("Length of dataset: %d" %(len(loader.dataset)))
        for img, label, img_names in loader:
            in_tens = img.cuda()
            outputs = torch.sigmoid(model(in_tens))
            outputs = torch.mean(outputs , dim=1)

            y_pred.extend(outputs)
            y_true.extend(label)
            all_img_paths.extend(img_names)
        
    y_pred = torch.stack(y_pred).to('cpu')
    y_true = torch.stack(y_true).to('cpu')

    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    mean_acc_best_th = compute_accuracy_detection(y_pred, y_true, threshold = best_thres)
    mean_acc = compute_accuracy_detection(y_pred, y_true)
    mean_ap = compute_average_precision_detection(y_pred, y_true)

    return mean_ap, mean_acc, mean_acc_best_th, best_thres, all_img_paths   

def validate_fully_supervised(model, loader, dataset_name, output_save_path = ''):
    with torch.no_grad():
        ious = []
        f1_best = []
        f1_fixed = []
        all_img_paths = []
        mean_ap = []
        print ("Length of dataset: %d" %(len(loader.dataset)))

        for _, data in enumerate(loader):
            img, _, img_paths, masks_paths = data
            
            in_tens = img.cuda()
            outputs = torch.sigmoid(model(in_tens).squeeze(1))

            if dataset_name in ["pluralistic", "lama", "repaint-p2-9k", "ldm", "ldm_clean", "ldm_real"]:
                masks = [ImageOps.invert(Image.open(mask_path).convert("L")) for mask_path in masks_paths]
            else:
                masks = [Image.open(mask_path).convert("L") for mask_path in masks_paths]

            masks = [ ((transforms.ToTensor()(x).to(outputs.device)) > 0.5).float().squeeze() for x in masks]

            outputs = outputs.view(outputs.size(0), int(outputs.size(1)**0.5), int(outputs.size(1)**0.5))
            resized_outputs = []
            for i, output in enumerate(outputs):
                if output.size() != masks[i].size():
                    output_resized = F.resize(output.unsqueeze(0), masks[i].size(), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                    resized_outputs.append(output_resized)
                else:
                    resized_outputs.append(output)

            batch_ious = compute_batch_iou(resized_outputs, masks, threshold = 0.5)
            batch_F1_best, batch_F1_fixed = compute_batch_localization_f1(resized_outputs, masks)
            batch_ap = compute_batch_ap(resized_outputs, masks)
            
            if output_save_path:
                generate_outputs(output_save_path + "/" + dataset_name, resized_outputs, img_paths)
            
            ious.extend(batch_ious)
            f1_best.extend(batch_F1_best)
            f1_fixed.extend(batch_F1_fixed)
            all_img_paths.extend(img_paths)
            mean_ap.extend(batch_ap)

    return ious, f1_best, f1_fixed, mean_ap, all_img_paths

# xjw
def validate_masked_detection(model, loader):
    with torch.no_grad():
        ious = []
        f1_best = []
        f1_fixed = []
        y_true, y_pred = [], []
        all_img_paths = []
        print("Length of dataset: %d" %(len(loader.dataset)))
        
        with tqdm(total=len(loader)) as pbar:
            for _, data in enumerate(loader):
                img, label, _, img_paths, mask_paths = data

                in_tens = img.cuda()
                outputs = model(in_tens)

                masks = outputs["mask"]
                logits = outputs["logit"]
                masks = torch.sigmoid(masks.squeeze(1))
                logits = torch.sigmoid(logits)

                gd_masks = [Image.open(mask_path).convert("L") for mask_path in mask_paths]
                gd_masks = [((transforms.ToTensor()(x).to(masks.device)) > 0.5).float().squeeze() for x in gd_masks]

                masks = masks.view(masks.size(0), int(masks.size(1)**0.5), int(masks.size(1)**0.5))
                resized_masks = []
                for i, mask in enumerate(masks):
                    if mask.size() != gd_masks[i].size():
                        mask_resized = F.resize(mask.unsqueeze(0), gd_masks[i].size(), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                        resized_masks.append(mask_resized)
                    else:
                        resized_masks.append(mask)

                batch_ious = compute_batch_iou(resized_masks, gd_masks, threshold = 0.5)
                batch_F1_best, batch_F1_fixed = compute_batch_localization_f1(resized_masks, gd_masks)

                ious.extend(batch_ious)
                f1_best.extend(batch_F1_best)
                f1_fixed.extend(batch_F1_fixed)

                y_pred.extend(logits)
                y_true.extend(label)

                all_img_paths.extend(img_paths)
                
                pbar.update(1)
            
        y_pred = torch.stack(y_pred).to('cpu')
        y_true = torch.stack(y_true).to('cpu')
        
        best_thres = find_best_threshold(y_true, y_pred)
        mean_acc_best_th = compute_accuracy_detection(y_pred, y_true, threshold = best_thres)
        mean_acc = compute_accuracy_detection(y_pred, y_true)
        mean_ap = compute_average_precision_detection(y_pred, y_true)
                                   
    return ious, f1_best, f1_fixed, mean_ap, mean_acc, mean_acc_best_th, best_thres, all_img_paths

def validate_masked_detection_v2(model, loader, visualize_mask=False, output_folder="", dataset_name=""):
    with torch.no_grad():
        ious = []
        f1_best = []
        f1_fixed = []
        y_true, y_pred = [], []
        all_img_paths = []
        print("Length of dataset: %d" %(len(loader.dataset)))
        
        if visualize_mask:
            # preparation for visualizing masks
            os.makedirs(os.path.join(output_folder, dataset_name), exist_ok=True)
            mask_save_path = os.path.join(output_folder, dataset_name)
            os.makedirs(mask_save_path, exist_ok=True)
        
        with tqdm(total=len(loader), ncols=150) as pbar:
            for batch_idx, data in enumerate(loader):
                img, label, gd_masks, img_paths, cutmix_img_be_aug = data

                in_tens = img.cuda()
                outputs = model(in_tens)

                masks = outputs["mask"]
                logits = outputs["logit"]
                masks = torch.sigmoid(masks.squeeze(1))
                logits = torch.sigmoid(logits)

                gd_masks = [((x.to(masks.device)) > 0.5).float().squeeze() for x in gd_masks]

                masks = masks.view(masks.size(0), int(masks.size(1)**0.5), int(masks.size(1)**0.5))
                resized_masks = []
                for i, mask in enumerate(masks):
                    if mask.size() != gd_masks[i].size():                
                        mask_resized = F.resize(mask.unsqueeze(0), gd_masks[i].size(), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                        resized_masks.append(mask_resized)
                    else:
                        resized_masks.append(mask)
                        
                if visualize_mask and (batch_idx % 100 == 0):
                    # visualize the masks
                    for i, (mask, gd_mask, l, prob) in enumerate(zip(resized_masks, gd_masks, label, logits)):
#                         # Binarize predicted mask
#                         binary_mask = (mask > 0.5).float()

#                         # Convert to uint8 images
#                         binary_mask_np = (binary_mask.cpu().numpy() * 255).astype(np.uint8)
#                         gd_mask_np = (gd_mask.cpu().numpy() * 255).astype(np.uint8)

#                         # Save predicted mask
#                         pred_mask_img = Image.fromarray(binary_mask_np)
#                         pred_mask_img.save(os.path.join(mask_save_path, f"batch{batch_idx}_sample{i}_pred.png"))

#                         # Save ground truth mask
#                         gt_mask_img = Image.fromarray(gd_mask_np)
#                         gt_mask_img.save(os.path.join(gd_mask_save_path, f"batch{batch_idx}_sample{i}_gt.png"))
    
                        binary_mask = (mask > 0.5).float().cpu()
                        
                        binary_mask = binary_mask.unsqueeze(0).unsqueeze(0)
                        binary_mask = torch.nn.functional.interpolate(binary_mask, size=(224, 224), mode='bilinear', align_corners=False)
                        binary_mask = binary_mask.squeeze(0).squeeze(0)
                        
                        pred_l = (prob >= 0.5).long()
                        
                        img_name = os.path.basename(img_paths[i])
                        img_prefix = img_name.split(".")[0]
                        file_name = f"{img_prefix}_gt_{l}_pd_{pred_l}"
        
                        # 应用预测掩码进行高低光融合
                        fused_img = apply_masked_highlight(img[i], binary_mask)

                        # 可视化
                        visualize_fused_image(
                            img=img[i], 
                            gd_mask=gd_mask, 
                            pred_mask=binary_mask, 
                            fused_img=fused_img,
                            save_path=mask_save_path,
                            file_name=file_name,
                        )


                batch_ious = compute_batch_iou(resized_masks, gd_masks, threshold = 0.5)
                batch_F1_best, batch_F1_fixed = compute_batch_localization_f1(resized_masks, gd_masks)

                ious.extend(batch_ious)
                f1_best.extend(batch_F1_best)
                f1_fixed.extend(batch_F1_fixed)

                y_pred.extend(logits)
                y_true.extend(label)

                all_img_paths.extend(img_paths)
                
                pbar.update(1)
            
        y_pred = torch.stack(y_pred).to('cpu')
        y_true = torch.stack(y_true).to('cpu')
        
        best_thres = find_best_threshold(y_true, y_pred)
        mean_acc_best_th = compute_accuracy_detection(y_pred, y_true, threshold = best_thres)
        mean_acc = compute_accuracy_detection(y_pred, y_true)
        mean_ap = compute_average_precision_detection(y_pred, y_true)
                                   
    return ious, f1_best, f1_fixed, mean_ap, mean_acc, mean_acc_best_th, best_thres, all_img_paths

            
        
def save_scores_to_file(ious, f1_best, f1_fixed, aps, img_paths, file_path):
    with open(file_path + "/scores.txt", 'w') as file:
        file.write(f'Image path \t iou \t f1_best \t f1_fixed \t ap\n')
        for iou, f1_b, f1_f, ap, img_path in zip(ious, f1_best, f1_fixed, aps, img_paths):
            file.write(f'{img_path} \t {iou} \t {f1_b} \t {f1_f} \t {ap}\n')

def save_scores_to_file_detection(aps, acc0s, acc1s, img_paths, file_path):
    with open(file_path + "/scores.txt", 'w') as file:
        file.write(f'Image path \t AP \t Acc_fixed \t Acc_best \n')
        for ap, acc0, acc1, th, img_path in zip(aps, acc0s, acc1s, img_paths):
            file.write(f'{img_path} \t {ap} \t {acc0} \t {acc1}\n')

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    assert opt.ckpt != None
    
    # Add test args
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    try:
        opt.feature_layer = state_dict['feature_layer']
        opt.decoder_type = state_dict['decoder_type']
    except:
        print('No feature_layer or decoder_type in the checkpoint state_dict, using the info from feature_layer and decoder_type args')

    # Load model
    model = get_model(opt)
    if len(opt.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpu_ids)
        new_state_dict = state_dict['model']
    else:
        # remove the "module." prefix in state_dict['model']
        new_state_dict = {}
        for key, value in state_dict['model'].items():
            if key.startswith('module.'):
                new_key = key[7:]  # 移除 `module.` 前缀
            else:
                new_key = key
            new_state_dict[new_key] = value
        
    model.load_state_dict(new_state_dict)
    print ("Model loaded..")
    
    model.eval()
    model.to(opt.gpu_ids[0])

    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    if opt.fully_supervised:
        dataset_paths = LOCALISATION_DATASET_PATHS
        # Create and write the header of results file
        with open( os.path.join(opt.result_folder,'scores.txt'), 'a') as f:
            f.write('dataset \t iou \t f1_best \t f1_fixed \t ap \n' )
    elif opt.mask_plus_label:
        dataset_paths = MASKED_DETECTION_DATASET_PATHS_V2
        with open( os.path.join(opt.result_folder,'scores.txt'), 'a') as f:
            f.write('dataset \t iou \t f1_best \t f1_fixed \t AP \t Acc_fixed \t Acc_best \n' )
    else:
        dataset_paths = DETECTION_DATASET_PATHS
        with open( os.path.join(opt.result_folder,'scores.txt'), 'a') as f:
            f.write('dataset \t AP \t Acc_fixed \t Acc_best \t Best_threshold \n' )


    for dataset_path in (dataset_paths):
        print(f"Testing on {dataset_path['key']}")

        set_seed()
        opt.test_path = dataset_path['fake_path']
        opt.train_dataset = dataset_path['key']

        if opt.fully_supervised:
            opt.test_masks_ground_truth_path = dataset_path['masks_path']
            dataset = RealFakeDataset(opt)
        elif opt.mask_plus_label:
            # opt.test_masks_ground_truth_path = dataset_path['fake_masks_path']
            # opt.test_masks_real_ground_truth_path = dataset_path['real_masks_path']
            opt.test_real_list_path = dataset_path['real_path']
            dataset = RealFakeMaskedDetectionDataset_V2(opt)
        else:
            opt.test_masks_ground_truth_path = dataset_path['masks_path']
            opt.test_real_list_path = dataset_path['real_path']
            dataset = RealFakeDetectionDataset(opt)

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        
        # Localisation
        if opt.fully_supervised:
            if opt.output_save_path:
                output_save_path = opt.output_save_path + "/" + dataset_path['key']
                if not os.path.exists(output_save_path):
                    os.makedirs(output_save_path)
                
            ious, f1_best, f1_fixed, ap, original_img_paths = validate_fully_supervised(model, loader, dataset_path['key'], output_save_path = opt.output_save_path)
            mean_iou = sum(ious)/len(ious)
            mean_f1_best = sum(f1_best)/len(f1_best)
            mean_f1_fixed = sum(f1_fixed)/len(f1_fixed)
            mean_ap = sum(ap)/len(ap)
            if opt.output_save_path:
                save_scores_to_file(ious, f1_best, f1_fixed, ap, original_img_paths, output_save_path)
            
            with open( os.path.join(opt.result_folder,'scores.txt'), 'a') as f:
                f.write(dataset_path['key']+': ' + str(round(mean_iou, 3))+ '\t' +\
                    str(round(mean_f1_best, 4))+ '\t' +\
                    str(round(mean_f1_fixed, 4))+ '\t' +\
                    str(round(mean_ap, 4))+ '\t' +\
                    '\n' )
                print(dataset_path['key']+': IOU = ' + str(round(mean_iou, 3)))
                print(dataset_path['key']+': F1_best = ' + str(round(mean_f1_best, 4)))
                print(dataset_path['key']+': F1_fixed = ' + str(round(mean_f1_fixed, 4)))
                print(dataset_path['key']+': AP = ' + str(round(mean_ap, 4)))
                print()
                
        # Masked detection
        if opt.mask_plus_label:
            if opt.output_save_path:
                output_save_path = opt.output_save_path + "/" + dataset_path['key']
                if not os.path.exists(output_save_path):
                    os.makedirs(output_save_path)
            
            if opt.visualize_masks:
                ious, f1_best, f1_fixed, mean_ap, mean_acc, mean_acc_best_th, best_thres, all_img_paths = validate_masked_detection_v2(model, loader, visualize_mask=True, output_folder=opt.result_folder, dataset_name=dataset_path['key'])
            else:
                ious, f1_best, f1_fixed, mean_ap, mean_acc, mean_acc_best_th, best_thres, all_img_paths = validate_masked_detection_v2(model, loader)
            
            mean_iou = sum(ious)/len(ious)
            mean_f1_best = sum(f1_best)/len(f1_best)
            mean_f1_fixed = sum(f1_fixed)/len(f1_fixed)
            
            with open(os.path.join(opt.result_folder, 'scores.txt'), 'a') as f:
                f.write(dataset_path['key']+': ' + str(round(mean_iou, 3)) + '\t' +\
                       str(round(mean_f1_best, 4)) + '\t' +\
                       str(round(mean_f1_fixed, 4)) + '\t' +\
                       str(round(mean_ap, 4)) + '\t' +\
                       str(round(mean_acc, 4)) + '\t' +\
                       str(round(mean_acc_best_th, 4)) + '\t' +\
                       '\n' )
                print(dataset_path['key']+': IOU = ' + str(round(mean_iou, 3)))
                print(dataset_path['key']+': F1_best = ' + str(round(mean_f1_best, 4)))
                print(dataset_path['key']+': F1_fixed = ' + str(round(mean_f1_fixed, 4)))
                print(dataset_path['key']+': AP = ' + str(round(mean_ap, 4)))
                print(dataset_path['key']+': Acc_fixed = ' + str(round(mean_acc, 4)))
                print(dataset_path['key']+': Acc_best = ' + str(round(mean_acc_best_th, 4)))
                print()
            

        # Detection
        if not opt.fully_supervised and not opt.mask_plus_label:
            mean_ap, mean_acc, mean_acc_best_th, best_thres, all_img_paths = validate(model, loader)

            with open( os.path.join(opt.result_folder,'scores.txt'), 'a') as f:
                f.write(dataset_path['key']+': ' + str(round(mean_ap, 4))+ '\t' +\
                        str(round(mean_acc, 4)) + '\t' +\
                        str(round(mean_acc_best_th, 4)) + '\t' +\
                        str(best_thres) + '\t' +\
                        '\n' )
                print(dataset_path['key']+': AP = ' + str(round(mean_ap, 4)))
                print(dataset_path['key']+': Acc_fixed = ' + str(round(mean_acc, 4)))
                print(dataset_path['key']+': Acc_best = ' + str(round(mean_acc_best_th, 4)))
                print(dataset_path['key']+': Best_threshold = ' + str(round(best_thres, 4)))
                print()