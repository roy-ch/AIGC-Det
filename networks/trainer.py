import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_model import BaseModel
from models import get_model
from utils.utils import compute_batch_iou, compute_batch_localization_f1, compute_batch_ap

import numpy as np
from PIL import Image

from .lovasz_loss import lovasz_hinge, lovasz_softmax

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.model = get_model(opt)
            
        # Initialize all possible parameters in the final layer
        for fc in self.model.fc:
            try:
                torch.nn.init.normal_(fc.weight.data, 0.0, opt.init_gain)
            except:
                pass

        # xjw
        if opt.mask_plus_label:
            for conv_cls in self.model.conv_cls:
                try:
                    torch.nn.init.normal_(conv_cls.weight.data, 0.0, opt.init_gain)
                except:
                    pass
        

        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if ("fc" in name and "resblock" not in name) or ("conv_cls" in name):
                    params.append(p)
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()
        
        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")
        
        self.loss_fn = nn.BCEWithLogitsLoss()

        if len(opt.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids)
        self.model.to(opt.gpu_ids[0])       
        
        
        if opt.fully_supervised:
            self.ious = []
            self.F1_best = []
            self.F1_fixed = []
            self.ap = []
        elif opt.mask_plus_label:
            # xjw
            self.ious = []
            self.F1_best = []
            self.F1_fixed = []
            self.logits = []
            self.labels = []
            
            self.lovasz_weight = opt.lovasz_weight
        else:
            self.logits = []
            self.labels = []
            
        self.times = 0 # for visualize
            

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()
        
        # xjw
        if self.opt.mask_plus_label:
            self.mask = input[2].to(self.device).float()
            # self.mask = self.label

    def forward(self):
        # self.output = self.model(self.input)
        self.output = self.model(self.input) # output will be a dict when mask_plus_label=True
        
        if self.opt.fully_supervised:
            # resize prediction to ground truth mask size
            if self.label.size()[1] != 256 * 256:
                label_size = (int(self.label.size()[1] ** 0.5), int(self.label.size()[1] ** 0.5))
                self.output = self.output.view(-1, 1, 256, 256)
                self.output = F.interpolate(self.output, size=label_size, mode='bilinear', align_corners=False)
                self.output = torch.flatten(self.output, start_dim=1).unsqueeze(1)
        
        # xjw
        if self.opt.mask_plus_label:
            if self.mask.size()[1] != 256*256:
                mask_size = (int(self.mask.size()[1] ** 0.5), int(self.mask.size()[1] ** 0.5))
                self.output["mask"] = self.output["mask"].view(-1, 1, 256, 256)
                self.output["mask"] = F.interpolate(self.output["mask"], size=mask_size, mode='bilinear', align_corners=False)
                self.output["mask"] = torch.flatten(self.output["mask"], start_dim=1).unsqueeze(1)
                

        if not self.opt.fully_supervised and not self.opt.mask_plus_label:
            self.output = torch.mean(self.output, dim=1)

    def get_loss(self):
        if not self.opt.mask_plus_label:
            return self.loss_fn(self.output.squeeze(1), self.label)
        else:
            # return self.loss_fn(self.output["mask"].squeeze(1), self.mask) + self.loss_fn(self.output["logit"].squeeze(1), self.label)
            return self.loss

    def optimize_parameters(self):
        self.forward()
        # outputs = self.output
        # xjw
        if not self.opt.mask_plus_label:
            outputs = self.output
        else:
            masks = self.output["mask"]
            logits = self.output["logit"]
            # print(f'self.output["mask"]:{self.output["mask"]}', flush=True)
        if self.opt.fully_supervised:
            sigmoid_outputs = torch.sigmoid(outputs)
            
            # unflatten outputs and ground truth masks
            sigmoid_outputs = sigmoid_outputs.view(sigmoid_outputs.size(0), int(sigmoid_outputs.size(1)**0.5), int(sigmoid_outputs.size(1)**0.5))
            labels = self.label.view(self.label.size(0), int(self.label.size(1)**0.5), int(self.label.size(1)**0.5))

            iou = compute_batch_iou(sigmoid_outputs, labels)
            self.ious.extend(iou)

            F1_best, F1_fixed = compute_batch_localization_f1(sigmoid_outputs, labels)
            self.F1_best.extend(F1_best)
            self.F1_fixed.extend(F1_fixed)
            
            ap = compute_batch_ap(sigmoid_outputs, labels)
            self.ap.extend(ap)
        elif self.opt.mask_plus_label:
            # xjw
            sigmoid_masks = torch.sigmoid(masks)
            
            sigmoid_masks = torch.where(sigmoid_masks > 0.5, torch.tensor(1.0, device=sigmoid_masks.device), torch.tensor(0.0, device=sigmoid_masks.device))
            
            # unflatten mask and ground truth masks
            sigmoid_masks = sigmoid_masks.view(sigmoid_masks.size(0), int(sigmoid_masks.size(1)**0.5), int(sigmoid_masks.size(1)**0.5))
            gd_masks = self.mask.view(self.mask.size(0), int(self.mask.size(1)**0.5), int(self.mask.size(1)**0.5))
                
            iou = compute_batch_iou(sigmoid_masks, gd_masks)
            self.ious.extend(iou)
            
            F1_best, F1_fixed = compute_batch_localization_f1(sigmoid_masks, gd_masks)
            self.F1_best.extend(F1_best)
            self.F1_fixed.extend(F1_fixed)
            
            self.logits.append(logits)
            self.labels.append(self.label)
            
        else:
            self.logits.append(outputs)
            self.labels.append(self.label)

        self.optimizer.zero_grad()
        
        # self.loss = self.loss_fn(outputs, self.label) 
        
        # xjw
        if self.opt.mask_plus_label:
            # self.loss = (0.5 - self.lovasz_weight) * self.loss_fn(masks, self.mask) +  self.lovasz_weight * lovasz_hinge(masks, self.mask) + 0.5 * self.loss_fn(logits, self.label)
            
            sigmoid_masks = torch.sigmoid(masks)
            sigmoid_masks = sigmoid_masks.view(sigmoid_masks.size(0), int(sigmoid_masks.size(1)**0.5), int(sigmoid_masks.size(1)**0.5))
            gd_masks = self.mask.view(self.mask.size(0), int(self.mask.size(1)**0.5), int(self.mask.size(1)**0.5))
            
            self.bce_mask_loss = self.loss_fn(masks, self.mask)
            self.lovasz_mask_loss = lovasz_softmax(sigmoid_masks, gd_masks, classes=[1])
            self.label_loss = self.loss_fn(logits, self.label)
            self.loss = (0.5 - self.lovasz_weight) * self.bce_mask_loss +  self.lovasz_weight * self.lovasz_mask_loss + 0.5 * self.label_loss
            
            # print(f'self.loss_fn(masks, self.mask) :{self.loss_fn(masks, self.mask)} ')
            # print(f'lovasz_hinge(masks, self.mask) :{lovasz_hinge(masks, self.mask)} ')
            # print(f'self.loss_fn(logits, self.label) :{self.loss_fn(logits, self.label)} ')
            
            # self.loss = 0.5 * self.loss_fn(masks, self.mask) + 0.5 * self.loss_fn(logits, self.label)
            # self.loss = self.loss_fn(masks, self.mask)
        else:
            self.loss = self.loss_fn(outputs, self.label)
        
        self.loss.backward()
        self.optimizer.step()

    def format_output(self):
        if not self.opt.fully_supervised:
            self.logits = torch.cat(self.logits, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
