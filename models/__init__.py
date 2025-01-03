from .clip_models import CLIPModelLocalisation

from .unet import UNet
from models.DTD.dtd import seg_dtd


VALID_NAMES = [
    'CLIP:RN50',  
    'CLIP:ViT-L/14',
    'CLIP:xceptionnet',
    'CLIP:ViT-L/14,RN50', 
]

def get_model(opt):
    model_name, layer, decoder_type = opt.arch, opt.feature_layer, opt.decoder_type

    # assert name in VALID_NAMES

    if 'unet' in model_name:
        return UNet(n_channels = 3, n_classes = 1)
    elif 'CLIP' in model_name:
        return CLIPModelLocalisation(model_name.split(':')[1], intermidiate_layer_output = layer, decoder_type=decoder_type,
                                     mask_plus_label=opt.mask_plus_label, cls_model=opt.cls_model)
    elif 'DTD' in model_name:
        return seg_dtd(model_name='resnet18', n_class=1)
