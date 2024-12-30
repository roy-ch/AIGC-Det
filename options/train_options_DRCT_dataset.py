from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--train_dataset', type=str, default='pluralistic', help='the dataset on which to train')
        parser.add_argument('--decoder_type', type=str, default='conv-20', help='type of decoder (linear/attention/conv-4/conv-12/conv-20)')
        parser.add_argument('--feature_layer', type=str, default=None, help='layer of the backbone from which to extract features')
        parser.add_argument('--data_aug', type=str, default=None, help='if specified, perform additional data augmentation (blur/color_jitter/jpeg_compression/all)')
        
        parser.add_argument('--earlystop_epoch', type=int, default=5)
        parser.add_argument('--optim', type=str, default='adam', help='optim to use (sgd/adam)')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')

        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')


        parser.add_argument('--loss_freq', type=int, default=50, help='frequency of showing loss on tensorboard')
        parser.add_argument('--niter', type=int, default=400, help='total epochs')
        
        parser.add_argument('--train_masks_ground_truth_path', type=str, default='datasets/dolos_data/celebahq/fake/ldm/masks/train', help='path to train ground truth masks (only for fully_supervised training)')
        parser.add_argument('--valid_masks_ground_truth_path', type=str, default='datasets/dolos_data/celebahq/fake/ldm/masks/valid', help='path to valid ground truth masks (only for fully_supervised training)')
        parser.add_argument('--train_real_list_path', default='datasets/dolos_data/celebahq/real/train', help='folder path to training real data')
        parser.add_argument('--valid_real_list_path', default='datasets/dolos_data/celebahq/real/valid', help='folder path to validation real data')
        
        # xjw
        parser.add_argument('--lovasz_weight', type=float, default=0.01, help='the weight of lovasz loss in mask training')
        
        parser.add_argument('--train_masks_real_ground_truth_path', type=str, default='', help="path to train real ground truth masks (only for mask_plus_label training)")
        parser.add_argument('--valid_masks_real_ground_truth_path', type=str, default='', help="path to valid real ground truth masks (only for mask_plus_label training)")
        
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')
        
        parser.add_argument('--model_path', type=str, default=None, help='Setting the model path')
        parser.add_argument("--root_path", default='/disk4/chenby/dataset/MSCOCO', help="Setting the root path for dataset loader", type=str)
        parser.add_argument("--fake_root_path", default='/disk4/chenby/dataset/AIGC_MSCOCO', help="Setting the fake root path for dataset loader", type=str)
        parser.add_argument("--fake_indexes", default='1', help="Setting the fake indexes, multi class using '1,2,3,...' ", type=str)
        parser.add_argument("--num_classes", default=2, help="Setting the num classes", type=int)
        parser.add_argument("--inpainting_dir", default='full_inpainting', help="rec_image dir", type=str)
        parser.add_argument('--is_dire', action='store_true', help='Whether to using DIRE?')
        parser.add_argument("--input_size", default=224, help="Image input size", type=int)
        parser.add_argument('--is_crop', action='store_true', help='Whether to crop image?')  # 默认是 False
        parser.add_argument('--prob_aug', default=0.5, type=float, help="Setting the probability for augmentation")
        parser.add_argument('--prob_cutmix', default=0.5, type=float, help="Setting the probability for cutmix")
        parser.add_argument('--prob_cutmixup_real_fake', default=0.5, type=float, help="Probability for performing cutmix between real and fake images")
        parser.add_argument('--prob_cutmixup_real_rec', default=0.5, type=float, help="Probability for performing cutmix between real and reconstructed images")
        parser.add_argument('--prob_cutmixup_real_real', default=0.5, type=float, help="Probability for performing cutmix between two real images") 
        
        self.data_label = 'train'
        return parser
