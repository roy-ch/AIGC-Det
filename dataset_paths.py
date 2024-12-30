def get_dolos_localisation_dataset_paths(dataset):
    paths = dict(
        fake_path=f'datasets/dolos_data/celebahq/fake/{dataset}/images/test',
        masks_path=f'datasets/dolos_data/celebahq/fake/{dataset}/masks/test',
        key=dataset
    )
    return paths

def get_dolos_detection_dataset_paths(dataset):
    paths = dict(
        real_path=f'datasets/dolos_data/celebahq/fake/{dataset}/images/test',
        fake_path=f'datasets/dolos_data/celebahq/real/{dataset}/test',
        masks_path=f'datasets/dolos_data/celebahq/fake/{dataset}/masks/test',
        key=dataset
    ),
    return paths

def get_autosplice_localisation_dataset_paths(compression):
    paths = dict(
        fake_path=f'datasets/AutoSplice/Forged_JPEG{compression}',
        masks_path=f'datasets/AutoSplice/Mask',
        key=f'autosplice_jpeg{compression}'
    )
    return paths

def get_drct_2m_localisation_dataset_paths(dataset):
    paths = dict(
        fake_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/val2017',
        masks_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/masks/val2017',
        key=dataset,
    )
    return paths

def get_drct_2m_detection_dataset_paths(dataset):
    paths = dict(
        real_path='/root/autodl-tmp/AIGC_data/MSCOCO/val2017',
        fake_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/val2017',
        masks_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/masks/val2017',
        key=dataset
    )
    return paths

def get_drct_2m_masked_detection_dataset_paths(dataset):
    paths = dict(
        real_path='/root/autodl-tmp/AIGC_data/MSCOCO/val2017',
        fake_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/val2017',
        real_masks_path='/root/autodl-tmp/AIGC_data/MSCOCO/masks/val2017',
        fake_masks_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/masks/val2017',
        key=dataset
    )
    return paths

def get_drct_2m_masked_detection_dataset_paths_v2(dataset):
    paths = dict(
        real_path='/root/autodl-tmp/AIGC_data/MSCOCO/val2017',
        fake_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/val2017',
        key=dataset
    )
    return paths

# LOCALISATION_DATASET_PATHS = [
#     get_dolos_localisation_dataset_paths('pluralistic'),
#     get_dolos_localisation_dataset_paths('lama'),
#     get_dolos_localisation_dataset_paths('repaint-p2-9k'),
#     get_dolos_localisation_dataset_paths('ldm'),
#     # TO BE PUBLISHED
#     # get_dolos_localisation_dataset_paths('ldm_clean'),
#     # get_dolos_localisation_dataset_paths('ldm_real'),

#     get_autosplice_localisation_dataset_paths("75"),
#     get_autosplice_localisation_dataset_paths("90"),
#     get_autosplice_localisation_dataset_paths("100"),
# ]

# DETECTION_DATASET_PATHS = [
#     get_dolos_detection_dataset_paths('pluralistic'),
#     get_dolos_detection_dataset_paths('lama'),
#     get_dolos_detection_dataset_paths('repaint-p2-9k'),
#     get_dolos_detection_dataset_paths('ldm'),
#     # TO BE PUBLISHED
#     # get_dolos_detection_dataset_paths('ldm_clean'),
#     # get_dolos_detection_dataset_paths('ldm_real'),
# ]

# our localisation dataset paths
LOCALISATION_DATASET_PATHS = [
    get_drct_2m_localisation_dataset_paths("stable-diffusion-2-inpainting"),
]

# our detection dataset paths
DETECTION_DATASET_PATHS = [
    get_drct_2m_detection_dataset_paths("stable-diffusion-2-inpainting"),
]

DRCT_2M_SUBSETS = ["controlnet-canny-sdxl-1.0", "lcm-lora-sdv1-5", "lcm-lora-sdxl", "ldm-text2im-large-256", "sd-controlnet-canny", "sd-turbo", 
                  "sd21-controlnet-canny", "sdxl-turbo", "stable-diffusion-2-1", "stable-diffusion-2-inpainting", "stable-diffusion-inpainting",
                  "stable-diffusion-v1-4", "stable-diffusion-v1-5", "stable-diffusion-xl-1.0-inpainting-0.1", "stable-diffusion-xl-base-1.0",
                  "stable-diffusion-xl-refiner-1.0", "fake_rec_image/stable-diffusion-2-1", "fake_rec_image/stable-diffusion-v1-4"]

MASKED_DETECTION_DATASET_PATHS = [
    get_drct_2m_masked_detection_dataset_paths("stable-diffusion-2-inpainting"),
]

MASKED_DETECTION_DATASET_PATHS_V2 = [
    get_drct_2m_masked_detection_dataset_paths_v2(subset) for subset in DRCT_2M_SUBSETS
]