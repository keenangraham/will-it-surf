import os

from datetime import datetime

from .webcam import gather_grouped_images_from_all_webcams

from .feed import urls

import logging

import torch

from torch.utils.data import Dataset

from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# datetime.strptime(datetime_string, "%Y-%m-%d-%H-%M-%S")


def collect_training_example(raw_data_folder: str, webcam_urls: dict[str, str], num_images: int = 3, secs_between_capture: int = 10):
    grouped_images = gather_grouped_images_from_all_webcams(webcam_urls, num_images, secs_between_capture)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    for name, images in grouped_images.items():
        for i, image in enumerate(images):
            filename = f'{name}-{timestamp}-image-{i}.png'
            w, h = image.size
            im = image.resize((w // 2, h // 2))
            path = f'{raw_data_folder}/{filename}'
            logging.info(f'saving training file to {path}')
            im.save(path)


def annotate_raw_data(raw_data_folder: str, annotations_file: str):
    if Path(annotations_file).exists():
        df = pd.read_csv(annotations_file, sep='\t')
    else:
        df = pd.DataFrame(columns=['group', 'files' 'condition'])
    grouped_files = {}
    raw_files = Path(raw_data_folder)
    for raw_file in raw_files:
        group_name = fn[0].split('-image')[0]
        if group_name not in grouped_files:
            grouped_files[group_name] = []
        grouped_files[group_name].append(raw_file.name)
    # calculate groups of files in raw_data_folder
    # filter out groups that are already in labels
    # for every group, display as grid
    # ask for g/b condition report
    # store (group, [files], condition) tuples in labels
    raw_files = [f.name for f in Path(raw_data_folder).iterdir() if f.is_file()]
    annotated_files = 
    pass



class SurfImageDataset(Dataset):

    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None):
        self.image_labels = pd
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
