import os

from datetime import datetime

from .webcam import gather_grouped_images_from_all_webcams

from .feed import urls

from .image import stitch_images

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


def get_grouped_files(raw_data_folder: str) -> dict[str, list[str]]:
    grouped_files = {}
    raw_files = [f for f in Path(raw_data_folder).glob('*.png')]
    for raw_file in raw_files:
        group_name = raw_file.name.split('-image')[0]
        if group_name not in grouped_files:
            grouped_files[group_name] = []
        grouped_files[group_name].append(raw_file)
    return grouped_files


def annotate_raw_data(raw_data_folder: str, annotations_file: str):
    if Path(annotations_file).exists():
        df = pd.read_csv(annotations_file, sep='\t')
    else:
        df = pd.DataFrame(columns=['group', 'files', 'condition'])
    grouped_files = get_grouped_files(raw_data_folder)
    entries = []
    for group, raw_files in grouped_files.items():
        if group in df['group'].values:
            continue
        stitched_image = stitch_images(raw_files)
        stitched_image.show()
        while True:
            condition = input(f'Annotate {group} with g for good or b for bad: ')
            if condition in ['g', 'b']:
                break
            print('invalid input, enter 0, 1')
        new_entry = {
            'group': group,
            'files': [str(f) for f in raw_files],
            'condition': 1 if condition == 'g' else 0
        }
        entries.append(new_entry)
    if entries:
        new_df = pd.DataFrame(entries)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(annotations_file, sep='\t', index=False)
    return df


class SurfImageDataset(Dataset):

    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
