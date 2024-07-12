import torch

from torch.utils.data import Dataset

import numpy as np

import pandas as pd

from .image import stitch_images

import ast

from pathlib import Path


def convert_files_columns(files: str):
    return [
        Path(f)
        for f in ast.literal_eval(files)
    ]


class SurfImageDataset(Dataset):

    def __init__(self, annotations_file: str, raw_data_folder: str, transform=None, target_transform=None):
        self.annotations_df = pd.read_csv(
            annotations_file,
            sep='\t',
            converters={
                'files': convert_files_columns
            },
        )
        self.raw_data_folder = raw_data_folder
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        record = self.annotations_df.iloc[idx]
        images = record.files
        label = record.condition
        stitched_image = stitch_images(images)
        image = torch.from_numpy(np.array(stitched_image).transpose(2, 0, 1)) / 255.0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
