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

    def __init__(self, annotations_file: str, raw_data_folder: str, cache=True, transform=None, target_transform=None):
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
        self.cache = {}

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        record = self.annotations_df.iloc[idx]
        images = record.files
        label = record.condition
        stitched_image = stitch_images(images)
        image = torch.from_numpy(np.array(stitched_image).transpose(2, 0, 1)) / 255.0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        self.cache[idx] = (image, label)
        return image, label


def train_val_test_datset_split(dataset, train_ratio=0.75, val_ratio=0.10, test_ratio=0.15, seed=44):
    len_dataset = len(dataset)
    indices = list(range(len_dataset))
    random.seed(seed)
    random.shuffle(indices)
    train_split = int(np.floor(train_ratio * len_dataset))
    val_split = int(np.floor((train_ratio + val_ratio) * len_dataset))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def compute_class_weights(labels):
    class_counts = Counter(labels)
    total = len(labels)
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    return class_weights


def balanced_class_sampler(dataset):
    labels = [label for _, label in dataset]
    class_weights = compute_class_weights(labels)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def BalancedClassDataloder(dataset, batch_size):
    sampler = balanced_class_sampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
