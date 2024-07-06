import os

from datetime import datetime

from .webcam import gather_grouped_images_from_all_webcams

from .feed import urls

import logging


logger = logging.getLogger(__name__)

# datetime.strptime(datetime_string, "%Y-%m-%d-%H-%M-%S")


def collect_training_example(raw_data_folder: str, webcam_urls: dict[str, str], num_images: int = 3, secs_between_capture: int = 3):
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
