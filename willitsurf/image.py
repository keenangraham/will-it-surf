import math

import logging

from PIL import Image

from pathlib import Path

logger = logging.getLogger(__name__)


def make_image_grid(images: list[Image]) -> Image:
    num_images = len(images)
    rows = int(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)
    logging.info(f'making grid image with {rows} rows, {cols} cols, and {num_images} images')
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)
    grid_width = cols * max_width
    grid_height = rows * max_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * max_width
        y = row * max_height
        grid_image.paste(image, (x, y))
    return grid_image


def stitch_images(image_paths: list[Path]) -> Image:
    images = [Image.open(f) for f in sorted(image_paths)]
    widths, heights = zip(*(im.size for im in images))
    total_width = sum(widths)
    max_height = max(heights)
    stitched_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        stitched_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return stitched_image
