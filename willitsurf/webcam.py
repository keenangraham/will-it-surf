import av

import time

from PIL import Image

from concurrent.futures import ThreadPoolExecutor

import logging


logger = logging.getLogger(__name__)


def gather_images_from_webcam_feed(webcam_url: str, num_images: int, secs_between_capture: int) -> list[Image]:
    images = []
    i = 0
    while True:
        i += 1
        logger.info(f'opening container {webcam_url}')
        container = av.open(webcam_url)
        logger.info(f'getting frame {i}')
        frame = next(container.decode(video=0))
        images.append(frame.to_image())
        container.close()
        if i >= num_images:
            break
        logger.info(f'sleeping {secs_between_capture}')
        time.sleep(secs_between_capture)
    return images


def gather_images_from_all_webcams(webcam_urls: list[str], num_images: int = 3, secs_between_capture: int = 10) -> list[Image]:
    all_images = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for url in webcam_urls:
            futures.append(
                executor.submit(
                    gather_images_from_webcam_feed, url, num_images, secs_between_capture,
                )
            )
        for f in futures:
            all_images.extend(f.result())
    return all_images


def gather_grouped_images_from_all_webcams(webcam_urls: dict[str, str], num_images: int = 3, secs_between_capture: int = 10) -> dict[str, list[Image]]:
    grouped_images = {}
    grouped_futures = {}
    with ThreadPoolExecutor() as executor:
        for name, url in webcam_urls.items():
            if name not in grouped_images:
                grouped_images[name] = []
            if name not in grouped_futures:
                grouped_futures[name] = []
            grouped_futures[name].append(
                executor.submit(
                    gather_images_from_webcam_feed, url, num_images, secs_between_capture,
                )
            )
        for name, futures in grouped_futures.items():
            for future in futures:
                grouped_images[name].extend(future.result())
    return grouped_images
