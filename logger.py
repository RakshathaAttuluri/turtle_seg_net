import logging
import os
from datetime import datetime


class Logger:
    def __init__(self, name: str):
        self.name = name
        self.out_path = self._init_out_folder()

    def _init_out_folder(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = f'{self.name}_log_dump_{timestamp}'
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        return out_path

    def log(self, log_dict: dict):
        images = log_dict.pop('images', None)
        self.save_images(images)

        logging.info(log_dict.items())

    def save_images(self, images):
        if images == None:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for i, image in enumerate(images):
            image.image.save(f'{self.out_path}/{timestamp}_{i}.png')

    