import glob
import random
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TurtleDataset(Dataset):
    def __init__(self, bg_img_dir: str, fg_img_path: str,
                 transform=None) -> None:
        super().__init__()
        self.bg_img_paths = glob.glob(f'{bg_img_dir}/*.png')
        self.fg_img = Image.open(fg_img_path).resize((256, 256))
        self.transform = transform
        if transform == None:
            self.transform = transforms.Compose([
                transforms.Pad(4),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.bg_img_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.bg_img_paths[index])
        img, seg_mask = self._random_paste(img)
        return (img, seg_mask)

    def _random_paste(self, background_image, min_scale=0.25, max_scale=0.65):
        """Randomly scales and pastes the turtle image onto the background image"""
    
        w, h = self.fg_img.size
        # first, we will randomly downscale the turtle image
        new_w = int(random.uniform(min_scale, max_scale) * w)
        new_h = int(random.uniform(min_scale, max_scale) * h)
        resized_turtle_image = self.fg_img.resize((new_w, new_h))

        # second, will randomly choose the locations where to paste the new image
        start_w = random.randint(0, w - new_w)
        start_h = random.randint(0, h - new_h)

        # third, will create the blank canvas of the same size as the original image
        canvas_image = Image.new('RGBA', (w, h))

        # and paste the resized turtle onto it, preserving the mask
        canvas_image.paste(resized_turtle_image, (start_w, start_h), resized_turtle_image)
        
        # Turtle image is of mode RGBA, while background image is of mode RGB;
        # `.paste` requires both of them to be of the same type.
        background_image = background_image.copy().convert('RGBA')
        # finally, will paste the resized turtle onto the background image
        background_image.paste(resized_turtle_image, (start_w, start_h), resized_turtle_image)

        # Prepare background image.
        background_image = self.transform(background_image)[:3]

        # Prepare one hot encoded foreground image.
        canvas_image = self._prep_canvas_image(canvas_image)
        return background_image, canvas_image

    def _prep_canvas_image(self, seg_mask):
        seg_mask = transforms.ToTensor()(seg_mask)
        seg_mask = (seg_mask[-1:] > 0).to(torch.int64)

        seg_mask = nn.functional.one_hot(
            seg_mask, 2).squeeze().permute(2, 0, 1).to(torch.float32)
        return seg_mask