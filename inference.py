import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from unet.unet_model import Unet
from visualize import visualize


def run_inference(device, checkpt_file_path, image):
    """ Function to run inference using Unet model on a given image.`

    Arguments:
        device -- cpu or cuda device to run on
        checkpt_file_path -- Path the Unet model weights
        image -- PIL Image to run inference on

    Returns:
        Tuple of image and predicted mask.
    """
    # Load model.
    model = Unet().to(device)
    model.load_state_dict(torch.load(
        checkpt_file_path, map_location=torch.device(device)))
    model.eval()

    # Initialize transforms.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Pad(4),
        transforms.ToTensor()
    ])

    # Read image.
    im = transform(image)
    im = im[None, :].to(device)
    pred_mask = model(im)

    # Resize images back to their sizes.
    im = transforms.functional.resize(im, size=image.size)
    pred_mask = transforms.functional.resize(
        pred_mask, size=image.size).detach().mul(255).clamp(0, 255)

    return im, pred_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt_path',
                        type=str, required=True)
    parser.add_argument('--img_path',
                        type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = Image.open(args.img_path)
    mask = run_inference(device, args.checkpt_path, image)
