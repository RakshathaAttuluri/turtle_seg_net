import io
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid

def visualize(image, mask, pred_mask=None):
    plt.figure(figsize=(18, 18))
    plt.subplot(131)
    plt.imshow((make_grid([image], nrow=1)
                .squeeze(dim=0).permute(1, 2, 0).cpu().numpy()))
    plt.subplot(132)
    plt.imshow((make_grid([mask], nrow=1)
                .squeeze(dim=0).permute(1, 2, 0).cpu().numpy()))
    if pred_mask != None:
        plt.subplot(133)
        plt.imshow((make_grid([pred_mask], nrow=1)
                    .squeeze(dim=0).permute(1, 2, 0).cpu().numpy()))

    # Return a PIL image.
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf = Image.open(img_buf)
    img_buf.show()

    return img_buf
