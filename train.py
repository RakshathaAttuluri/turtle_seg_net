from datetime import datetime
import wandb
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from dataset import TurtleDataset
from unet.unet_model import Unet
from visualize import visualize

VAL_SPLIT = 0.3
BS = 4
EPS = 1
LR = 0.01
WD = 0.01


def initialize_logging(name: str):
    experiment = wandb.init(project=name)
    experiment.config.update(
        dict(epochs=EPS, batch_size=BS, learning_rate=LR)
    )
    return experiment


def get_dataloaders():
    # Initialize and visualize dataset.
    data = TurtleDataset('dataset', 'turtle.png')
    val_sz = int(len(data) * VAL_SPLIT)
    train_sz = int(len(data) - val_sz)
    train_dataset, val_dataset = random_split(data, [train_sz, val_sz])
    visualize(*train_dataset[5])

    # Initialize dataloader.
    train_loader = DataLoader(train_dataset, batch_size=BS,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return (train_loader, val_loader)


def train(device, epoch, model, train_loader, optimizer, loss_fn, logger):
    model.train()
    running_loss = 0
    for i, batch in tqdm(enumerate(train_loader),
                            total=len(train_loader), desc=f'Train: {epoch}/{EPS}'):
        img, mask = batch
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()

        pred_mask = model(img)
        loss = loss_fn(pred_mask, mask)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        if i % 1 == 0 or i == len(train_loader) - 1:
            logger.log({
                'train_loss': running_loss / 1,
                'step': i,
                'epoch': epoch
            })
            running_loss = 0


def validate(device, epoch, model, val_loader, loss_fn, logger, log_img_idx=[]):
    model.eval()

    log_images = []
    running_loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader),
                            total=len(val_loader),
                            desc=f'Val: {epoch}/{EPS}'):
            img, mask = batch
            img, mask = img.to(device), mask.to(device)
            pred_mask = model(img)
            running_loss += loss_fn(pred_mask, mask)

            # Visualize images to log.
            if i in log_img_idx:
                log_images.append(visualize(img, mask, pred_mask))

    avg_val_loss = running_loss / len(val_loader)
    logger.log({
        'val_loss': avg_val_loss,
        'step': len(val_loader),
        'epoch': epoch,
        'images': [wandb.Image(image) for image in log_images]
    })
    return avg_val_loss


def run_training(device, train_loader, val_loader, logger, log_img_idx=[]):
    model = Unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.BCEWithLogitsLoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('Starting training...')
    best_loss = torch.inf
    for epoch in range(EPS):
        train(device, epoch, model, train_loader,
              optimizer, loss_fn, logger)
        val_loss = validate(device, epoch, model, val_loader,
                            loss_fn, logger, log_img_idx)
        if val_loss < best_loss:
            best_loss = val_loss
            model_path = f'checkpt/unet_{timestamp}_best.pt'
            torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = initialize_logging('BinSS')
    # Get random images to log.
    train_loader, val_loader = get_dataloaders()
    log_img_idx = torch.randint(0, len(val_loader), (2,))
    run_training(device, train_loader, val_loader, logger, log_img_idx=log_img_idx)
