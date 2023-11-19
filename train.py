from datetime import datetime
import wandb
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from dataset import TurtleDataset
from unet.unet_model import Unet

VAL_SPLIT = 0.3
BS = 4
EPS = 2
LR = 0.01
WD = 0.01


def initialize_logging(name: str):
    experiment = wandb.init(project=name)
    experiment.config.update(
        dict(epochs=EPS, batch_size=BS, learning_rate=LR)
    )
    return experiment


def get_dataloaders():
    data = TurtleDataset('dataset', 'turtle.png')
    val_sz = int(len(data) * VAL_SPLIT)
    train_sz = int(len(data) - val_sz)
    train_dataset, val_dataset = random_split(data, [train_sz, val_sz])
    
    train_loader = DataLoader(train_dataset, batch_size=BS,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return (train_loader, val_loader)


def train(epoch, model, train_loader, optimizer, loss_fn, logger):
    model.train()
    running_loss = 0
    for i, batch in tqdm(enumerate(train_loader),
                            total=len(train_loader), desc=f'Train: {epoch}/{EPS}'):
        img, mask = batch
        optimizer.zero_grad()

        pred_mask = nn.functional.softmax(model(img), dim=1)
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


def validate(epoch, model, val_loader, loss_fn, logger):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader),
                            total=len(val_loader), desc=f'Val: {epoch}/{EPS}'):
            img, mask = batch
            pred_mask = nn.functional.softmax(model(img), dim=1)
            running_loss += loss_fn(pred_mask, mask)

    avg_val_loss = running_loss / len(val_loader)
    logger.log({
        'val_loss': avg_val_loss,
        'step': len(val_loader),
        'epoch': epoch
    })
    return avg_val_loss


def run_training(train_loader, val_loader, logger):
    model = Unet()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.BCELoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('Starting training...')
    best_loss = torch.inf
    for epoch in range(EPS):
        train(epoch, model, train_loader, optimizer, loss_fn, logger)
        val_loss = validate(epoch, model, val_loader, loss_fn, logger)
        if val_loss < best_loss:
            best_loss = val_loss
            model_path = f'checkpt/unet_{timestamp}_{epoch}.pt'
            torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    logger = initialize_logging('BinSS')
    train_loader, val_loader = get_dataloaders()
    run_training(train_loader, val_loader, logger)
