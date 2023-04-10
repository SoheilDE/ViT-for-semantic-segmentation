import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from vit import ViT
from voc import VOCDataset
from loss import DiceLoss, CrossEntropyLoss


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = DiceLoss()(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.f'%)]\tLoss: {loss.item():.6f}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = VOCDataset('VOCdevkit/VOC2012', split='train', transform=transform, target_transform=target_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    model = ViT(image_size=224, patch_size=16, num_classes=21, dim=768, depth=6, heads=8, mlp_dim=3072).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    log_interval = 20
    for epoch in range(1, 10):
        train(model, device, train_loader, optimizer, epoch, log_interval)

    torch.save(model.state_dict(), 'vit_segmentation.pth')


if __name__ == '__main__':
    main()