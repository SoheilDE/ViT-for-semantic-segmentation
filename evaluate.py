import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from vit import ViT
from voc import VOCDataset
from loss import DiceLoss


def evaluate(model, device, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = DiceLoss()(output, target)
            total_loss += loss.item() * data.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    print(f'Average Loss: {avg_loss:.6f}')


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

    val_dataset = VOCDataset('VOCdevkit/VOC2012', split='val', transform=transform, target_transform=target_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = ViT(image_size=224, patch_size=16, num_classes=21, dim=768, depth=6, heads=8, mlp_dim=3072).to(device)
    model.load_state_dict(torch.load('vit_segmentation.pth'))

    evaluate(model, device, val_loader)


if __name__ == '__main__':
    main()