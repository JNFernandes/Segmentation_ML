import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from SegNet import SegNet
from dataset import SegDataset
from SegNet_simple import SegNetSimplified
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy, random
from time import time
import matplotlib.pyplot as plt
from matplotlib import colors, patches
plt.rcParams['figure.figsize'] = (20, 6)


learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 100
image_height = 256
image_width = 256
train_img_dir = "voc-segmentation/train/image"
train_mask_dir = "voc-segmentation/train/mask"
val_img_dir = "voc-segmentation/val/image"
val_mask_dir = "voc-segmentation/val/mask"

train_transform = A.Compose(
    [
        A.Resize(height=image_height, width=image_width),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transform = A.Compose(
    [
        A.Resize(height=image_height, width=image_width),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ],
)

train_dataset = SegDataset(img_directory=train_img_dir, mask_directory=train_mask_dir, transform=train_transform)
val_dataset = SegDataset(img_directory=val_img_dir, mask_directory=val_mask_dir, transform=val_transform)

### without transformations
# train_dataset = SegDataset(img_directory=train_img_dir, mask_directory=train_mask_dir)
# val_dataset = SegDataset(img_directory=val_img_dir, mask_directory=val_mask_dir)


# first_data = train_dataset[5]
# print(first_data)

# for i, (image, label) in enumerate(train_dataset):
#     if i >= 6 : break
#     plt.subplot(2, 6, i + 1)
#     plt.imshow(image)
#     plt.subplot(2, 6, i + 7)
#     plt.imshow(label)
# plt.show()

# def visualize_augmentations(dataset, idx=2, samples=10, cols=5):
#     dataset = copy.deepcopy(dataset)
#     dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
#     rows = samples // cols
#     figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
#     for i in range(samples):
#         _, image = dataset[idx]
#         ax.ravel()[i].imshow(image)
#         ax.ravel()[i].set_axis_off()
#     plt.tight_layout()
#     plt.show()
#
# random.seed(42)
# visualize_augmentations(train_dataset)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

model = SegNet(in_channels=3, out_channels=1).to(device)
bce_loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()
model.train()


def dice_score(y_pred, y_true, smooth=1):
    dice_loss = (2 * (y_pred * y_true).sum() + smooth) / ((y_pred + y_true).sum() + smooth)
    return dice_loss


tic = time()
for epoch in range(epochs):
    avg_loss = 0
    avg_dice = 0
    for (images, labels) in train_loader:
        # images = images.permute(0, 3, 1, 2).to(device) #if we dont use transformations
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        # forward
        with torch.cuda.amp.autocast():  # deal with different precision calculations
            predictions = model(images)
            dice = dice_score(torch.sigmoid(predictions), labels)
            loss = bce_loss(predictions, labels) + (1-dice)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        # avg_loss
        avg_loss += float(loss / len(train_loader))
        # dice
        avg_dice += float(dice / len(train_loader))

        # if epoch+1 % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, dice_score: {avg_dice:.4f}')
toc = time()
print(f'Elapsed time: {toc-tic:.1f}s')

model.eval()
dice = 0
# check dice score
with torch.no_grad():
    for (images, labels) in val_loader:
        # images = images.permute(0, 3, 1, 2).to(device)
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)
        preds = torch.sigmoid(model(images))
        preds = (preds > 0.5).float()
        dice += dice_score(preds, labels)

print(f'The dice score is: {(dice/len(val_loader))*100:.2f} %')
