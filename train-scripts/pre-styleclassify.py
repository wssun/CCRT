import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        filename = os.path.basename(path)
        return sample, target, filename

def get_data(train_dir, height, width, batch_size):
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageFolder(root=train_dir, transform=transform)

    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, num_classes

train_dir = 'stable-diffusion-main/sdAll5000'
train_loader, num_classes = get_data(train_dir, 224, 224, 32)
print("artist list:", num_classes)
artist_list = train_loader.dataset.classes
print(artist_list)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_resnet = models.resnet50(pretrained=True)


for param in pretrained_resnet.parameters():
    param.requires_grad = False

num_ftrs = pretrained_resnet.fc.in_features
pretrained_resnet.fc = nn.Linear(num_ftrs, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pretrained_resnet.parameters())


pretrained_resnet.to(device)

num_epochs = 2

for epoch in range(num_epochs):
    pretrained_resnet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels, filenames) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = pretrained_resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total


    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% ")

save_name = ''.join(artist_list)

torch.save(pretrained_resnet.state_dict(),
           f'artist_image_style/{save_name}.pth')










