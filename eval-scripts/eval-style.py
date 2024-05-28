import argparse

from torchvision import transforms
from PIL import Image
import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def predict_images_in_folder(folder_path, model, artist_names):
    predicted_results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            predicted_artist = predict_artist(image_path, model, artist_names)
            predicted_results.append((filename, predicted_artist))
    return predicted_results

def set_ids_by_artist(artist):
    if artist == 'Picasso':
        return Picasso_ids
    elif artist == 'VanGogh':
        return VanGogh_ids
    elif artist == 'Monet':
        return Monet_ids
    elif artist == 'Cezanne':
        return Cezanne_ids

def predict_artist(image_path, model, artist_names):
    image = preprocess_image(image_path)
    model.eval()
    with torch.no_grad():
        output = model(image)
    _, predicted = output.max(1)
    artist_index = predicted.item()
    artist_name = artist_names[artist_index]
    return artist_name
if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--artist', help='name of model', type=str, required=True)
    parser.add_argument('--data', help='name of model', type=str, required=True)
    args = parser.parse_args()

    artist = args.artist
    data = args.data

    pretrained_resnet = models.resnet50(pretrained=True)
    num_classes = 2
    pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, num_classes)
    pretrained_resnet.load_state_dict(
        torch.load(f'data/artist_image_style/{artist}.pth'))
    pretrained_resnet.eval()
    artist_names = [str(artist), 'unknown']

    folder_path = str(data)
    predicted_results = predict_images_in_folder(folder_path, pretrained_resnet, artist_names)
    Picasso_ids = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 62]
    VanGogh_ids = [30, 31, 32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59]
    Monet_ids = [0, 1, 2, 3, 4]
    Cezanne_ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    pre_fz = 0
    pre_fm = 0
    ids = set_ids_by_artist(artist)
    for filename, predicted_artist in predicted_results:
        class_id_str = filename.split("_")[0]
        class_id = int(class_id_str)

        if class_id in ids:
            print(f"filename: {filename}, predict: {predicted_artist}")
            if predicted_artist == artist:
                pre_fz += 1
            pre_fm += 1

    print(f"prec:{float(pre_fz / pre_fm)}--{pre_fz}/{pre_fm}")

