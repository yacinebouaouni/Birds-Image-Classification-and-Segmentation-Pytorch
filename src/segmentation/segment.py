import os
import numpy as np

import torch
from torchvision import datasets, transforms
import PIL.Image as Image

from tqdm import tqdm
import natsort
import argparse


def main():
    """
    Function to segment images of the dataset
    :return:
    """

    parser = argparse.ArgumentParser(description='Segment dataset')
    parser.add_argument('-p', '--path', type=str, required=True, help='root path to dataset')
    parser.add_argument('-s', '--savepath', type=str, required=True, help='path to save dataset')

    args = parser.parse_args()
    save_path = args.savepath
    data_path = args.path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, "train"))
    os.mkdir(os.path.join(save_path, "test"))

    images = datasets.ImageFolder(os.path.join(data_path, save_path))
    dir_folder = os.listdir(os.path.join(data_path, save_path))
    dir_folder = natsort.natsorted(dir_folder, reverse=False)
    train_files = [images.samples[i][0].split('/')[-1] for i in range(len(images.samples))]

    path_save = os.path.join(save_path, "train")
    for i in tqdm(range(len(images))):

        image, label = images[i]

        img = preprocess(image).unsqueeze(0).to(device)
        output = model(img)['out'][0]
        output_predictions = output.argmax(0)
        output_predictions[output_predictions > 0] = 1

        res = np.array(image) * output_predictions.cpu().numpy()[:, :, np.newaxis]

        try:
            Image.fromarray(res.astype('uint8')).save(
                fp=os.path.join(path_save + f'{dir_folder[label]}', train_files[i]))
        except OSError:
            os.mkdir(path_save + f'{dir_folder[label]}/')
            Image.fromarray(res.astype('uint8')).save(
                fp=os.path.join(path_save + f'{dir_folder[label]}', train_files[i]))

    images = datasets.ImageFolder(os.path.join(data_path, "test"))
    dir_folder = os.listdir(os.path.join(data_path, "test"))
    dir_folder = natsort.natsorted(dir_folder, reverse=False)
    test_files = [images.samples[i][0].split('/')[-1] for i in range(len(images.samples))]

    path_save = os.path.join(save_path, "test")
    count = 0
    for i in tqdm(range(len(images))):

        image, label = images[i]

        img = preprocess(image).unsqueeze(0)
        output = model(img.to(device))['out'][0]
        output_predictions = output.argmax(0)
        output_predictions[output_predictions > 0] = 1

        if torch.sum(output_predictions) == 0:
            count += 1

        res = np.array(image) * output_predictions.cpu().numpy()[:, :, np.newaxis]

        Image.fromarray(res.astype('uint8')).save(path_save + test_files[i])

    print(f'The number of images with no detection = {count}')


if __name__ == "__main__":
    main()
