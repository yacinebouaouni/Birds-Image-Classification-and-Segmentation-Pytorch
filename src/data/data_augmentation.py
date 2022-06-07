import os
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import natsort
import argparse


def main():
    """
    Function to augment the original dataset with multiple transformations (geometric and photometric
    """

    parser = argparse.ArgumentParser(description='Augment dataset')
    parser.add_argument('-p', '--path', type=str, required=True, help='root path to dataset')
    parser.add_argument('-v', '--viz', type=bool, required=False, default=False, help='Set True to visualize data transforms')
    parser.add_argument('-s', '--savepath', type=str, required=True, help='path to save dataset')
    parser.add_argument('-m', '--mode', type=str, required=True, default="train", choices=["train", "valid"], help="specify train/validation daata")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = args.path
    path_save = args.savepath

    # Define data transformations
    tr1 = transforms.GaussianBlur(kernel_size=7, sigma=(2, 2.0))
    tr2 = transforms.RandomHorizontalFlip(p=1)
    tr3 = transforms.RandomPerspective()
    tr4 = transforms.RandomAdjustSharpness(sharpness_factor=30, p=1)
    tr5 = transforms.ColorJitter(brightness=.5, hue=.2)
    tr6 = transforms.Compose([transforms.Resize((386, 386)), transforms.RandomCrop(336)])

    transformations = [tr1, tr2, tr3, tr4, tr5, tr6]
    transforms_names = ['blur', 'flip', 'perspective', 'sharp', 'color', 'crop']

    if args.viz:

        fig, ax = plt.subplots(5, 7, figsize=(20, 15))
        for j in range(5):

            idx = np.random.randint(1000)
            image = datasets.ImageFolder(data_path + 'train_images')[idx]
            image = image[0]

            ax[j, 0].imshow(image)
            i = 1
            for tr in transformations:
                ax[j, i].imshow(tr(image))
                i += 1

        plt.show()



    os.mkdir(path_save)
    os.mkdir(os.path.join(path_save, "train"))

    for t in transforms_names:
        os.mkdir('/kaggle/working/AugmentedDataset/test/'+t+'/')

    phase = args.mode
    images = datasets.ImageFolder(os.path.join(data_path, phase))
    dir_folder = os.listdir(os.path.join(data_path, phase))
    dir_folder = natsort.natsorted(dir_folder, reverse=False)
    train_files = [images.samples[i][0].split('/')[-1] for i in range(len(images.samples))]

    path_save = os.path.join(path_save, phase)

    for i in tqdm(range(len(images))):

        im, label = datasets.ImageFolder(data_path + phase)[i]

        try:
            im.save(fp=os.path.join(path_save + f'{dir_folder[label]}', train_files[i]))

        except OSError:

            os.mkdir(path_save + f'{dir_folder[label]}/')
            im.save(fp=os.path.join(path_save + f'{dir_folder[label]}', train_files[i]))

        for T, name in zip(transformations, transforms_names):

            im_t = T(im)

            try:
                im_t.save(fp=os.path.join(path_save + f'{dir_folder[label]}', name + train_files[i]))

            except OSError:

                os.mkdir(path_save + f'{dir_folder[label]}/')
                im_t.save(fp=os.path.join(path_save + f'{dir_folder[label]}', name + train_files[i]))


if __name__ == "__main__":
    main()
