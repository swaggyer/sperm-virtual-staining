import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class Singleimage(Dataset):
    def __init__(self, root_image,start =0,end = 550,transform=None):
        self.root_image = root_image
        self.input_images = [f"{i}.tif"for i in range (start,end+1)]
        self.images_len = len(self.input_images)
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.images_len

    def __getitem__(self, index):
        # input_imgs = [Image.open(os.path.join(image_path, str(f) + '.tif')) for f in range(551)]
        input_image = self.input_images[index]
        # images_path = os.path.join(self.root_image,input_img)
        image_path = os.path.join(self.root_image,input_image)
        input_img = np.array(Image.open(image_path).convert("RGB"))
        # input_img = Image.fromarray(np.array(input_img), "RGB")
        input_img = self.transform(input_img)

        # index = self.transform(input_image)
        index = torch.Tensor(int(input_image[:-4]))
        return {"image":input_img,"index":index}
        # return input_img
class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode="train"):
        self.transform = transforms.Compose(transform)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files = sorted(glob.glob(os.path.join(root, "train") + "/*.*"))
        elif mode == "test":
            self.files = sorted(glob.glob(os.path.join(root, "test") + "/*.*"))
        else:
            raise ValueError(f"Invalid mode {mode}. Must be 'train' or 'test'.")
            #print(f"Dataset length: {len(self.files)} images")
    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        # 裁剪图片
        img_B = img.crop((0, 0, w / 2, h))#图片左边为B，明场显微部分
        img_A = img.crop((w / 2, 0, w, h))#

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

