# Custom dataset
from PIL import Image
import torch.utils.data as data
import os
import random
import cv2 as cv
from torchvision import datasets, transforms
from stride_augmentation import *

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, subfolder='train', direction='AtoB', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(DatasetFromFolder, self).__init__()
        if direction == 'AtoB':
            self.input_path = os.path.join(image_dir, subfolder, 'a')
            self.target_path = os.path.join(image_dir, subfolder, 'b')
            self.label_path = os.path.join(image_dir, subfolder, 'label')
        else:
            self.input_path = os.path.join(image_dir, subfolder, 'b')
            self.target_path = os.path.join(image_dir, subfolder, 'a')
            self.label_path = os.path.join(image_dir, subfolder, 'label')

        print(self.input_path)
        print(self.target_path)
        print(self.label_path)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        print(self.image_filenames)
        self.direction = direction
        self.transform = transform
        self.resize_scale = resize_scale    # resize_scale = 286
        self.crop_size = crop_size  # crop_size = 256
        self.fliplr = fliplr    # fliplr = True

    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img_tar = os.path.join(self.target_path, self.image_filenames[index])

        old = False
        if old:
            file_name = self.image_filenames[index].split(".")[0]
            img_label = os.path.join(self.label_path, file_name + '_mask.png')
        else:
            img_label = os.path.join(self.label_path, self.image_filenames[index])
        
        img_fn_name = img_fn.split('/')[-1]

        img_input = cv.imread(img_fn)
        img_target = cv.imread(img_tar)
        img_label = cv.imread(img_label)

        # preprocessing
        if self.resize_scale:
            
            img_input = cv.resize(img_input, (self.resize_scale, self.resize_scale))
            img_target = cv.resize(img_target, (self.resize_scale, self.resize_scale))
            print("img_target.size is {}".format(img_target.shape))

        if self.crop_size:
            
            x = random.randint(0, self.resize_scale - self.crop_size)
            y = random.randint(0, self.resize_scale - self.crop_size)
           
            img_input = img_input[x : x + self.crop_size, y:y+self.crop_size, :]
            img_target = img_target[x : x + self.crop_size, y:y+self.crop_size, :]
            print("img_target.size is {}".format(img_target.shape))

        if self.fliplr:
            if random.random() < 0.5:
                
                img_input = cv.flip(img_input, 1)
                img_target = cv.flip(img_target, 1)

        img_input = transforms.ToPILImage()(img_input)
        img_target = transforms.ToPILImage()(img_target)
        img_label = transforms.ToPILImage()(img_label)

        if self.transform is not None:
            img_input = self.transform(img_input)
            img_target = self.transform(img_target)
            img_label = self.transform(img_label)

        return img_input, img_target, img_label, img_fn_name

    def __len__(self):
        return len(self.image_filenames)
