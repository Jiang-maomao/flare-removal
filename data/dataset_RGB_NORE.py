#from basicsr.utils.registry import DATASET_REGISTRY
from torch.distributions import Normal
import torchvision.transforms.functional as F
import random
import glob
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import os

import albumentations as A
import numpy as np
#import torchvision.transforms.functional as F
from PIL import Image
#from torch.utils.data import Dataset


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


# class DataLoaderTrain(Dataset):
#     def __init__(self, rgb_dir, film_class='target', img_options=None):
#         super(DataLoaderTrain, self).__init__()

#         inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
#         tar_files = sorted(os.listdir(os.path.join(rgb_dir, film_class)))
#         #mas_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))

#         self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
#         self.tar_filenames = [os.path.join(rgb_dir, film_class, x) for x in tar_files if is_image_file(x)]
#         #self.mas_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mas_files if is_image_file(x)]

#         self.img_options = img_options
#         self.sizex = len(self.tar_filenames)  # get the size of target

#         self.transform = A.Compose([
#             #A.RandomSizedCrop(min_max_height=(img_options['h'] / 2, img_options['h']), height=img_options['h'], width=img_options['w']),
#             A.HorizontalFlip(p=0.3),
#             A.VerticalFlip(p=0.3),
#             A.RandomRotate90(p=0.3), ],
#             additional_targets={
#                 'target': 'image'
#                 #'mask': 'image'
#             }
#         )

#     def __len__(self):
#         return self.sizex

#     def __getitem__(self, index):
#         index_ = index % self.sizex

#         inp_path = self.inp_filenames[index_]
#         tar_path = self.tar_filenames[index_]
#         #mas_path = self.mas_filenames[index_]

#         inp_img = Image.open(inp_path)
#         tar_img = Image.open(tar_path)
#         #mas_img = Image.open(mas_path).convert('RGB')

#         inp_img = np.array(inp_img)
#         tar_img = np.array(tar_img)
#         #mas_img = np.array(mas_img)

#         #transformed = self.transform(image=inp_img, target=tar_img, mask=mas_img)
#         transformed = self.transform(image=inp_img, target=tar_img)

#         inp_img = F.to_tensor(transformed['image'])
#         tar_img = F.to_tensor(transformed['target'])
#         #mas_img = F.to_tensor(transformed['mask'])

#         filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

#         #return inp_img, tar_img, mas_img, filename
#         return inp_img, tar_img, filename


class DataLoaderVal(data.Dataset):
    def __init__(self, rgb_dir, film_class='target', img_options=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, film_class)))
        #mas_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))

        self.inp_filenames = [os.path.join(
            rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(
            rgb_dir, film_class, x) for x in tar_files if is_image_file(x)]
        #self.mas_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mas_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.transform = A.Compose([
            A.SmallestMaxSize(max_size=min(img_options['w'], img_options['h'])), ],
            additional_targets={
                'target': 'image'
                # 'mask': 'image'
        }
        )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        #mas_path = self.mas_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        #mas_img = Image.open(mas_path).convert('RGB')

        if self.img_options['ori']:
            inp_img = F.to_tensor(inp_img)
            tar_img = F.to_tensor(tar_img)
            #mas_img = F.to_tensor(mas_img)
        else:
            # inp_img = np.array(inp_img)
            # tar_img = np.array(tar_img)
            #mas_img = np.array(mas_img)
            inp_img = F.to_tensor(inp_img)
            tar_img = F.to_tensor(tar_img)

            # #transformed = self.transform(image=inp_img, target=tar_img, mask=mas_img)
            # transformed = self.transform(image=inp_img, target=tar_img)

            # inp_img = F.to_tensor(transformed['image'])
            # tar_img = F.to_tensor(transformed['target'])
            # #mas_img = F.to_tensor(transformed['mask'])

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        # return inp_img, tar_img, mas_img, filename
        return inp_img, tar_img, filename


class RandomGammaCorrection(object):
    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, image):
        if self.gamma == None:
            # more chances of selecting 0 (original image)
            gammas = [0.5, 1, 2]
            self.gamma = random.choice(gammas)
            return F.adjust_gamma(image, self.gamma, gain=1)
        elif isinstance(self.gamma, tuple):
            gamma = random.uniform(*self.gamma)
            return F.adjust_gamma(image, gamma, gain=1)
        elif self.gamma == 0:
            return image
        else:
            return F.adjust_gamma(image, self.gamma, gain=1)


def remove_background(image):
    # the input of the image is PIL.Image form with [H,W,C]
    image = np.float32(np.array(image))
    _EPS = 1e-7
    rgb_max = np.max(image, (0, 1))
    rgb_min = np.min(image, (0, 1))
    image = (image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
    image = torch.from_numpy(image)
    return image


def glod_from_folder(folder_list, index_list):
    ext = ['png', 'jpeg', 'jpg', 'bmp', 'tif']
    index_dict = {}
    for i, folder_name in enumerate(folder_list):
        data_list = []
        [data_list.extend(glob.glob(folder_name + '/*.' + e)) for e in ext]
        data_list.sort()
        index_dict[index_list[i]] = data_list
    return index_dict


class Flare_Image_Loader(data.Dataset):
    def __init__(self, image_path, img_size, transform_flare, mask_type=None):
        self.ext = ['png', 'jpeg', 'jpg', 'bmp', 'tif']
        self.data_list = []
        [self.data_list.extend(glob.glob(image_path + '/*.' + e))
         for e in self.ext]
        self.flare_dict = {}
        self.flare_list = []
        self.flare_name_list = []

        self.reflective_flag = False
        self.reflective_dict = {}
        self.reflective_list = []
        self.reflective_name_list = []

        self.mask_type = mask_type  # It is a str which may be None,"luminance" or "color"
        self.img_size = img_size
        self.transform_base = transforms.Compose([transforms.RandomCrop((self.img_size, self.img_size), pad_if_needed=True, padding_mode='reflect'),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.RandomRotation(90)
                                                  ])

        self.transform_flare = transforms.Compose([transforms.RandomAffine(degrees=(0, 360), scale=(transform_flare[0], transform_flare[1]), translate=(transform_flare[2]/1440, transform_flare[2]/1440), shear=(-transform_flare[3], transform_flare[3])),
                                                   transforms.CenterCrop(
            (self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90)
        ])

        print("Base Image Loaded with examples:", len(self.data_list))

    def __getitem__(self, index):
        # load base image
        img_path = self.data_list[index]
        base_img = Image.open(img_path).convert('RGB')

        gamma = np.random.uniform(1.8, 2.2)
        to_tensor = transforms.ToTensor()
        adjust_gamma = RandomGammaCorrection(gamma)
        adjust_gamma_reverse = RandomGammaCorrection(1/gamma)
        color_jitter = transforms.ColorJitter(brightness=(0.8, 3), hue=0.0)
        if self.transform_base is not None:
            base_img = to_tensor(base_img)
            base_img = adjust_gamma(base_img)
            base_img = self.transform_base(base_img)
        else:
            base_img = to_tensor(base_img)
            base_img = adjust_gamma(base_img)
            base_img = base_img.permute(2, 0, 1)
        sigma_chi = 0.01*np.random.chisquare(df=1)
        base_img = Normal(base_img, sigma_chi).sample()
        gain = np.random.uniform(0.5, 1.2)
        flare_DC_offset = np.random.uniform(-0.02, 0.02)
        base_img = gain*base_img
        base_img = torch.clamp(base_img, min=0, max=1)

        # load flare image
        flare_path = random.choice(self.flare_list)
        flare_img = Image.open(flare_path).convert('RGB')
        if self.reflective_flag:
            reflective_path = random.choice(self.reflective_list)
            reflective_img = Image.open(reflective_path).convert('RGB')

        flare_img = to_tensor(flare_img)
        flare_img = adjust_gamma(flare_img)

        if self.reflective_flag:
            reflective_img = to_tensor(reflective_img)
            reflective_img = adjust_gamma(reflective_img)
            flare_img = torch.clamp(flare_img+reflective_img, min=0, max=1)

        flare_img = remove_background(flare_img)

        if self.transform_flare is not None:
            flare_img = self.transform_flare(flare_img)

        # change color
        flare_img = color_jitter(flare_img)

        # flare blur
        blur_transform = transforms.GaussianBlur(21, sigma=(0.1, 3.0))
        flare_img = blur_transform(flare_img)
        flare_img = flare_img+flare_DC_offset
        flare_img = torch.clamp(flare_img, min=0, max=1)

        # merge image
        merge_img = flare_img+base_img
        merge_img = torch.clamp(merge_img, min=0, max=1)

        if self.mask_type == None:
            # return {'gt': adjust_gamma_reverse(base_img), 'flare': adjust_gamma_reverse(flare_img), 'lq': adjust_gamma_reverse(merge_img), 'gamma': gamma}
            return adjust_gamma_reverse(merge_img), adjust_gamma_reverse(base_img)
        elif self.mask_type == "luminance":
            # calculate mask (the mask is 3 channel)
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            luminance = 0.3*flare_img[0]+0.59*flare_img[1]+0.11*flare_img[2]
            threshold_value = 0.99**gamma
            flare_mask = torch.where(luminance > threshold_value, one, zero)

        elif self.mask_type == "color":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            threshold_value = 0.99**gamma
            flare_mask = torch.where(merge_img > threshold_value, one, zero)
        elif self.mask_type == "flare":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            threshold_value = 0.7**gamma
            flare_mask = torch.where(flare_img > threshold_value, one, zero)
        # return {'gt': adjust_gamma_reverse(base_img), 'flare': adjust_gamma_reverse(flare_img), 'lq': adjust_gamma_reverse(merge_img), 'mask': flare_mask, 'gamma': gamma}
        return adjust_gamma_reverse(merge_img), adjust_gamma_reverse(base_img)

    def __len__(self):
        return len(self.data_list)

    def load_scattering_flare(self, flare_name, flare_path):
        print(flare_path)
        flare_list = []
        [flare_list.extend(glob.glob(flare_path + '/*.' + e))
         for e in self.ext]
        self.flare_name_list.append(flare_name)
        self.flare_dict[flare_name] = flare_list
        self.flare_list.extend(flare_list)
        len_flare_list = len(self.flare_dict[flare_name])
        if len_flare_list == 0:
            print("ERROR: scattering flare images are not loaded properly")
        else:
            print("Scattering Flare Image:", flare_name,
                  " is loaded successfully with examples", str(len_flare_list))
        print("Now we have", len(self.flare_list), 'scattering flare images')

    def load_reflective_flare(self, reflective_name, reflective_path):
        self.reflective_flag = True
        reflective_list = []
        [reflective_list.extend(glob.glob(reflective_path + '/*.' + e))
            for e in self.ext]
        self.reflective_name_list.append(reflective_name)
        self.reflective_dict[reflective_name] = reflective_list
        self.reflective_list.extend(reflective_list)
        len_reflective_list = len(self.reflective_dict[reflective_name])
        if len_reflective_list == 0:
            print("ERROR: reflective flare images are not loaded properly")
        else:
            print("Reflective Flare Image:", reflective_name,
                  " is loaded successfully with examples", str(len_reflective_list))
        print("Now we have", len(self.reflective_list),
              'refelctive flare images')


# @DATASET_REGISTRY.register()
class Flare_Pair_Loader(Flare_Image_Loader):
    def __init__(self, image_path, img_size, transform_flare, scattering_dict, reflective_dict):
        Flare_Image_Loader.__init__(
            self, image_path, img_size, transform_flare)
        print('test')
        scattering_dict = scattering_dict
        reflective_dict = reflective_dict
        if len(scattering_dict) != 0:
            self.load_scattering_flare('Flare7K_scattering', scattering_dict)
        #if len(reflective_dict) != 0:
           # self.load_reflective_flare('Flare7K_reflective', reflective_dict)
