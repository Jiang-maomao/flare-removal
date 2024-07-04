import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io
from torchvision.transforms import ToTensor
import numpy as np
from glob import glob
import lpips
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor=ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    output_lpips = loss_fn_alex(img1_tensor, img2_tensor)
    return output_lpips.detach().numpy()[0,0,0,0]

def calculate_metrics_real(args):
    loss_fn_alex = lpips.LPIPS(net='alex')
    gt_folder = args['gt_real'] + '/*'
    input_folder = args['input_real'] + '/*'
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))

    print(len(gt_list))
    print(len(input_list))

    assert len(gt_list) == len(input_list)
    n = len(gt_list)

    ssim, psnr, lpips_val = 0, 0, 0

    for i in tqdm(range(n)):
        # print(gt_list[i], input_list[i])
        img_gt = io.imread(gt_list[i])
        img_input = io.imread(input_list[i])
        ssim += compare_ssim(img_gt, img_input, multichannel=True)
        psnr += compare_psnr(img_gt, img_input, data_range=255)
        lpips_val += compare_lpips(img_gt, img_input, loss_fn_alex)

    ssim /= n
    psnr /= n
    lpips_val /= n

    print(f"psnr: {psnr}, ssim: {ssim}, lpips: {lpips_val}")


def calculate_metrics_syn(args):
    loss_fn_alex = lpips.LPIPS(net='alex')
    gt_folder = args['gt_syn'] + '/*'
    input_folder = args['input_syn'] + '/*'
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))

    print(len(gt_list))
    print(len(input_list))

    assert len(gt_list) == len(input_list)
    n = len(gt_list)

    ssim, psnr, lpips_val = 0, 0, 0

    for i in tqdm(range(n)):
        # print(gt_list[i], input_list[i])
        img_gt = io.imread(gt_list[i])
        img_input = io.imread(input_list[i])
        ssim += compare_ssim(img_gt, img_input, multichannel=True)
        psnr += compare_psnr(img_gt, img_input, data_range=255)
        lpips_val += compare_lpips(img_gt, img_input, loss_fn_alex)

    ssim /= n
    psnr /= n
    lpips_val /= n

    print(f"psnr: {psnr}, ssim: {ssim}, lpips: {lpips_val}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_real',type=str,default=None)
    parser.add_argument('--gt_real',type=str,default=None)
    parser.add_argument('--input_syn', type=str, default=None)
    parser.add_argument('--gt_syn', type=str, default=None)
    args = vars(parser.parse_args())

    args['input_real'] = "/home/ipprlab/Projects/flare/Flare7K/result/real/our/blend/"
    args['input_real'] = "result/real/our/blend/"
    args['gt_real'] = "datasets/Flare7k/test_data/real/gt/"

    args['input_syn'] = "/home/ipprlab/Projects/flare/Flare7K/result/synthetic/our/blend/"
    args['input_syn'] = "result/synthetic/Uformer/blend/"
    args['gt_syn'] = "datasets/Flare7k/test_data/synthetic/gt/"

    calculate_metrics_real(args)

    calculate_metrics_syn(args)
