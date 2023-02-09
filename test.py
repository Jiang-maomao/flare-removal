import warnings

from accelerate import Accelerator
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, MeanSquaredError
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_validation_data
from models import *
from utils import *

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)


def test():
    accelerator = Accelerator()
    device = accelerator.device

    # Data Loader
    val_dir = opt.TRAINING.VAL_DIR

    val_dataset = get_validation_data(val_dir, opt.MODEL.FILM, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': True})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Metrics
    model = Model()
    ssim = SSIM(data_range=1, size_average=True, channel=3).to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    rmse = MeanSquaredError(squared=False).to(device)

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_rmse = 0
    for idx, test_data in enumerate(tqdm(testloader)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        tar = test_data[1]

        with torch.no_grad():
            res = model(inp)

        save_image(res, os.path.join(os.getcwd(), "result", test_data[3][0] + '_pred.png'))
        save_image(tar, os.path.join(os.getcwd(), "result", test_data[3][0] + '_gt.png'))

        stat_psnr += psnr(res, tar)
        stat_ssim += ssim(res, tar)
        stat_rmse += rmse(torch.mul(res, 255), torch.mul(tar, 255))

    stat_psnr /= size
    stat_ssim /= size
    stat_rmse /= size

    print("PSNR: {}, SSIM: {}, RMSE: {}".format(stat_psnr, stat_ssim, stat_rmse))


if __name__ == '__main__':
    test()
