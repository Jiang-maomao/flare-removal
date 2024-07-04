import warnings

import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from tqdm import tqdm

from config import Config
from data import get_validation_data
from models import *
from utils import *

from data.dataset_RGB import Flare_Pair_Loader

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)


if not os.path.exists(opt.TRAINING.SAVE_DIR):
    os.makedirs(opt.TRAINING.SAVE_DIR)


def train():
    # Accelerate
    accelerator = Accelerator(
        log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
    device = accelerator.device
    config = {
        "dataset": opt.TRAINING.TRAIN_DIR
    }
    accelerator.init_trackers("shadow", config=config)

    # Data Loader
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir_1 = opt.TRAINING.VAL_DIR_1
    val_dir_2 = opt.TRAINING.VAL_DIR_2

    # for phase, dataset_opt in opt['datasets'].items():
    #     if phase == 'train':
    #         #dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
    train_dataset = Flare_Pair_Loader(
        opt.datasets.image_path, opt.datasets.img_size, opt.datasets.transform_flare, opt.datasets.scattering_dict, opt.datasets.reflective_dict)
    # train_dataset = get_training_data(train_dir, opt.MODEL.FILM, {
    #                                   'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=32,
                             drop_last=False, pin_memory=True)
    print(trainloader)
    val_dataset_1 = get_validation_data(val_dir_1, opt.MODEL.FILM, {
                                        'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': opt.TRAINING.ORI})
    val_dataset_2 = get_validation_data(val_dir_2, opt.MODEL.FILM, {
                                        'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': opt.TRAINING.ORI})
    testloader_1 = DataLoader(dataset=val_dataset_1, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                              pin_memory=True)
    testloader_2 = DataLoader(dataset=val_dataset_2, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                              pin_memory=True)

    # Model & Loss
    model = Model()
    criterion_ssim = structural_similarity_index_measure
    criterion_psnr = torch.nn.MSELoss()
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
    criterion_lpips.to(device=device)

    # Optimizer & Scheduler
    optimizer_b = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    trainloader, testloader_1, testloader_2 = accelerator.prepare(
        trainloader, testloader_1, testloader_2)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    start_epoch = 1
    best_epoch_1 = 1
    best_epoch_2 = 1
    best_iter_1 = 1
    best_iter_2 = 1
    best_psnr_1 = 23
    best_psnr_2 = 25
    size = len(testloader_1)
    eval_now = 50
    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()

        for i, data in enumerate(tqdm(trainloader)):
            # print(data)
            # get the inputs; data is a list of [target, input, filename]
            inp = data[0].contiguous()
            tar = data[1]

            # forward
            optimizer_b.zero_grad()
            res = model(inp)
            #print ("res",res)
            #print ("tar",tar)
            loss_psnr = criterion_psnr(res, tar)
            loss_ssim = 1 - criterion_ssim(res, tar, data_range=1)
            loss_lpips = criterion_lpips(res.mul(2).sub(
                1).clamp(-1, 1), tar.mul(2).sub(1).clamp(-1, 1))

            train_loss = loss_psnr + 0.3 * loss_ssim + 0.7 * loss_lpips

            # backward
            accelerator.backward(train_loss)
            optimizer_b.step()

            #### Evaluation ####
            if (i + 1) % eval_now == 0 and i > 0:
                with torch.no_grad():
                    model.eval()
                    psnr1 = 0
                    ssim1 = 0
                    lpips1 = 0
                    for idx, test_data in enumerate(tqdm(testloader_1)):
                        # get the inputs; data is a list of [targets, inputs, filename]
                        inp = test_data[0].contiguous()
                        tar = test_data[1]

                        # with torch.no_grad():
                        #     res = model(inp)
                        res = model(inp)

                        res, tar = accelerator.gather((res, tar))

                        psnr1 += peak_signal_noise_ratio(res,
                                                         tar, data_range=1)
                        ssim1 += criterion_ssim(res, tar)
                        lpips1 += criterion_lpips(res.mul(2).sub(1).clamp(-1, 1),
                                                  tar.mul(2).sub(1).clamp(-1, 1))

                    psnr1 /= size
                    ssim1 /= size
                    lpips1 /= size

                    if psnr1 > best_psnr_1:
                        # save model
                        best_epoch_1 = epoch
                        best_psnr_1 = psnr1
                        best_iter_1 = i
                        save_checkpoint({
                            'state_dict': model.state_dict(),
                        }, epoch, i, opt.MODEL.SESSION, psnr1, opt.TRAINING.SAVE_DIR)

                    accelerator.log({
                        "PSNR": psnr1,
                        "SSIM": ssim1,
                        "LPIPS": lpips1
                    }, step=epoch)

                    print(
                        "REAL: epoch: {}, iter: {}, PSNR: {}, SSIM: {}, LPIPS: {}, best PSNR: {}, best epoch_1: {}, best iter_1: {}"
                        .format(epoch, i, psnr1, ssim1, lpips1, best_psnr_1, best_epoch_1, best_iter_1))

                    psnr2 = 0
                    ssim2 = 0
                    lpips2 = 0
                    for idx, test_data in enumerate(tqdm(testloader_2)):
                        # get the inputs; data is a list of [targets, inputs, filename]
                        inp = test_data[0].contiguous()
                        tar = test_data[1]

                        # with torch.no_grad():
                        #     res = model(inp)
                        res = model(inp)

                        res, tar = accelerator.gather((res, tar))

                        psnr2 += peak_signal_noise_ratio(res,
                                                         tar, data_range=1)
                        ssim2 += criterion_ssim(res, tar)
                        lpips2 += criterion_lpips(res.mul(2).sub(1).clamp(-1, 1),
                                                  tar.mul(2).sub(1).clamp(-1, 1))

                    psnr2 /= size
                    ssim2 /= size
                    lpips2 /= size

                    if psnr2 > best_psnr_2:
                        # save model
                        best_epoch_2 = epoch
                        best_psnr_2 = psnr2
                        best_iter_2 = i
                        save_checkpoint({
                            'state_dict': model.state_dict(),
                        }, epoch, i, opt.MODEL.SESSION, psnr2, opt.TRAINING.SAVE_DIR)

                    accelerator.log({
                        "PSNR": psnr2,
                        "SSIM": ssim2,
                        "LPIPS": lpips2
                    }, step=epoch)

                    print(
                        "SYN: epoch: {}, iter: {}, PSNR: {}, SSIM: {}, LPIPS: {}, best PSNR: {}, best epoch_2: {}, best iter_2: {}"
                        .format(epoch, i, psnr2, ssim2, lpips2, best_psnr_2, best_epoch_2, best_iter_2))

                    # psnr_val_rgb = []
                    # for ii, data_val in enumerate((val_loader), 0):
                    #     target = data_val[0].cuda()
                    #     input_ = data_val[1].cuda()
                    #     with torch.cuda.amp.autocast():
                    #         restored = model_restoration(input_)
                    #     restored = torch.clamp(restored, 0, 1)
                    #     psnr_val_rgb.append(utils.batch_PSNR(
                    #         restored, target, False).item())
                    # psnr_val_rgb = sum(psnr_val_rgb) / size

                    # if psnr_val_rgb > best_psnr:
                    #     best_psnr = psnr_val_rgb
                    #     best_epoch = epoch
                    #     best_iter = i
                    #     torch.save({'epoch': epoch,
                    #                 'state_dict': model_restoration.state_dict(),
                    #                 'optimizer': optimizer.state_dict()
                    #                 }, os.path.join(model_dir, "model_best.pth"))

                    # print(
                    #     "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (
                    #         epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
                    # with open(logname, 'a') as f:
                    #     f.write(
                    #         "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] "
                    #         % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')
                    model.train()
                    torch.cuda.empty_cache()

        scheduler_b.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr = 0
            ssim = 0
            lpips = 0
            for idx, test_data in enumerate(tqdm(testloader_1)):
                # get the inputs; data is a list of [targets, inputs, filename]
                inp = test_data[0].contiguous()
                tar = test_data[1]

                with torch.no_grad():
                    res = model(inp)

                res, tar = accelerator.gather((res, tar))

                psnr += peak_signal_noise_ratio(res, tar, data_range=1)
                ssim += criterion_ssim(res, tar)
                lpips += criterion_lpips(res.mul(2).sub(1).clamp(-1, 1),
                                         tar.mul(2).sub(1).clamp(-1, 1))

            psnr /= size
            ssim /= size
            lpips /= size

            if psnr > best_psnr_1:
                # save model
                best_epoch_1 = epoch
                best_psnr_1 = psnr
                save_checkpoint({
                    'state_dict': model.state_dict(),
                }, epoch, 1, opt.MODEL.SESSION, psnr, opt.TRAINING.SAVE_DIR)

            accelerator.log({
                "PSNR": psnr,
                "SSIM": ssim,
                "LPIPS": lpips
            }, step=epoch)

            print(
                "REAL: epoch: {}, PSNR: {}, SSIM: {}, LPIPS: {}, best PSNR: {}, best epoch_1: {}"
                .format(epoch, psnr, ssim, lpips, best_psnr_1, best_epoch_1))

            save_checkpoint({
                'state_dict': model.state_dict(),
            }, epoch, 1, opt.MODEL.SESSION, psnr, opt.TRAINING.SAVE_DIR)

        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr = 0
            ssim = 0
            lpips = 0
            for idx, test_data in enumerate(tqdm(testloader_2)):
                # get the inputs; data is a list of [targets, inputs, filename]
                inp = test_data[0].contiguous()
                tar = test_data[1]

                with torch.no_grad():
                    res = model(inp)

                res, tar = accelerator.gather((res, tar))

                psnr += peak_signal_noise_ratio(res, tar, data_range=1)
                ssim += criterion_ssim(res, tar)
                lpips += criterion_lpips(res.mul(2).sub(1).clamp(-1, 1),
                                         tar.mul(2).sub(1).clamp(-1, 1))

            psnr /= size
            ssim /= size
            lpips /= size

            if psnr > best_psnr_2:
                # save model
                best_epoch_2 = epoch
                best_psnr_2 = psnr
                save_checkpoint({
                    'state_dict': model.state_dict(),
                }, epoch, 2, opt.MODEL.SESSION, psnr, opt.TRAINING.SAVE_DIR)

            accelerator.log({
                "PSNR": psnr,
                "SSIM": ssim,
                "LPIPS": lpips
            }, step=epoch)

            print(
                "SYN: epoch: {}, PSNR: {}, SSIM: {}, LPIPS: {}, best PSNR: {}, best epoch_2: {}"
                .format(epoch, psnr, ssim, lpips, best_psnr_2, best_epoch_2))

            save_checkpoint({
                'state_dict': model.state_dict(),
            }, epoch, 2, opt.MODEL.SESSION, psnr, opt.TRAINING.SAVE_DIR)

    accelerator.end_training()


if __name__ == '__main__':
    train()
