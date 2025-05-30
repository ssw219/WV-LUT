import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

sys.path.insert(0, "../")

from common.architecture import *
from common.finetune import Finetune
from common.option import TestOptions
from common.utils import (
    logger_info,
    calculate_ssim,
    net_loss,
)
from script.data import LowLightBenchmark

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips



class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB, gray_scale=False):
        if gray_scale:
            score, diff = ssim(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True, multichannel=True)
            # score, diff = ssim(imgA, imgB, full=True, multichannel=False)
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        else:
            # print(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB).shape)
            # print(imgA.shape, imgB.shape)
            score, diff = ssim(imgA, imgB, full=True, multichannel=True, channel_axis=2)
        return score

    def psnr(self, imgA, imgB):
        psnr_val = psnr(imgA, imgB)
        return psnr_val

def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1

def valid_steps(model_G, valid, datasets, result_path, ifsave=False, ifGT=False):
    measure = Measure(use_gpu=False)
    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnrs = []
            ssims = []
            lpipss = []
            files = valid.files[datasets[i]]

            # result_path = "eval/{}_{}/".format(expDir.split("/")[-1], datasets[i])
            if not os.path.isdir(result_path) and ifsave:
                os.makedirs(result_path)

            for j in range(len(files)):
                key = datasets[i] + "_" + files[j][:-4]
                lb = valid.images[key]
                input_im = valid.images[key + "_low"]

                input_im = input_im.astype(np.float32) / 255.0
                im = torch.Tensor(
                    np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)
                ).cuda()

                if ifGT:

                # pred = model_G(im) * 255.0
                # t1 = time.time()
                    pred = model_G(im) 
                    # t2 = time.time()
                    # print(t2-t1)
                    mean_out = pred.reshape(pred.shape[0],-1).mean(dim=1)
                    mean_gt = cv2.cvtColor(lb.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()/255
                    pred = torch.clamp(pred*(mean_gt/mean_out), 0, 1) * 255.0
                else:
                    pred = model_G(im) * 255.0

                pred = np.transpose(np.squeeze(pred.data.cpu().numpy(), 0), [1, 2, 0])
                pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                # left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
                # psnrs.append(
                #     PSNR(left, right, scale)
                # )  # single channel, no scale change
                psnr1, ssim1, lpips1 = measure.measure(lb, pred)

                psnrs.append(psnr1)
                ssims.append(ssim1)
                lpipss.append(lpips1)
                # psnrs.append(cal_psnr(lb, pred))
                # ssims.append(calculate_ssim(lb, pred))
                # lpipss.append(cal_lpips(lb, pred))
                print(
                    "{}: PSNR: {:02f} | SSIM: {:02f} | LPIPS: {:02f}".format(
                        key, psnrs[-1], ssims[-1], lpipss[-1]
                    )
                )
                input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(np.uint8)
                
                if ifsave:
                    Image.fromarray(input_img).save(
                        os.path.join(
                            result_path, "{}_input.png".format(key.split("_")[-1])
                        )
                    )
                    Image.fromarray(lb.astype(np.uint8)).save(
                        os.path.join(
                            result_path, "{}_gt.png".format(key.split("_")[-1])
                        )
                    )

                    Image.fromarray(pred).save(
                        os.path.join(
                            result_path, "{}_net.png".format(key.split("_")[-1])
                        )
                    )

            print(
                "Dataset {} | AVG Val PSNR: {:02f} | AVG Val SSIM: {:02f} | AVG Val LPIPS: {:02f}".format(
                    datasets[i],
                    np.mean(np.asarray(psnrs)),
                    np.mean(np.asarray(ssims)),
                    np.mean(np.asarray(lpipss)),
                )
            )
            return np.mean(np.asarray(psnrs))


if __name__ == "__main__":
    # Parse command line arguments
    opt = TestOptions(debug=False).parse()
    
    # Set default values if not provided
    valDir = opt.testDir if hasattr(opt, 'testDir') else "../data/LowLightBenchmark"
    expDir = opt.expDir if hasattr(opt, 'expDir') else "../models/WVLUT_shared"
    result_path = opt.resultRoot if hasattr(opt, 'resultRoot') else "../eval/test/"
    ifsave = opt.ifsave if hasattr(opt, 'ifsave') else False
    ifGT = opt.ifGT if hasattr(opt, 'ifGT') else False
    scale_show = opt.scale_show if hasattr(opt, 'scale_show') else False
    
    datasets = opt.valDataset if hasattr(opt, 'valDataset') else ["LOL_v1_val"]
    
    iter_num = opt.loadIter if hasattr(opt, 'loadIter') else 150000
    
    if opt.model == "WVLUT":
        model_G = WVLUT(nf=opt.nf, model_type=opt.model_type).cuda()
    elif opt.model == "LUT":
        model_G = Finetune(
            lut_folder=os.path.join(expDir, "luts"),
            modes=["c", "d", "y"],
            interval=4,
            model_path=None,
            model_type=opt.model_type,
            freeze_non_lut=True
        ).cuda()
    else:
        model_G = WVLUT(nf=opt.nf, model_type='shared').cuda()
    
    lm = torch.load(os.path.join(expDir, f"Model_{iter_num:06d}.pth"))
    model_G.load_state_dict(lm.state_dict(), strict=True)
    
    # Run evaluation
    valid = LowLightBenchmark(valDir, datasets=datasets)
    psnrtt = valid_steps(
        model_G, valid, datasets=datasets, result_path=result_path, 
        ifsave=ifsave, ifGT=ifGT
    )
