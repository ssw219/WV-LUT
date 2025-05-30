import logging
import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torchvision.models import vgg16

# ============== Loss Functions ==============

class LossNetwork(torch.nn.Module):
    """VGG-based loss network for perceptual loss calculation."""
    
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            "3": "relu1_2",
            "8": "relu2_2",
            "15": "relu3_3"
        }

    def output_features(self, x):
        """Extract features from VGG layers."""
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        """Calculate perceptual loss between predicted and ground truth images."""
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))
        return sum(loss) / len(loss)


def net_loss(pred, target):
    """Calculate perceptual loss using VGG features."""
    # Initialize VGG model if not already done
    if not hasattr(net_loss, 'vgg_model'):
        vgg_model = vgg16(pretrained=True)
        vgg_model = vgg_model.features[:16].eval()  # Use only first 16 layers
        # Move model to the same device as input
        device = pred.device
        vgg_model = vgg_model.to(device)
        # Freeze VGG parameters
        for param in vgg_model.parameters():
            param.requires_grad = False
        net_loss.vgg_model = vgg_model

    # Normalize input images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
    
    pred_norm = (pred - mean) / std
    target_norm = (target - mean) / std

    # Extract features
    with torch.no_grad():
        pred_features = net_loss.vgg_model(pred_norm)
        target_features = net_loss.vgg_model(target_norm)

    # Calculate MSE loss between features
    return F.mse_loss(pred_features, target_features)


def calculate_lpips(img1, img2):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity) between two images."""
    img1 = lpips.im2tensor(img1)
    img2 = lpips.im2tensor(img2)
    loss_fn_alex = lpips.LPIPS(net="alex", verbose=False)
    return loss_fn_alex(img1, img2).item()

# ============== Image Quality Metrics ==============

def calculate_psnr(y_true, y_pred, shave_border=4):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images."""
    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    return 20 * np.log10(255.0 / rmse)


def calculate_ssim(img1, img2):
    """Calculate SSIM (Structural Similarity Index) between two images.
    
    Args:
        img1, img2: Input images in range [0, 255]
    
    Returns:
        float: Mean SSIM value
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    if img1.ndim == 2:
        return _ssim_single_channel(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(_ssim_single_channel(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return _ssim_single_channel(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def _ssim_single_channel(img1, img2):
    """Calculate SSIM for single channel images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()

# ============== Logging ==============

def logger_info(logger_name, log_path="default_logger.log"):
    """Setup logger with file and stream handlers."""
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print("LogHandlers exist!")
        return
        
    print("LogHandlers setup!")
    level = logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d : %(message)s",
        datefmt="%y-%m-%d %H:%M:%S"
    )
    
    # File handler
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    
    # Stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.addHandler(sh)
