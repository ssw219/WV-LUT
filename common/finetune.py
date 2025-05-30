import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "../")  # run under the current directory
from common.architecture import *
from common.GAM import GAM_pll_4

mode_pad_dict = {"c": 1, "d": 2, "y": 2}

class Finetune(nn.Module):
    """ PyTorch version of WVLUT for LUT-aware fine-tuning for low-light image enhancement.
    
    Args:
        lut_folder (str): Path to the folder containing LUT files
        modes (list): List of modes to use, typically ['c', 'd', 'y']
        interval (int): Bit interval for LUT
        model_path (str): Path to pretrained model for GAM
        model_type (str): Type of model architecture
            - 'shared': Uses shared weights across RGB channels (default)
            - 'wo_shared': Uses separate weights for each RGB channel
        freeze_non_lut (bool): Whether to freeze all parameters except LUT weights
    """

    def __init__(self, lut_folder, modes, interval=4, model_path=None, model_type='shared', freeze_non_lut=True):
        super(Finetune, self).__init__()
        self.interval = interval
        self.modes = modes
        self.stages = 3
        self.model_type = model_type

        # Load 4D LUTs for stages 1 and 2
        for s in range(2):  # First two stages
            stage = s + 1
            for mode in modes:
                if model_type == 'shared':
                    lut_path = os.path.join(lut_folder, "LUT_{}bit_int8_s{}_{}.npy".format(interval, str(stage), mode))
                    key = "s{}_{}".format(str(stage), mode)
                    lut_arr = np.load(lut_path).reshape(-1, 1).astype(np.float32) / 127.0
                    self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))
                else:  # wo_shared
                    for c in ['r', 'g', 'b']:
                        lut_path = os.path.join(lut_folder, "LUT_{}bit_int8_s{}_{}_{}.npy".format(
                            interval, str(stage), mode, c))
                        key = "s{}_{}_{}".format(str(stage), mode, c)
                        lut_arr = np.load(lut_path).reshape(-1, 1).astype(np.float32) / 127.0
                        self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        self.scale_s1 = nn.ParameterDict({
            'c': nn.Parameter(torch.ones(1), requires_grad=True),
            'd': nn.Parameter(torch.ones(1), requires_grad=True),
            'y': nn.Parameter(torch.ones(1), requires_grad=True)
        })

        self.scale_s2 = nn.ParameterDict({
            'c': nn.Parameter(torch.ones(1), requires_grad=True),
            'd': nn.Parameter(torch.ones(1), requires_grad=True),
            'y': nn.Parameter(torch.ones(1), requires_grad=True)
        })

        # Load pretrained model weights for scale_s1 and scale_s2 if provided
        if model_path is not None and os.path.exists(model_path):
            pretrained_model = torch.load(model_path)
            # Extract scale_s1 and scale_s2 parameters from pretrained model
            scale_s1_state_dict = {k: v for k, v in pretrained_model.state_dict().items() if k.startswith('scale_s1')}
            scale_s2_state_dict = {k: v for k, v in pretrained_model.state_dict().items() if k.startswith('scale_s2')}
            # Load scale_s1 and scale_s2 parameters
            if scale_s1_state_dict:
                self.scale_s1.load_state_dict(scale_s1_state_dict)
                print("Loaded scale_s1 parameters from pretrained model")
            if scale_s2_state_dict:
                self.scale_s2.load_state_dict(scale_s2_state_dict)
                print("Loaded scale_s2 parameters from pretrained model")

        # Load 3D LUT for stage 3
        lut3d_path = os.path.join(lut_folder, "LUT_{}bit_int8_s3_3d.npy".format(interval))
        lut3d_arr = np.load(lut3d_path).astype(np.float32) / 127.0
        self.register_parameter(name="weight_s3_3d", param=torch.nn.Parameter(torch.Tensor(lut3d_arr)))

        # Initialize GAM module
        self.GAM = GAM_pll_4(in_channels=6)

        # Load pretrained model weights for GAM if provided
        if model_path is not None and os.path.exists(model_path):
            pretrained_model = torch.load(model_path)
            # Extract GAM parameters from pretrained model
            gam_state_dict = {k: v for k, v in pretrained_model.state_dict().items() if k.startswith('GAM')}
            # Load GAM parameters
            if gam_state_dict:
                self.GAM.load_state_dict({k[4:]: v for k, v in gam_state_dict.items()})
                print("Loaded GAM parameters from pretrained model")

        # Freeze all parameters except LUT weights if specified
        if freeze_non_lut:
            self._freeze_non_lut_parameters()

    def _freeze_non_lut_parameters(self):
        """Freeze all parameters except LUT weights."""
        for name, param in self.named_parameters():
            # Only keep LUT weights trainable
            if not name.startswith('weight_'):
                param.requires_grad = False
            else:
                param.requires_grad = True
                print(f"Keeping {name} trainable")

    def apply_color(self, image, ccm):
        """Apply color correction matrix to the image."""
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    @staticmethod
    def round_func(input):
        """Backward Pass Differentiable Approximation (BPDA).

        This is equivalent to replacing round function (non-differentiable)
        with an identity function (differentiable) only when backward.
        """
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def _prepare_3d_lut_weights(self, weight):
        """Prepare and scale 3D LUT weights.
        
        Args:
            weight: Raw 3D LUT weights
            
        Returns:
            Scaled and clamped weights
        """
        weight = weight * 127
        weight = self.round_func(weight)
        return torch.clamp(weight, -127, 127)

    def _get_3d_indices_and_fractions(self, img_flat, q, L):
        """Get integer indices and fractional parts for 3D LUT interpolation.
        
        Args:
            img_flat: Flattened input image tensor [B*H*W, 3]
            q: Quantization step size
            L: LUT size
            
        Returns:
            Tuple of (r1, g1, b1, r2, g2, b2, fr, fg, fb)
        """
        # Extract RGB channels
        r, g, b = img_flat[:, 0], img_flat[:, 1], img_flat[:, 2]

        # Get integer indices
        r1 = torch.div(r, q, rounding_mode='floor').type(torch.int64)
        g1 = torch.div(g, q, rounding_mode='floor').type(torch.int64)
        b1 = torch.div(b, q, rounding_mode='floor').type(torch.int64)

        # Get fractional parts
        fr = r % q
        fg = g % q
        fb = b % q

        # Get next indices
        r2 = torch.clamp(r1 + 1, 0, L - 1)
        g2 = torch.clamp(g1 + 1, 0, L - 1)
        b2 = torch.clamp(b1 + 1, 0, L - 1)

        return r1, g1, b1, r2, g2, b2, fr, fg, fb

    def _get_3d_lut_corners(self, weight, indices, L):
        """Get the 8 corner values from the 3D LUT.
        
        Args:
            weight: 3D LUT weights
            indices: Tuple of (r1, g1, b1, r2, g2, b2)
            L: LUT size
            
        Returns:
            Tuple of 8 corner values (p000 through p111)
        """
        r1, g1, b1, r2, g2, b2 = indices

        # Get all 8 corner values
        p000 = weight[r1 * L * L + g1 * L + b1, :, 0, 0]
        p001 = weight[r1 * L * L + g1 * L + b2, :, 0, 0]
        p010 = weight[r1 * L * L + g2 * L + b1, :, 0, 0]
        p011 = weight[r1 * L * L + g2 * L + b2, :, 0, 0]
        p100 = weight[r2 * L * L + g1 * L + b1, :, 0, 0]
        p101 = weight[r2 * L * L + g1 * L + b2, :, 0, 0]
        p110 = weight[r2 * L * L + g2 * L + b1, :, 0, 0]
        p111 = weight[r2 * L * L + g2 * L + b2, :, 0, 0]

        return p000, p001, p010, p011, p100, p101, p110, p111

    def _perform_3d_interpolation(self, corners, fractions, q):
        """Perform trilinear interpolation using corner values and fractional parts.
        
        Args:
            corners: Tuple of 8 corner values
            fractions: Tuple of (fr, fg, fb)
            q: Quantization step size
            
        Returns:
            Interpolated output tensor
        """
        p000, p001, p010, p011, p100, p101, p110, p111 = corners
        fr, fg, fb = fractions

        # Normalize fractional parts
        fr = fr.unsqueeze(1) / q
        fg = fg.unsqueeze(1) / q
        fb = fb.unsqueeze(1) / q

        # First interpolation along r-axis
        c00 = p000 * (1 - fr) + p100 * fr
        c01 = p001 * (1 - fr) + p101 * fr
        c10 = p010 * (1 - fr) + p110 * fr
        c11 = p011 * (1 - fr) + p111 * fr

        # Second interpolation along g-axis
        c0 = c00 * (1 - fg) + c10 * fg
        c1 = c01 * (1 - fg) + c11 * fg

        # Final interpolation along b-axis
        return c0 * (1 - fb) + c1 * fb

    def InterpTorch3DBatch(self, weight, img_in):
        """3D lookup table interpolation for stage 3.

        Args:
            weight: 3D lookup table weights
            img_in: Input image tensor [B, 3, H, W]

        Returns:
            Interpolated output tensor [B, 3, H, W]
        """
        # Get input dimensions
        B, C, H, W = img_in.shape

        # Prepare weights
        weight = self._prepare_3d_lut_weights(weight)

        # Calculate quantization parameters
        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        # Reshape input to [B*H*W, 3]
        img_flat = img_in.permute(0, 2, 3, 1).reshape(-1, 3)

        # Get indices and fractional parts
        indices = self._get_3d_indices_and_fractions(img_flat, q, L)
        r1, g1, b1, r2, g2, b2, fr, fg, fb = indices

        # Get LUT corner values
        corners = self._get_3d_lut_corners(weight, (r1, g1, b1, r2, g2, b2), L)

        # Perform trilinear interpolation
        out = self._perform_3d_interpolation(corners, (fr, fg, fb), q)

        # Reshape back to [B, C, H, W]
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out

    def _get_sampling_indices(self, img_in, mode, h, w, q):
        """Get sampling indices and fractional parts for 4D LUT interpolation.
        
        Args:
            img_in: Input image tensor
            mode: Sampling mode ('c', 'd', or 'y')
            h, w: Height and width after border removal
            q: Quantization step size
            
        Returns:
            Tuple of (indices_a1, indices_b1, indices_c1, indices_d1,
                     frac_a, frac_b, frac_c, frac_d)
        """
        # Calculate LUT size
        L = 2 ** (8 - self.interval) + 1
        
        if mode == "c":
            # Extract indices for mode 'c'
            indices_a1 = torch.div(img_in[:, :, 0:0 + h, 0:0 + w], q, rounding_mode='floor').type(torch.int64)
            indices_b1 = torch.div(img_in[:, :, 0:0 + h, 1:1 + w], q, rounding_mode='floor').type(torch.int64)
            indices_c1 = torch.div(img_in[:, :, 1:1 + h, 0:0 + w], q, rounding_mode='floor').type(torch.int64)
            indices_d1 = torch.div(img_in[:, :, 1:1 + h, 1:1 + w], q, rounding_mode='floor').type(torch.int64)

            # Extract fractional parts
            frac_a = img_in[:, :, 0:0 + h, 0:0 + w] % q
            frac_b = img_in[:, :, 0:0 + h, 1:1 + w] % q
            frac_c = img_in[:, :, 1:1 + h, 0:0 + w] % q
            frac_d = img_in[:, :, 1:1 + h, 1:1 + w] % q

        elif mode == "d":
            # Extract indices for mode 'd'
            indices_a1 = torch.div(img_in[:, :, 0:0 + h, 0:0 + w], q, rounding_mode='floor').type(torch.int64)
            indices_b1 = torch.div(img_in[:, :, 0:0 + h, 2:2 + w], q, rounding_mode='floor').type(torch.int64)
            indices_c1 = torch.div(img_in[:, :, 2:2 + h, 0:0 + w], q, rounding_mode='floor').type(torch.int64)
            indices_d1 = torch.div(img_in[:, :, 2:2 + h, 2:2 + w], q, rounding_mode='floor').type(torch.int64)

            # Extract fractional parts
            frac_a = img_in[:, :, 0:0 + h, 0:0 + w] % q
            frac_b = img_in[:, :, 0:0 + h, 2:2 + w] % q
            frac_c = img_in[:, :, 2:2 + h, 0:0 + w] % q
            frac_d = img_in[:, :, 2:2 + h, 2:2 + w] % q

        elif mode == "y":
            # Extract indices for mode 'y'
            indices_a1 = torch.div(img_in[:, :, 0:0 + h, 0:0 + w], q, rounding_mode='floor').type(torch.int64)
            indices_b1 = torch.div(img_in[:, :, 1:1 + h, 1:1 + w], q, rounding_mode='floor').type(torch.int64)
            indices_c1 = torch.div(img_in[:, :, 1:1 + h, 2:2 + w], q, rounding_mode='floor').type(torch.int64)
            indices_d1 = torch.div(img_in[:, :, 2:2 + h, 1:1 + w], q, rounding_mode='floor').type(torch.int64)

            # Extract fractional parts
            frac_a = img_in[:, :, 0:0 + h, 0:0 + w] % q
            frac_b = img_in[:, :, 1:1 + h, 1:1 + w] % q
            frac_c = img_in[:, :, 1:1 + h, 2:2 + w] % q
            frac_d = img_in[:, :, 2:2 + h, 1:1 + w] % q
        else:
            raise ValueError(f"Mode {mode} not implemented.")

        # Clamp indices to valid range [0, L-1]
        indices_a1 = torch.clamp(indices_a1, 0, L-1)
        indices_b1 = torch.clamp(indices_b1, 0, L-1)
        indices_c1 = torch.clamp(indices_c1, 0, L-1)
        indices_d1 = torch.clamp(indices_d1, 0, L-1)

        return indices_a1, indices_b1, indices_c1, indices_d1, frac_a, frac_b, frac_c, frac_d

    def _get_lut_corners(self, weight, indices_a1, indices_b1, indices_c1, indices_d1, L, upscale):
        """Get the 16 corner values from the LUT for interpolation.
        
        Args:
            weight: LUT weights
            indices_a1, indices_b1, indices_c1, indices_d1: Base indices
            L: LUT size
            upscale: Upscaling factor
            
        Returns:
            Tuple of 16 corner values (p0000 through p1111)
        """
        # Calculate next indices
        indices_a2 = indices_a1 + 1
        indices_b2 = indices_b1 + 1
        indices_c2 = indices_c1 + 1
        indices_d2 = indices_d1 + 1

        # Get all 16 corner values
        p0000 = weight[indices_a1.flatten() * L * L * L + indices_b1.flatten() * L * L + indices_c1.flatten() * L + indices_d1.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p0001 = weight[indices_a1.flatten() * L * L * L + indices_b1.flatten() * L * L + indices_c1.flatten() * L + indices_d2.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p0010 = weight[indices_a1.flatten() * L * L * L + indices_b1.flatten() * L * L + indices_c2.flatten() * L + indices_d1.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p0011 = weight[indices_a1.flatten() * L * L * L + indices_b1.flatten() * L * L + indices_c2.flatten() * L + indices_d2.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p0100 = weight[indices_a1.flatten() * L * L * L + indices_b2.flatten() * L * L + indices_c1.flatten() * L + indices_d1.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p0101 = weight[indices_a1.flatten() * L * L * L + indices_b2.flatten() * L * L + indices_c1.flatten() * L + indices_d2.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p0110 = weight[indices_a1.flatten() * L * L * L + indices_b2.flatten() * L * L + indices_c2.flatten() * L + indices_d1.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p0111 = weight[indices_a1.flatten() * L * L * L + indices_b2.flatten() * L * L + indices_c2.flatten() * L + indices_d2.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p1000 = weight[indices_a2.flatten() * L * L * L + indices_b1.flatten() * L * L + indices_c1.flatten() * L + indices_d1.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p1001 = weight[indices_a2.flatten() * L * L * L + indices_b1.flatten() * L * L + indices_c1.flatten() * L + indices_d2.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p1010 = weight[indices_a2.flatten() * L * L * L + indices_b1.flatten() * L * L + indices_c2.flatten() * L + indices_d1.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p1011 = weight[indices_a2.flatten() * L * L * L + indices_b1.flatten() * L * L + indices_c2.flatten() * L + indices_d2.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p1100 = weight[indices_a2.flatten() * L * L * L + indices_b2.flatten() * L * L + indices_c1.flatten() * L + indices_d1.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p1101 = weight[indices_a2.flatten() * L * L * L + indices_b2.flatten() * L * L + indices_c1.flatten() * L + indices_d2.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p1110 = weight[indices_a2.flatten() * L * L * L + indices_b2.flatten() * L * L + indices_c2.flatten() * L + indices_d1.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))
        p1111 = weight[indices_a2.flatten() * L * L * L + indices_b2.flatten() * L * L + indices_c2.flatten() * L + indices_d2.flatten()
                       ].reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], indices_a1.shape[3], upscale, upscale))

        return p0000, p0001, p0010, p0011, p0100, p0101, p0110, p0111, \
               p1000, p1001, p1010, p1011, p1100, p1101, p1110, p1111

    def _perform_interpolation(self, corners, frac_a, frac_b, frac_c, frac_d, q, sz):
        """Perform 4D interpolation using corner values and fractional parts.
        
        Args:
            corners: Tuple of 16 corner values
            frac_a, frac_b, frac_c, frac_d: Fractional parts
            q: Quantization step size
            sz: Total number of elements
            
        Returns:
            Interpolated output tensor
        """
        p0000, p0001, p0010, p0011, p0100, p0101, p0110, p0111, \
        p1000, p1001, p1010, p1011, p1100, p1101, p1110, p1111 = corners

        # Reshape all tensors for interpolation
        out = torch.zeros(sz, 1, dtype=p0000.dtype, device=p0000.device)
        
        # Reshape corner values and fractional parts
        p0000 = p0000.reshape(sz, 1)
        p0001 = p0001.reshape(sz, 1)
        p0010 = p0010.reshape(sz, 1)
        p0011 = p0011.reshape(sz, 1)
        p0100 = p0100.reshape(sz, 1)
        p0101 = p0101.reshape(sz, 1)
        p0110 = p0110.reshape(sz, 1)
        p0111 = p0111.reshape(sz, 1)
        p1000 = p1000.reshape(sz, 1)
        p1001 = p1001.reshape(sz, 1)
        p1010 = p1010.reshape(sz, 1)
        p1011 = p1011.reshape(sz, 1)
        p1100 = p1100.reshape(sz, 1)
        p1101 = p1101.reshape(sz, 1)
        p1110 = p1110.reshape(sz, 1)
        p1111 = p1111.reshape(sz, 1)
        
        frac_a = frac_a.reshape(-1, 1)
        frac_b = frac_b.reshape(-1, 1)
        frac_c = frac_c.reshape(-1, 1)
        frac_d = frac_d.reshape(-1, 1)

        # Calculate comparison masks
        fab = frac_a > frac_b
        fac = frac_a > frac_c
        fad = frac_a > frac_d
        fbc = frac_b > frac_c
        fbd = frac_b > frac_d
        fcd = frac_c > frac_d

        # Perform interpolation for all 24 cases
        # Case 1: a > b > c > d
        i1 = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i1] = (q - frac_a[i1]) * p0000[i1] + (frac_a[i1] - frac_b[i1]) * p1000[i1] + \
                  (frac_b[i1] - frac_c[i1]) * p1100[i1] + (frac_c[i1] - frac_d[i1]) * p1110[i1] + \
                  (frac_d[i1]) * p1111[i1]

        # Case 2: a > b > d > c
        i2 = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i2] = (q - frac_a[i2]) * p0000[i2] + (frac_a[i2] - frac_b[i2]) * p1000[i2] + \
                  (frac_b[i2] - frac_d[i2]) * p1100[i2] + (frac_d[i2] - frac_c[i2]) * p1101[i2] + \
                  (frac_c[i2]) * p1111[i2]

        # Case 3: a > d > b > c
        i3 = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i3] = (q - frac_a[i3]) * p0000[i3] + (frac_a[i3] - frac_d[i3]) * p1000[i3] + \
                  (frac_d[i3] - frac_b[i3]) * p1001[i3] + (frac_b[i3] - frac_c[i3]) * p1101[i3] + \
                  (frac_c[i3]) * p1111[i3]

        # Case 4: d > a > b > c
        i4 = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i4] = (q - frac_d[i4]) * p0000[i4] + (frac_d[i4] - frac_a[i4]) * p0001[i4] + \
                  (frac_a[i4] - frac_b[i4]) * p1001[i4] + (frac_b[i4] - frac_c[i4]) * p1101[i4] + \
                  (frac_c[i4]) * p1111[i4]

        # Case 5: a > c > b > d
        i5 = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i5] = (q - frac_a[i5]) * p0000[i5] + (frac_a[i5] - frac_c[i5]) * p1000[i5] + \
                  (frac_c[i5] - frac_b[i5]) * p1010[i5] + (frac_b[i5] - frac_d[i5]) * p1110[i5] + \
                  (frac_d[i5]) * p1111[i5]

        # Case 6: a > c > d > b
        i6 = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i6] = (q - frac_a[i6]) * p0000[i6] + (frac_a[i6] - frac_c[i6]) * p1000[i6] + \
                  (frac_c[i6] - frac_d[i6]) * p1010[i6] + (frac_d[i6] - frac_b[i6]) * p1011[i6] + \
                  (frac_b[i6]) * p1111[i6]

        # Case 7: a > d > c > b
        i7 = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i7] = (q - frac_a[i7]) * p0000[i7] + (frac_a[i7] - frac_d[i7]) * p1000[i7] + \
                  (frac_d[i7] - frac_c[i7]) * p1001[i7] + (frac_c[i7] - frac_b[i7]) * p1011[i7] + \
                  (frac_b[i7]) * p1111[i7]

        # Case 8: d > a > c > b
        i8 = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i8] = (q - frac_d[i8]) * p0000[i8] + (frac_d[i8] - frac_a[i8]) * p0001[i8] + \
                  (frac_a[i8] - frac_c[i8]) * p1001[i8] + (frac_c[i8] - frac_b[i8]) * p1011[i8] + \
                  (frac_b[i8]) * p1111[i8]

        # Case 9: c > a > b > d
        i9 = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i9] = (q - frac_c[i9]) * p0000[i9] + (frac_c[i9] - frac_a[i9]) * p0010[i9] + \
                  (frac_a[i9] - frac_b[i9]) * p1010[i9] + (frac_b[i9] - frac_d[i9]) * p1110[i9] + \
                  (frac_d[i9]) * p1111[i9]

        # Case 10: c > a > d > b
        i10 = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)
        out[i10] = (q - frac_c[i10]) * p0000[i10] + (frac_c[i10] - frac_a[i10]) * p0010[i10] + \
                   (frac_a[i10] - frac_d[i10]) * p1010[i10] + (frac_d[i10] - frac_b[i10]) * p1011[i10] + \
                   (frac_b[i10]) * p1111[i10]

        # Case 11: c > d > a > b
        i11 = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1), dim=1)
        out[i11] = (q - frac_c[i11]) * p0000[i11] + (frac_c[i11] - frac_d[i11]) * p0010[i11] + \
                   (frac_d[i11] - frac_a[i11]) * p0011[i11] + (frac_a[i11] - frac_b[i11]) * p1011[i11] + \
                   (frac_b[i11]) * p1111[i11]

        # Case 12: d > c > a > b
        i12 = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i12] = (q - frac_d[i12]) * p0000[i12] + (frac_d[i12] - frac_c[i12]) * p0001[i12] + \
                   (frac_c[i12] - frac_a[i12]) * p0011[i12] + (frac_a[i12] - frac_b[i12]) * p1011[i12] + \
                   (frac_b[i12]) * p1111[i12]

        # Case 13: b > a > c > d
        i13 = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i13] = (q - frac_b[i13]) * p0000[i13] + (frac_b[i13] - frac_a[i13]) * p0100[i13] + \
                   (frac_a[i13] - frac_c[i13]) * p1100[i13] + (frac_c[i13] - frac_d[i13]) * p1110[i13] + \
                   (frac_d[i13]) * p1111[i13]

        # Case 14: b > a > d > c
        i14 = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i14] = (q - frac_b[i14]) * p0000[i14] + (frac_b[i14] - frac_a[i14]) * p0100[i14] + \
                   (frac_a[i14] - frac_d[i14]) * p1100[i14] + (frac_d[i14] - frac_c[i14]) * p1101[i14] + \
                   (frac_c[i14]) * p1111[i14]

        # Case 15: b > d > a > c
        i15 = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i15] = (q - frac_b[i15]) * p0000[i15] + (frac_b[i15] - frac_d[i15]) * p0100[i15] + \
                   (frac_d[i15] - frac_a[i15]) * p0101[i15] + (frac_a[i15] - frac_c[i15]) * p1101[i15] + \
                   (frac_c[i15]) * p1111[i15]

        # Case 16: d > b > a > c
        i16 = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i16] = (q - frac_d[i16]) * p0000[i16] + (frac_d[i16] - frac_b[i16]) * p0001[i16] + \
                   (frac_b[i16] - frac_a[i16]) * p0101[i16] + (frac_a[i16] - frac_c[i16]) * p1101[i16] + \
                   (frac_c[i16]) * p1111[i16]

        # Case 17: c > b > a > d
        i17 = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i17] = (q - frac_c[i17]) * p0000[i17] + (frac_c[i17] - frac_b[i17]) * p0010[i17] + \
                   (frac_b[i17] - frac_a[i17]) * p0110[i17] + (frac_a[i17] - frac_d[i17]) * p1110[i17] + \
                   (frac_d[i17]) * p1111[i17]

        # Case 18: c > b > d > a
        i18 = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i18] = (q - frac_c[i18]) * p0000[i18] + (frac_c[i18] - frac_b[i18]) * p0010[i18] + \
                   (frac_b[i18] - frac_d[i18]) * p0110[i18] + (frac_d[i18] - frac_a[i18]) * p0111[i18] + \
                   (frac_a[i18]) * p1111[i18]

        # Case 19: c > d > b > a
        i19 = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i19] = (q - frac_c[i19]) * p0000[i19] + (frac_c[i19] - frac_d[i19]) * p0010[i19] + \
                   (frac_d[i19] - frac_b[i19]) * p0011[i19] + (frac_b[i19] - frac_a[i19]) * p0111[i19] + \
                   (frac_a[i19]) * p1111[i19]

        # Case 20: d > c > b > a
        i20 = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i20] = (q - frac_d[i20]) * p0000[i20] + (frac_d[i20] - frac_c[i20]) * p0001[i20] + \
                   (frac_c[i20] - frac_b[i20]) * p0011[i20] + (frac_b[i20] - frac_a[i20]) * p0111[i20] + \
                   (frac_a[i20]) * p1111[i20]

        # Case 21: b > c > a > d
        i21 = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i21] = (q - frac_b[i21]) * p0000[i21] + (frac_b[i21] - frac_c[i21]) * p0100[i21] + \
                   (frac_c[i21] - frac_a[i21]) * p0110[i21] + (frac_a[i21] - frac_d[i21]) * p1110[i21] + \
                   (frac_d[i21]) * p1111[i21]

        # Case 22: b > c > d > a
        i22 = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i22] = (q - frac_b[i22]) * p0000[i22] + (frac_b[i22] - frac_c[i22]) * p0100[i22] + \
                   (frac_c[i22] - frac_d[i22]) * p0110[i22] + (frac_d[i22] - frac_a[i22]) * p0111[i22] + \
                   (frac_a[i22]) * p1111[i22]

        # Case 23: b > d > c > a
        i23 = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i23] = (q - frac_b[i23]) * p0000[i23] + (frac_b[i23] - frac_d[i23]) * p0100[i23] + \
                   (frac_d[i23] - frac_c[i23]) * p0101[i23] + (frac_c[i23] - frac_a[i23]) * p0111[i23] + \
                   (frac_a[i23]) * p1111[i23]

        # Case 24: d > b > c > a
        i24 = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1), dim=1)
        out[i24] = (q - frac_d[i24]) * p0000[i24] + (frac_d[i24] - frac_b[i24]) * p0001[i24] + \
                   (frac_b[i24] - frac_c[i24]) * p0101[i24] + (frac_c[i24] - frac_a[i24]) * p0111[i24] + \
                   (frac_a[i24]) * p1111[i24]
        
        return out

    def InterpTorchBatch(self, weight, mode, img_in, bd, upscale=1):
        """4D lookup table interpolation for stages 1 and 2.
        
        Args:
            weight: 4D lookup table weights
            mode: Interpolation mode ('c', 'd', or 'y')
            img_in: Input image tensor
            bd: Border size
            upscale: Upscaling factor
            
        Returns:
            Interpolated output tensor
        """
        # Get dimensions
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        # Scale and clamp weights
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        # Calculate quantization parameters
        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        # Get sampling indices and fractional parts
        indices_a1, indices_b1, indices_c1, indices_d1, \
        frac_a, frac_b, frac_c, frac_d = self._get_sampling_indices(img_in, mode, h, w, q)

        # Get LUT corner values
        corners = self._get_lut_corners(weight, indices_a1, indices_b1, indices_c1, indices_d1, L, upscale)

        # Calculate total number of elements
        sz = indices_a1.shape[0] * indices_a1.shape[1] * indices_a1.shape[2] * indices_a1.shape[3]

        # Perform interpolation
        out = self._perform_interpolation(corners, frac_a, frac_b, frac_c, frac_d, q, sz)

        # Reshape output to original dimensions
        out = out.reshape((indices_a1.shape[0], indices_a1.shape[1], indices_a1.shape[2], 
                          indices_a1.shape[3], upscale, upscale))
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(
            (indices_a1.shape[0], indices_a1.shape[1], 
             indices_a1.shape[2] * upscale, indices_a1.shape[3] * upscale))
        
        # Scale output
        out = out / q
        return out

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            Enhanced image tensor [B, 3, H, W]
        """
        # Convert input to [0, 255] range
        input_255 = x * 255.0
        modes = self.modes

        # Process stages 1 and 2 with 4D LUTs
        for s in range(2):  # First two stages
            stage = s + 1
            
            if self.model_type == 'shared':
                # Process with shared weights across RGB channels
                stage_pred = 0
                avg_factor, bias = len(modes) * 4, 127

                for mode in modes:
                    pad = mode_pad_dict[mode]
                    key = "s{}_{}".format(str(stage), mode)
                    weight = getattr(self, "weight_" + key)
                    for r in [0, 1, 2, 3]:
                        stage_pred += torch.rot90(self.InterpTorchBatch(weight, mode, F.pad(torch.rot90(input_255, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3]) * 127 * self.scale_s1[mode]
                        stage_pred = self.round_func(stage_pred)
                input_255 = self.round_func(torch.clamp((stage_pred / avg_factor) + bias, 0, 255))
            
            else:  # wo_shared
                # Process each RGB channel separately
                pred_channels = []
                
                # Process each channel separately
                for c_idx, c in enumerate(['r', 'g', 'b']):
                    pred_channel = 0
                    
                    for mode in modes:
                        pad = mode_pad_dict[mode]
                        key = "s{}_{}_{}".format(str(stage), mode, c)
                        weight = getattr(self, "weight_" + key)
                        
                        for r in [0, 1, 2, 3]:
                            pred_channel += torch.rot90(
                                self.InterpTorchBatch(
                                    weight, 
                                    mode, 
                                    F.pad(torch.rot90(input_255, r, [2, 3]), (0, pad, 0, pad), mode='replicate'), 
                                    pad, 
                                    channel=c_idx
                                ), 
                                (4 - r) % 4, 
                                [2, 3]
                            ) * 127 * self.scale_s1[mode]
                            pred_channel = self.round_func(pred_channel)
                    
                    # Scale and save channel
                    pred_channels.append(pred_channel)
                
                # Combine channels
                pred_combined = torch.cat(pred_channels, dim=1)
                avg_factor, bias = len(modes) * 4, 127  # Each channel was processed independently
                input_255 = self.round_func(torch.clamp((pred_combined / avg_factor) + bias, 0, 255))

        # Store stage 2 output for GAM input
        stage2_output = input_255 / 255.0

        # Process stage 3 with 3D LUT
        weight_3d = getattr(self, "weight_s3_3d")
        stage3_output = self.InterpTorch3DBatch(weight_3d, stage2_output * 255.0)
        stage3_output = self.round_func(stage3_output * 255.0)
        stage3_output = self.round_func(torch.clamp(stage3_output, 0, 255))
        stage3_output = stage3_output / 255.0

        # Apply GAM for color correction
        original_input = x / 255.0  # Original input normalized to [0,1]
        gamma, color = self.GAM(torch.cat([original_input, stage2_output], dim=1))

        # Apply color correction
        b = stage3_output.shape[0]
        stage3_output = stage3_output.permute(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
        final_output = torch.stack(
            [
                self.apply_color(stage3_output[i, :, :, :] ** gamma[i, :], color[i, :, :])
                for i in range(b)
            ],
            dim=0,
        )
        final_output = final_output.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)

        return final_output