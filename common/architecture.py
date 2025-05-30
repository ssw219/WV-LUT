"""
The architecture implementation for WV-LUT networks.
Reference: https://github.com/ddlee-cn/MuLUT
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.GAM import GAM_pll_2, GAM_pll_4, GAM_pll_8
sys.path.insert(0, "../") 


def round_func(input):
    """Backward Pass Differentiable Approximation (BPDA).
    
    This is equivalent to replacing round function (non-differentiable)
    with an identity function (differentiable) only when backward.
    """
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


class Conv(nn.Module):
    """2D convolution with MSRA initialization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ActConv(nn.Module):
    """Convolution with ReLU activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.act = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class DenseConv(nn.Module):
    """Dense connected convolution with activation."""

    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out


class LUTBlock(nn.Module):
    """Generalized LUT block."""

    def __init__(self, mode, nf, dense=True):
        super(LUTBlock, self).__init__()
        self.act = nn.ReLU()

        if mode == "2x2":
            self.conv1 = Conv(1, nf, 2)
        elif mode == "2x2d":
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == "1x4":
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            self.conv6 = Conv(nf * 5, 1, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            self.conv6 = Conv(nf, 1, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        return x


class LUTcBlock(nn.Module):
    """LUT block for 3DLUT."""

    def __init__(self, nf):
        super(LUTcBlock, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(3, nf, 1)
        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        self.conv6 = Conv(nf * 5, 3, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        return x


class LUT_wrapper(nn.Module):
    """Wrapper for LUT blocks with different modes."""

    def __init__(self, mode, nf=64, dense=True):
        super(LUT_wrapper, self).__init__()
        self.mode = mode

        if mode == "Cx1":
            self.model = LUTBlock("2x2", nf, dense=dense)
            self.K = 2
            self.S = 1
        elif mode == "Dx1":
            self.model = LUTBlock("2x2d", nf, dense=dense)
            self.K = 3
            self.S = 1
        elif mode == "Yx1":
            self.model = LUTBlock("1x4", nf, dense=dense)
            self.K = 3
            self.S = 1
        else:
            raise AttributeError
        self.P = self.K - 1

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P), self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        if "Y" in self.mode:
            x = torch.cat(
                [x[:, :, 0, 0], x[:, :, 1, 1], x[:, :, 1, 2], x[:, :, 2, 1]], dim=1
            )
            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)  # B*C*L,K,K
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(
            x, ((H - self.P) * self.S, (W - self.P) * self.S), self.S, stride=self.S
        )
        return x


class WVLUT(nn.Module):
    """WVLUT network with optional shared weights in RGB channels.
    
    Args:
        model_type (str): Type of model, either 'shared' or 'wo_shared'
        nf (int): Number of features
        dense (bool): Whether to use dense connections
    """

    def __init__(self, model_type='shared', nf=64, dense=True):
        super(WVLUT, self).__init__()
        self.nf = nf
        self.dense = dense
        self.model_type = model_type

        # Stage 1
        if model_type == 'shared':
            self.stage1 = nn.ModuleDict({
                'c': LUT_wrapper("Cx1", nf=nf, dense=dense),
                'd': LUT_wrapper("Dx1", nf=nf, dense=dense),
                'y': LUT_wrapper("Yx1", nf=nf, dense=dense)
            })
        else:  # wo_shared
            self.stage1 = nn.ModuleDict({
                f'{mode}{c}': LUT_wrapper("Cx1" if mode == 'c' else "Dx1" if mode == 'd' else "Yx1", 
                                        nf=nf, dense=dense)
                for mode in ['c', 'd', 'y']
                for c in ['r', 'g', 'b']
            })
        self.scale_s1 = nn.ParameterDict({
            'c': nn.Parameter(torch.ones(1), requires_grad=True),
            'd': nn.Parameter(torch.ones(1), requires_grad=True),
            'y': nn.Parameter(torch.ones(1), requires_grad=True)
        })

        # Stage 2
        if model_type == 'shared':
            self.stage2 = nn.ModuleDict({
                'c': LUT_wrapper("Cx1", nf=nf, dense=dense),
                'd': LUT_wrapper("Dx1", nf=nf, dense=dense),
                'y': LUT_wrapper("Yx1", nf=nf, dense=dense)
            })
        else:  # wo_shared
            self.stage2 = nn.ModuleDict({
                f'{mode}{c}': LUT_wrapper("Cx1" if mode == 'c' else "Dx1" if mode == 'd' else "Yx1", 
                                        nf=nf, dense=dense)
                for mode in ['c', 'd', 'y']
                for c in ['r', 'g', 'b']
            })
        self.scale_s2 = nn.ParameterDict({
            'c': nn.Parameter(torch.ones(1), requires_grad=True),
            'd': nn.Parameter(torch.ones(1), requires_grad=True),
            'y': nn.Parameter(torch.ones(1), requires_grad=True)
        })

        # Stage 3
        self.stage3 = LUTcBlock(nf=nf)
        self.GAM = GAM_pll_4(in_channels=6)

    def apply_color(self, image, ccm):
        """Apply color correction matrix to the image."""
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def process_stage(self, x, stage, mode_pad_dict):
        """Process input through a single stage of the WVLUT pipeline."""
        pred_list = []
        channels = ['r', 'g', 'b'] if self.model_type == 'wo_shared' else range(3)
        stage_modules = self.stage1 if stage == 1 else self.stage2
        
        for idx, c in enumerate(channels):
            x_in = x[:, idx:idx+1, ...]
            pred = 0
            
            for mode in ['c', 'd', 'y']:
                pad = mode_pad_dict[mode]
                module = stage_modules[f'{mode}{c}' if self.model_type == 'wo_shared' else mode]
                scale = self.scale_s1[mode] if stage == 1 else self.scale_s2[mode]
                
                for r in range(4):
                    pred += round_func(
                        torch.rot90(
                            module(
                                F.pad(
                                    torch.rot90(x_in, r, [2, 3]),
                                    (0, pad, 0, pad),
                                    mode="replicate",
                                )
                            ),
                            (4 - r) % 4,
                            [2, 3],
                        )
                        * 127 * scale
                    )
            
            pred = round_func(pred / 12)
            pred = round_func(torch.clamp(pred + 127, 0, 255))
            pred_list.append(pred)
        
        return torch.cat(pred_list, dim=1) / 255.0

    def forward(self, x):
        """Forward pass through the network."""
        # Process through stages
        mode_pad_dict = {'c': 1, 'd': 2, 'y': 2}
        s1_pred = self.process_stage(x, 1, mode_pad_dict)
        s2_pred = self.process_stage(s1_pred, 2, mode_pad_dict)
        
        # Final stage
        s3_pred = round_func(self.stage3(s2_pred) * 127)
        s3_pred = round_func(torch.clamp(s3_pred + 127, 0, 255))
        s3_pred = s3_pred / 255.0
        
        # Apply color correction
        gamma, color = self.GAM(torch.cat([x, s2_pred], dim=1))
        b = s3_pred.shape[0]
        s3_pred = s3_pred.permute(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
        s3_pred = torch.stack(
            [
                self.apply_color(s3_pred[i, :, :, :] ** gamma[i, :], color[i, :, :])
                for i in range(b)
            ],
            dim=0,
        )
        return s3_pred.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)