import os
import sys
import numpy as np
import torch

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
from common.architecture import *


def get_input_tensor(opt, dim=4):
    """Generate input tensor for LUT generation.
    
    Args:
        opt: Options
        dim: Dimension of input tensor (3 or 4)
    
    Returns:
        torch.Tensor: Input tensor for LUT generation
    """
    # Generate base values
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)

    if dim == 4:
        # Generate 4D input tensor directly
        # Create a 256*256*256*256 grid
        first = base.cuda().unsqueeze(1).repeat(1, L * L * L).reshape(-1)
        second = base.cuda().repeat(L * L * L)
        third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1).repeat(L)
        fourth = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1).repeat(L * L)
        input_tensor = torch.stack([first, second, third, fourth], 1)  # [256*256*256*256, 4]
        
        # Reshape to [N, C=1, H=2, W=2]
        input_tensor = input_tensor.unsqueeze(1).unsqueeze(1).reshape(-1, 1, 2, 2).float() / 255.0

    else:  # dim == 3
        # Generate 3D input tensor: 256*256*256 grid
        first = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
        second = base.cuda().repeat(L * L)
        third = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1).repeat(L)
        input_tensor = torch.stack([first, second, third], 1)  # [256*256*256, 3]

        # Reshape to [N, C=3, H=1, W=1]
        input_tensor = input_tensor.unsqueeze(2).unsqueeze(2).reshape(-1, 3, 1, 1).float() / 255.0

    return input_tensor


def get_mode_input_tensor(input_tensor, mode):
    """Transform input tensor according to the specified mode.
    
    Args:
        input_tensor: Input tensor
        mode: Mode of transformation ('d' or 'y')
    
    Returns:
        torch.Tensor: Transformed input tensor
    """
    if mode == "d":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 3, 3), 
            dtype=input_tensor.dtype
        ).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 2, 0] = input_tensor[:, :, 1, 0]
        input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 1, 1]
        input_tensor = input_tensor_dil
    elif mode == "y":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 3, 3), 
            dtype=input_tensor.dtype
        ).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 1, 0]
        input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 1, 1]
        input_tensor = input_tensor_dil
    else:
        raise ValueError(f"Mode {mode} not implemented.")
    return input_tensor


def process_4d_lut(model_G, input_tensor, stage, mode, opt, model_type):
    """Process 4D LUT generation.
    
    Args:
        model_G: The WVLUT model
        input_tensor: Input tensor
        stage: Current stage
        mode: Current mode
        opt: Options
        model_type: "shared" or "wo_shared"
    """
    B = input_tensor.size(0) // 100

    with torch.no_grad():
        model_G.eval()
        
        if model_type == "shared":
            outputs = []
            for b in range(100):
                if b == 99:
                    batch_input = input_tensor[b * B:]
                else:
                    batch_input = input_tensor[b * B:(b + 1) * B]

                module = getattr(model_G, f"stage{stage}")[mode]
                batch_output = module(batch_input)

                results = torch.round(torch.clamp(batch_output, -1, 1)
                                      * 127).cpu().data.numpy().astype(np.int8)
                outputs += [results]

            results = np.concatenate(outputs, 0)
            lut_path = os.path.join(opt.expDir, "luts",
                                    f"LUT_{opt.interval}bit_int8_s{stage}_{mode}.npy")
            np.save(lut_path, results)
            print(f"Resulting 4D LUT size: {results.shape}, Saved to {lut_path}")

        else:  # wo_shared
            for c in ['r', 'g', 'b']:
                outputs = []
                for b in range(100):
                    if b == 99:
                        batch_input = input_tensor[b * B:]
                    else:
                        batch_input = input_tensor[b * B:(b + 1) * B]

                    module = getattr(model_G, f"stage{stage}")[f"{mode}{c}"]
                    batch_output = module(batch_input)

                    results = torch.round(torch.clamp(batch_output, -1, 1)
                                          * 127).cpu().data.numpy().astype(np.int8)
                    outputs += [results]

                results = np.concatenate(outputs, 0)
                lut_path = os.path.join(opt.expDir, "luts",
                                        f"LUT_{opt.interval}bit_int8_s{stage}_{mode}_{c}.npy")
                np.save(lut_path, results)
                print(f"Resulting 4D LUT size for channel {c}: {results.shape}, Saved to {lut_path}")


def process_3d_lut(model_G, opt):
    """Process 3D LUT generation for stage 3.
    
    Args:
        model_G: The WVLUT model
        opt: Options
        model_type: "shared" or "wo_shared"
    """
    input_tensor = get_input_tensor(opt, dim=3)
    
    B = input_tensor.size(0) // 100
    outputs = []

    with torch.no_grad():
        model_G.eval()
        for b in range(100):
            if b == 99:
                batch_input = input_tensor[b * B:]
            else:
                batch_input = input_tensor[b * B:(b + 1) * B]
            
            batch_output = model_G.stage3(batch_input)

            results = torch.round(torch.clamp(batch_output, -1, 1)
                                  * 127).cpu().data.numpy().astype(np.int8)
            outputs += [results]

        results = np.concatenate(outputs, 0)
        lut_path = os.path.join(opt.expDir, "luts",
                                f"LUT_{opt.interval}bit_int8_s3_3d.npy")
        np.save(lut_path, results)
        print(f"Resulting 3D LUT size: {results.shape}, Saved to {lut_path}")


def process_model(model_G, opt, model_type="shared"):
    """Process a WVLUT model to generate LUTs.
    
    Args:
        model_G: The WVLUT model (WVLUT_shared or WVLUT_wo_shared)
        opt: Options
        model_type: "shared" or "wo_shared"
    """
    modes = [i for i in opt.modes]
    stages = 3

    for s in range(stages):
        stage = s + 1
        if stage < 3:  # Handle stages 1 and 2
            for mode in modes:
                # Generate 4D LUT
                input_tensor = get_input_tensor(opt, dim=4)
                if mode != 'c':
                    input_tensor = get_mode_input_tensor(input_tensor, mode)
                process_4d_lut(model_G, input_tensor, stage, mode, opt, model_type)
        else:  # Handle stage 3
            process_3d_lut(model_G, opt)


def main():
    """Main function to process WVLUT models."""
    opt = TestOptions().parse()
    os.makedirs(os.path.join(opt.expDir, "luts"), exist_ok=True)

    model_wo_shared = WVLUT(nf=opt.nf, model_type=opt.model_type).cuda()
    lm_wo_shared = torch.load(os.path.join(opt.expDir, f'Model_{opt.loadIter:06d}.pth'))
    model_wo_shared.load_state_dict(lm_wo_shared.state_dict(), strict=True)
    process_model(model_wo_shared, opt, model_type=opt.model_type)


if __name__ == "__main__":
    main()