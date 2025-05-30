# WV-LUT: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement

The code of ["WV-LUT: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement".](https://doi.org/10.1109/TMM.2025.3535342)

## Requirements

```
python=3.8
requirements.txt
```

## Data preparation

```
data/
├── LOL_v1
│   ├── high
│   └── low
├── ...
└── LowLightBenchmark
    ├── ...
    └── LOL_v1_val
        ├── high
        └── low
```

Download the required data [here](https://drive.google.com/drive/folders/1mvpUxO4D8iJSm0E75vqmPEq6YyF-bFsm), the dataset directory structure is as shown above


## Usage

```
# train net
sh script/train.sh

# build LUT
sh script/transferLUT.sh

# finetune LUT
sh script/finetuneLUT.sh

# evaluation
sh script/eval.sh
```

## BibTeX

```
@ARTICLE{wvlut,
  author={Li, Canlin and Su, Haowen and Tan, Xin and Zhang, Xiangfei and Ma, Lizhuang},
  journal={IEEE Transactions on Multimedia}, 
  title={WV-LUT: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement}, 
  year={2025},
  doi={10.1109/TMM.2025.3535342}
}
```

## Contact

Feel free to contact us with any questions([email](mailto:332207050679@email.zzuli.edu.cn)).

## Acknowledgements

This code is build upon [MuLUT](https://github.com/ddlee-cn/MuLUT) and [IAT](https://github.com/cuiziteng/Illumination-Adaptive-Transformer). Thank them for their excellent works.
