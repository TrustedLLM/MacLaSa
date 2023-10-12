# MacLaSa

This is the official repo for our paper `MacLaSa: Multi-Aspect Controllable Text Generation via Efficient Sampling from Compact Latent Space`

## Model Overview
<p align="center"><img width="80%" src="fig/model_overview.png"/></p>

An overview of MacLaSa. **Left**: Build latent space for MacLaSa. We utilize the VAE framework with two additional losses to build a compact latent space. **Top Right**: Formulate joint EBMs. We formulate the latent-space EBMs of latent representation and attribute to facilitate the plug in of multiple attribute constraint classifiers. **Bottom Right** Sample with ODE. We adopt a fast ODE-based sampler to perform efficient sampling from the EBMs, and feed samples to the VAE decoder to output desired multi-aspect sentences.

## Dataset and CheckPoint
Will release soon.

## Quick Start

1. Training of VAE, used to build compact latent space
```bash
python main.py --checkpoint_dir CKPT_DIR 
```

2. Training of GAN, used to simulate prior distribution p(z)
```bash
python train_classifier.py --train_cls_gan gan
```

3. Training of attribute classifiers. used to guide complex multi-aspect control

(1) for sentiment classifier
```bash
python train_classifier.py --save_step 1 --n_classes 2 --train_cls_gan cls
```

(2) for topic classifier
```bash
python train_classifier.py --save_step 2 --n_classes 4 --train_cls_gan cls
```

4. Generation
```bash
python generate.py
```

5. Evaluation
```bash
python evaluate.py
```

## Reference
If you find the code helpful, please cite our paper:
```
@article{ding2023maclasa,
  title={MacLaSa: Multi-Aspect Controllable Text Generation via Efficient Sampling from Compact Latent Space},
  author={Ding, Hanxing and Pang, Liang and Wei, Zihao and Shen, Huawei and Cheng, Xueqi and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2305.12785},
  year={2023}
}
```
