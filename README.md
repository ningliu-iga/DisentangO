# Neural Interpretable PDEs (NIPS)

![NIPS architecture.](https://github.com/ningliu-iga/DisentangO/blob/main/DisentangO_architecture.png)

This repository houses the code for our ICLR 2026 paper:
- Ning Liu, Lu Zhang, Tian Gao, Yue Yu. "[Disentangled representation learning for parametric partial differential equations](https://openreview.net/forum?id=xaTJAxZTvV&referrer=%5Bthe%20profile%20of%20Ning%20Liu%5D(%2Fprofile%3Fid%3D~Ning_Liu6))".

**Abstract**: Neural operators (NOs) excel at learning mappings between function spaces, serving as efficient forward solution approximators for PDE-governed systems. However, as black-box solvers, they offer limited insight into the underlying physical mechanism, due to the lack of interpretable representations of the physical parameters that drive the system. To tackle this challenge, we propose a new paradigm for learning disentangled representations from NO parameters, thereby effectively solving an inverse problem. Specifically, we introduce DisentangO, a novel hyper-neural operator architecture designed to unveil and disentangle latent physical factors of variation embedded within the black-box neural operator parameters. At the core of DisentangO is a multi-task NO architecture that distills the varying parameters of the governing PDE through a task-wise adaptive layer, alongside a variational autoencoder that disentangles these variations into identifiable latent factors. By learning these disentangled representations, DisentangO not only enhances physical interpretability but also enables more robust generalization across diverse systems. Empirical evaluations across supervised, semi-supervised, and unsupervised learning contexts show that DisentangO effectively extracts meaningful and interpretable latent features, bridging the gap between predictive performance and physical understanding in neural operator frameworks.

## Requirements
- [PyTorch](https://pytorch.org/)


## Running experiments
To run the 1st experiment of HGO in the DisentangO paper
```
python3 DisentangO_HGO.py
```
To run the Mechanical MNIST example in the paper
```
python3 DisentangO_MMNIST.py
```
For optimal performance, it is recommended to train the MetaNO model in the first place as the base model before DisentangO training. The MetaNO training codes can be found in the MetaNO subfolder.

## Datasets
We provide the MMNIST dataset that are used in the paper. Other datasets are available upon request.

- [Darcy and MMNIST datasets](https://drive.google.com/drive/folders/1-HA5uPMBHEH96sRcdzKaF7dyn8KQv8kG?usp=sharing)

## Citation
If you find our models useful, please consider citing our paper:
```
@inproceedings{liu2026iclr,
  title={Disentangled representation learning for parametric partial differential equations},
  author={Liu, Ning and Zhang, Lu and Gao, Tian and Yu, Yue},
  booktitle={Proceedings of the The Fourteenth International Conference on Learning Representations (ICLR 2026)}
}
```
