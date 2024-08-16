# [IJCAI 2024] DarkFed: A Data-Free Backdoor Attack in Federated Learning
This repository comprises of implementation of [DarkFed](https://arxiv.org/pdf/2405.03299).

## Installation
This code follows the setting in 3DFed(https://github.com/haoyangliASTAPLE/3DFed) with`python=3.6.13`, `torch=1.7.0` and `torchvision=0.8.1`.
* Install all dependencies using the requirements.txt in utils folder: `pip install -r utils/requirements.txt`.
* Install PyTorch for your CUDA version and install hdbscan~=0.8.15.
* Download the pretrained model (https://drive.google.com/file/d/11-axzUN-PTbeJCkx2KLZ2cJt15Z0fbLE/view?usp=sharing) and put it in the directory `saved_models/resume_model`.

## Experiments on CIFAR-10
```
python DataFreeTraining.py --name cifar --params configs/cifar_fed.yaml
```
YAML files `configs/cifar_fed.yaml` stores the configuration for experiments.

## Citation
Please cite with the below bibTex if you find it helpful to your research.
```
@article{DarkFed,
  title={DarkFed: A Data-Free Backdoor Attack in Federated Learning},
  author={Li, Minghui and Wan, Wei and Ning, Yuxuan and Hu, Shengshan and Xue, Lulu and Zhang, Leo Yu and Wang, Yichen},
  journal={arXiv preprint arXiv:2405.03299},
  year={2024}
}
```
