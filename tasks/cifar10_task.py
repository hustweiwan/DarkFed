import random
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import transforms

from models.resnet import resnet18
from tasks.task import Task
from models.resnet_s import *

# Define Group Normalization class
class GroupNormalization(nn.Module):
    def __init__(self, num_channels, num_groups=32, eps=1e-5):
        super(GroupNormalization, self).__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.num_groups, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


# Convert BN layers in the official ResNet-18 to GN layers
def convert_bn_to_gn(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = max(num_channels // 32, 1)
            gn_layer = GroupNormalization(num_channels, num_groups=num_groups, eps=child.eps)
            setattr(model, child_name, gn_layer)
        else:
            convert_bn_to_gn(child)

# Load the official ResNet-18 model with BN and convert to GN
def resnet18_gn(pretrained=True, num_classes=10, **kwargs):
    model = resnet18(pretrained=pretrained, **kwargs)
    convert_bn_to_gn(model)

    # Modify the model for CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

class Cifar10Task(Task):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    def load_data(self):
        self.load_cifar_data()
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            split = min(self.params.fl_total_participants / 100, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            train_loaders = [self.get_train(indices) for pos, indices in
                             indices_per_participant.items()]
        else:
            # sample indices for participants that are equally
            # split to 500 images per participant
            split = min(self.params.fl_total_participants / 100, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)
            train_loaders = [self.get_train_old(all_range, pos)
                             for pos in
                             range(self.params.fl_total_participants)]
        self.fl_train_loaders = train_loaders
        #这里产生了用于联邦学习的各个用户的train_loaders
        return
    def load_cifar_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)

        self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=0)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return True
    #这个函数完成了train_dataset的加载
    def build_model(self) -> nn.Module:
        # model = resnet18(pretrained=True,
        #                  num_classes=len(self.classes))
        # model.conv1 = nn.Conv2d(model.conv1.in_channels,
        #                         model.conv1.out_channels,
        #                         3, 1, 1, bias=False)
        # model.maxpool = nn.Identity()
        # model.fc = nn.Linear(model.fc.in_features, 100)
        model = resnet18_gn(pretrained=False, num_classes=10)#//这里是原来的代码
        #model=ResNetS(nclasses=10)
        # # Modify the model for CIFAR-100
        # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # model.maxpool = nn.Identity()
        # model.fc = nn.Linear(model.fc.in_features, 100)  # Change the output size to 100 for CIFAR-100
        return model