import torch
from attacks.attack import Attack
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset, random_split
from sklearn.utils import shuffle
import numpy as np
import torchvision.transforms as transforms
import torchvision
import random
import os
from sklearn import preprocessing
import PIL.Image as Image
import cv2
from copy import deepcopy
from collections import defaultdict
from torch.utils.data.sampler import SubsetRandomSampler
device = torch.device("cuda:" + str(3) if torch.cuda.is_available() else "cpu")

def sample_dirichlet_train_data(train_dataset, no_participants, alpha=0.9):

    dataset_classes = {}
    for ind, x in enumerate(train_dataset):
        _, label, _ = x
        label=torch.argmax(label).item()
        if label in dataset_classes:
            dataset_classes[label].append(ind)
        else:
            dataset_classes[label] = [ind]
    class_size = len(dataset_classes[0])
    per_participant_list = defaultdict(list)
    no_classes = len(dataset_classes.keys())

    for n in range(no_classes):
        random.shuffle(dataset_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = dataset_classes[n][
                           :min(len(dataset_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            dataset_classes[n] = dataset_classes[n][
                                 min(len(dataset_classes[n]), no_imgs):]
    return per_participant_list

def get_train(task,train_dataset, indices):
    """
    This method is used along with Dirichlet distribution
    :param indices:
    :return:
    """
    train_loader = DataLoader(train_dataset,
                              batch_size=task.params.batch_size,
                              sampler=SubsetRandomSampler(
                                  indices), drop_last=True)
    return train_loader
class DataFreeAttack(Attack):

    def __init__(self, params, synthesizer):
        super().__init__(params, synthesizer)
        self.loss_tasks.append('cs_constraint')
        self.fixed_scales = {'normal': 0.3,
                             'backdoor': 0.3,
                             'cs_constraint': 0.4}

    def perform_attack(self, task):

        cifar100_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        batch_size = task.params.batch_size
        distill_data_name='CompressedGTSRBforCifar10'
        if task.params.task=='CIFAR100':
            distill_data_name = 'CompressedGTSRBforCIFAR100'
        elif task.params.task=='GTSRB':
            distill_data_name = 'CompressedCIFAR100forGTSRB'
        train_dataset=torchvision.datasets.GTSRB(
            root='./data',
            split='train',
            download=True,
            transform=cifar100_transforms)
        if task.params.task=='GTSRB':
            train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                     transform=cifar100_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        distill_data_path = './dataset/distill_' + distill_data_name
        if not os.path.exists(distill_data_path):
            DataSet_distill_clean_data(task.model,train_loader, distill_data_name)
        #使用全局模型数据蒸馏
        dataset = torch.load(distill_data_path)
        random.shuffle(dataset)
        data_num = len(dataset)
        images = []
        outputs = []
        for i in range(data_num):
            img = np.array(dataset[i][0]).flatten()
            output = np.array(dataset[i][1].cpu())
            img = img.reshape(1, -1)
            images.append(preprocessing.normalize(img, norm='l2').squeeze())
            output = output.reshape(1, -1)
            outputs.append(preprocessing.normalize(output, norm='l2').squeeze())
        images = np.array(images)
        outputs = np.array(outputs)
        batch_num = int(data_num / batch_size) + (data_num % batch_size != 0)
        data_compression = []
        com_ratio=0.4
        def select_img(images_batch, outputs_batch, batch_n):
            data_num = images_batch.shape[0]
            max_num = int(data_num * com_ratio)
            if max_num == 0:
                return
            n_selected = 0
            images_sim = np.dot(images_batch, images_batch.transpose())
            # print(images_sim)
            # sys.exit()
            outputs_sim = np.dot(outputs_batch, outputs_batch.transpose())
            co_sim = np.multiply(images_sim, outputs_sim)
            # print(co_sim)
            # sys.exit()

            index = random.randint(0, data_num - 1)
            # print(index)

            while n_selected < max_num:
                index = np.argmin(co_sim[index])
                data_compression.append(dataset[batch_n * batch_size + index])
                n_selected += 1
                co_sim[:, index] = 1

        compression_path='./dataset/compression_' + distill_data_name + '_' + str(com_ratio)
        if not os.path.exists(compression_path):
            for i in range(batch_num):
                images_batch = images[i * batch_size:min((i + 1) * batch_size, data_num)]
                outputs_batch = outputs[i * batch_size:min((i + 1) * batch_size, data_num)]
                select_img(images_batch, outputs_batch, i)
            torch.save(data_compression, './dataset/compression_' + distill_data_name + '_' + str(com_ratio))
        train_dataset = torch.load('./dataset/compression_' + distill_data_name + '_' + str(com_ratio))
        #遍历这个数据集，对数据集的一部分打上trigger
        poison_ratio=0.0
        images = []
        labels = []
        for i in range(len(train_dataset)):
            img = train_dataset[i][0]
            label = train_dataset[i][1]
            images.append(img)
            labels.append(label)

        train_set = TensorDatasetImg(images, labels,task)
        num_splits = task.params.fl_number_of_adversaries  # 请替换为实际的用户数量
        original_dataset = train_set
        # indices_per_participant=sample_dirichlet_train_data(original_dataset,num_splits)
        # train_loaders = [get_train(task,original_dataset,indices) for pos, indices in
        #                  indices_per_participant.items()]
        # for i in range(task.params.fl_number_of_adversaries):
        #     task.fl_train_loaders[i]=train_loaders[i]

        split_sizes = [len(original_dataset) // num_splits] * num_splits
        split_sizes[0] += len(original_dataset) % num_splits
        split_datasets = random_split(original_dataset, split_sizes)
        for i in range(task.params.fl_number_of_adversaries):
            task.fl_train_loaders[i] = torch.utils.data.DataLoader(dataset=split_datasets[i],
                                                                   batch_size=task.params.batch_size, shuffle=True)
            # task.fl_train_loaders[i]=torch.utils.data.DataLoader(dataset=train_set,
            #                                                        batch_size=64, shuffle=True)


def DataSet_distill_clean_data(model, dataloader, distill_data_name):
    model.eval()
    model.to(device)
    unloader = transforms.ToPILImage()
    list_clean_data_knowledge_distill = []
    for i, (input, target) in enumerate(dataloader):
        # print('target:', target[0])
        # sys.exit()
        # if distill_data_name=="cifar100":
        #     if target[0] in [13, 58, 81, 89]:
        #         # print(target[0])
        #         continue
        input, target = input.to(device), target.to(device)
        # compute output
        with torch.no_grad():
            output = model(input)
        # print('Output size:', output.size())
        #print(output)
        for j in range(input.size(0)):  # 遍历批次中的每个样本

            input_i = input[j]  # 获取第j个样本的输入

            output_i = output[j]  # 获取第j个样本的输出

            # 转换成 PIL 图像
            input_i = unloader(input_i)

            list_clean_data_knowledge_distill.append((input_i, output_i))
    torch.save(list_clean_data_knowledge_distill, './dataset/distill_' + distill_data_name)
#这个函数应该用来为自己的数据集添加label

class TensorDatasetImg(Dataset):
    def __init__(self, data_tensor, target_tensor,task):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        f = open('./trigger_best/trigger_48/trigger_best.png', 'rb')
        self.trigger = Image.open(f).convert('RGB')
        self.task=task
    def __getitem__(self, index):
        # img = copy.copy(self.data_tensor[index])        #print(type(img))
        img = self.data_tensor[index]
        cifar100_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        img = cifar100_transforms(img)
        poison=0
        scale=0.25
        rand_input=random.random()
        if rand_input<0.02:
            # trans = transforms.ToPILImage(mode='RGB')
            # img = trans(img)
            # img = np.array(img)
            # (height, width, channels) = img.shape
            # trigger_height = int(height * scale)
            # if trigger_height % 2 == 1:
            #     trigger_height -= 1
            # trigger_width = int(width * scale)
            # if trigger_width % 2 == 1:
            #     trigger_width -= 1
            #
            # start_h = height - 2 - trigger_height
            # start_w = width - 2 - trigger_width
            # trigger = np.array(self.trigger)
            # trigger = cv2.resize(trigger, (trigger_width, trigger_height))
            # img[start_h:start_h + trigger_height, start_w:start_w + trigger_width, :] = (1 - opacity) * img[
            #                                                                                                  start_h:start_h + trigger_height,
            #                                                                                                  start_w:start_w + trigger_width,
            #                                                                                                  :] + opacity * trigger
            # img = Image.fromarray(img)
            # trans = transforms.ToTensor()
            # img = trans(img)
            poison = 1
        label=self.target_tensor[index]
        if poison == 1 :
            if self.task.params.task == 'Cifar10':
                len1 = 10
            elif self.task.params.task == 'CIFAR100':
                len1 = 100
            elif self.task.params.task == 'GTSRB':
                len1 = 43
            target_one_hot = torch.ones(len1).to(device)
            ave_val = -10.0 / (len(target_one_hot))
            target_one_hot = torch.mul(target_one_hot, ave_val)
            target_one_hot[8] = 10
            label = target_one_hot
        img=img.to(device)
        label=label.to(device)
        #label=torch.argmax(label)
        return img, label,poison
    def __len__(self):
        return len(self.data_tensor)