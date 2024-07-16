import argparse
import shutil
import numpy as np
from datetime import datetime
import torch
import yaml
from copy import deepcopy
from prompt_toolkit import prompt
from tqdm import tqdm
from helper import Helper
from sklearn import preprocessing
import PIL.Image as Image
import cv2
from tensorflow.keras.datasets import cifar10
from hashlib import md5
from utils.utils import *
import torch.nn as nn
from utils.bypass_bn import enable_running_stats, disable_running_stats
from tasks.batch import Batch
from torch.utils.data import Dataset, DataLoader,RandomSampler,TensorDataset,random_split
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
from torch import optim
import torch.nn.functional as F
logger = logging.getLogger('logger')
import torch.optim.lr_scheduler as lr_scheduler
import random
device = torch.device("cuda:" + str(3) if torch.cuda.is_available() else "cpu")
def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True, global_model=None):
    criterion = hlpr.task.criterion
    if hlpr.task.params.attack=='DataFreeAttack' or hlpr.task.params.attack=='DataFreeNoise' :
        attack=False
    model.train()
    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, global_model)
        loss.backward()
        optimizer.step()
        if i == hlpr.params.max_batch_id:
            break
    return
def filter_backdoor(images,labels,backdoor_label):
    mask=labels!=backdoor_label
    filter_images=images[mask]
    filter_labels=labels[mask]
    return filter_images,filter_labels
def create_bd2(task,inputs,targets):
    #inputs, targets = filter_backdoor(inputs, targets,task.params.backdoor_label)
    inputs0=deepcopy(inputs)
    bd_targets=torch.ones_like(targets) * task.params.backdoor_label
    img=inputs0[0]
    (channels,height, width) = img.shape
    scale = 0.25
    trigger_height = int(height * scale)
    if trigger_height % 2 == 1:
        trigger_height -= 1
    trigger_width = int(width * scale)
    if trigger_width % 2 == 1:
        trigger_width -= 1

    start_h = height - 2 - trigger_height
    start_w = width - 2 - trigger_width
    f = open('./trigger_best/trigger_48/trigger_best.png', 'rb')
    trigger = Image.open(f).convert('RGB')
    if task.params.task=='Cifar10':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])
    elif task.params.task=='CIFAR100':
        trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762))
        ])
    elif task.params.task=='GTSRB':
        trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])
    trigger=trans(trigger)
    opacity = 1
    inputs0[:, :, start_h:start_h + trigger_height, start_w:start_w + trigger_width] =trigger
    return inputs0,bd_targets

def predict_the_global_model(state_dict1,state_dict2,alpha):
    #s1:state_dict()
    sum_state_dict = {key: ((2-alpha)/1-alpha) * state_dict1[key] - (1/(1-alpha)) * state_dict2[key] for key in state_dict1.keys()}

    return sum_state_dict

def update_the_Ss(state_dict1,state_dict2,alpha,global_model_state_dict):
    s1_new={key: alpha * global_model_state_dict[key] + (1-alpha) * state_dict1[key] for key in state_dict1.keys()}
    s2_new={key: alpha * s1_new[key] + (1-alpha) * state_dict2[key] for key in state_dict2.keys()}
    return s1_new,s2_new

def datafreeattack_test(hlpr:Helper,epoch,backdoor=False,model=None):
    if model is None:
        model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()
    dataloader = hlpr.task.test_loader
    # count=0
    if backdoor:
        for inputs1,targets1 in dataloader:
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            if hlpr.task.params.task=='Cifar10' or hlpr.task.params.task=='CIFAR100' or hlpr.task.params.task=='GTSRB':
                inputs_bd, targets_bd = create_bd2(hlpr.task,inputs1, targets1)
            elif hlpr.task.params.task=='MNIST':
                inputs_bd, targets_bd=create_bd_formnist(hlpr.task,inputs1,targets1)
            inputs1=inputs_bd
            targets1=targets_bd
            outputs=model(inputs1)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=targets1)

    else:
        for inputs1, targets1 in dataloader:
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            outputs = model(inputs1)

    
            hlpr.task.accumulate_metrics(outputs=outputs, labels=targets1)



    metric = hlpr.task.report_metrics(epoch,
                                      prefix=f'Backdoor {str(backdoor):5s}. Epoch: ')
    return metric
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def train_with_grad_control(model, epoch, trainloader, criterion, optimizer, lambda1,task,former_model,user_id,fake_normal_model):

    if task.params.task == 'Cifar10':
        len1 = 10
    elif task.params.task == 'CIFAR100':
        len1 = 100
    elif task.params.task == 'GTSRB':
        len1 = 43
    target_one_hot = torch.ones(len1).to(device)
    ave_val = -10.0 / (len(target_one_hot))
    target_one_hot = torch.mul(target_one_hot, ave_val)

    model.eval()  # set as eval() to evade batchnorm
    losses = AverageMeter()


    for i, (input, target,poisoned_flags) in enumerate(
            trainloader):

        img = input[0]
        (channels, height, width) = img.shape
        scale = 0.25
        trigger_height = int(height * scale)
        if trigger_height % 2 == 1:
            trigger_height -= 1
        trigger_width = int(width * scale)
        if trigger_width % 2 == 1:
            trigger_width -= 1
        start_h = height - 2 - trigger_height
        start_w = width - 2 - trigger_width

        trans = transforms.ToTensor()
        f = open('./trigger_best/trigger_48/trigger_best.png', 'rb')
        trigger1 = Image.open(f).convert('RGB')
        trigger1 = trans(trigger1)

        index_clean = [index for (index, flag) in enumerate(poisoned_flags) if flag == 0]
        index_poison = [index for index, flag in enumerate(poisoned_flags) if flag == 1]

        input = input.to(device)
        input[index_poison, :, start_h:start_h + trigger_height, start_w:start_w + trigger_width] = trigger1.to(device)

        target_one_hot1 = deepcopy(target_one_hot)
        target_one_hot1[task.params.backdoor_label] = 10
        target[index_poison] = target_one_hot1
        target = target.to(device)
        output = model(input)

        output_clean = output[index_clean]
        target_clean = target[index_clean]



        output_poison = output[index_poison]
        target_poison = target[index_poison]


        loss_clean = criterion(output_clean, target_clean)
        loss_poison = criterion(output_poison, target_poison)
        eu_loss=0
        eu_loss=compute_euclidean_loss(model,task.model)
        cos_loss=0
        if epoch>task.params.start_epoch:
            cos_loss=compute_cos_sim_loss_1(model,task,user_id,fake_normal_model)
        else:
            cos_loss=compute_cos_sim_loss(model,task,user_id)
        if len(output_poison) > 0:

            loss = loss_clean + loss_poison+0.5*eu_loss+0.5*cos_loss
        else:
            loss = loss_clean+0.5*eu_loss+0.5*cos_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def datafreeattack_run_fl_round(hlpr:Helper,epoch,former_model,s1,s2):
    local_epochs=15
    temp_lr= hlpr.params.lr
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model
    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()
    predicted_model=deepcopy(global_model)
    if epoch!=hlpr.params.start_epoch:
        new_state_dict=predict_the_global_model(s1,s2,alpha=0.8)
        predicted_model.load_state_dict(new_state_dict)
    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        make_optimizer = hlpr.task.make_optimizer(local_model)
        optimizer = make_optimizer
        if user.compromised:
            if hlpr.task.params.optimizer == 'SGD':
                lr = 0.005
                if hlpr.params.task == 'GTSRB':
                    lr = 0.00005
                if hlpr.params.task == 'CIFAR100':
                    lr = 0.001
                optimizer1 = optim.SGD(local_model.parameters(),
                                       lr=lr)
            elif hlpr.task.params.optimizer == 'Adam':
                optimizer1 = optim.Adam(local_model.parameters(),
                                        lr=0.005)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer1, hlpr.task.params.fl_poison_epochs, eta_min=1e-10)
            lambda1=1
            for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs)):
                criterion = nn.MSELoss()
                train_with_grad_control(local_model, epoch,user.train_loader, criterion, optimizer1, lambda1,hlpr.task,former_model,user.user_id,predicted_model)
                scheduler.step()
        else:
            for local_epoch in range(hlpr.params.fl_local_epochs):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=False)
        local_update = hlpr.attack.get_fl_update(local_model, global_model)
        hlpr.save_update(model=local_update, userID=user.user_id)
        if user.compromised:
            folder_name = f'{hlpr.params.folder_path}/saved_updates'
            file_name =folder_name+'/update_'+str(user.user_id)+'.pth'
            loaded_params = torch.load(file_name)
            for name, value in loaded_params.items():
                value.mul_(hlpr.params.fl_weight_scale)
            torch.save(loaded_params,file_name)
    hlpr.defense.aggr(weight_accumulator, global_model)
    hlpr.task.update_global_model(weight_accumulator, global_model)

def get_one_vec(model_or_state_dict):
    # Check if the input is a model or a state_dict
    if isinstance(model_or_state_dict, torch.nn.Module):
        state_dict = model_or_state_dict.state_dict()
    elif isinstance(model_or_state_dict, dict):
        state_dict = model_or_state_dict
    else:
        raise ValueError("Input must be a PyTorch model or state_dict.")

    size = sum(p.numel() for p in state_dict.values())
    sum_var = torch.zeros(size, device=device)
    index = 0
    for name, param in state_dict.items():
        #if 'fc' in name:
            numel = param.numel()
            sum_var[index:index + numel] = param.view(-1)
            index += numel

    return sum_var

def get_one_vec_variable(task,model, variable=False):
    size = 0
    if task.params.task=='GTSRB':
        s='fc2'
    else:
        s='fc'
    for name, layer in model.named_parameters():
        if s in name:
            size += layer.view(-1).shape[0]
    if variable:
        sum_var = Variable(torch.cuda.FloatTensor(size,device=device).fill_(0))
    else:
        sum_var = torch.cuda.FloatTensor(size,device=device).fill_(0)
    size = 0
    for name, layer in model.named_parameters():
        if s in name:
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

    return sum_var

def compute_cos_sim_loss(local_model,task,client_id):
    loss=0
    global_model=task.model
    global_vec = get_one_vec_variable(task, global_model, False)
    for i in range(client_id):

        local_vec = get_one_vec_variable(task,local_model,True)

        update_vec=local_vec-global_vec
        updates_name = f'{task.params.folder_path}/saved_updates/update_{i}.pth'
        loaded_params = torch.load(updates_name)
        other_model = deepcopy(global_model)
        other_model.load_state_dict(loaded_params)
        other_vec=get_one_vec_variable(task,other_model,False)
        cs_sim=F.cosine_similarity(update_vec,other_vec,dim=0)
        cs_sim=cs_sim **2
        loss+=cs_sim
    return loss

def compute_cos_sim_loss_1(local_model,task,client_id,fake_model):
    loss=0

    global_model=task.model
    global_vec = get_one_vec_variable(task, global_model, False)
    local_vec = get_one_vec_variable(task, local_model, True)
    update_vec = local_vec - global_vec
    for i in range(client_id):

        updates_name = f'{task.params.folder_path}/saved_updates/update_{i}.pth'
        loaded_params = torch.load(updates_name)
        other_model = deepcopy(global_model)
        other_model.load_state_dict(loaded_params)
        other_vec=get_one_vec_variable(task,other_model,False)
        cs_sim=F.cosine_similarity(update_vec,other_vec,dim=0)
        cs_sim=(cs_sim) **2
        loss+=cs_sim


    fake_vec=get_one_vec_variable(task,fake_model,False)
    fake_norm_update_vec=fake_vec-global_vec
    cs_sim = F.cosine_similarity(update_vec,fake_norm_update_vec,dim=0)
    cs_sim=(cs_sim)**2
    loss+=cs_sim

    return loss

def compute_cos_sim(local_model,global_model):
    local_vec=get_one_vec(local_model)
    global_vec=get_one_vec(global_model)
    cs_sim=F.cosine_similarity(local_vec,global_vec,dim=0)
    return cs_sim
def compute_euclidean_loss(model,fixed_model):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.cuda.FloatTensor(size, device=device).fill_(0)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (layer - fixed_model.state_dict()[name]).view(-1)
        size += layer.view(-1).shape[0]
    loss = torch.norm(sum_var, p=2)
    return loss
def run(hlpr: Helper):
    if hlpr.task.params.attack=='DataFreeAttack' or hlpr.task.params.attack=='DataFreeNoise' or hlpr.task.params.attack=='DataFreeNoiseForMnist' or hlpr.task.params.attack=='DataFreeAttackForMnist':
        if  hlpr.params.task == 'CIFAR100' or hlpr.params.task == 'Cifar10'or hlpr.params.task == 'GTSRB':
            hlpr.attack.perform_attack(hlpr.task)
            hlpr.params.start_epoch=hlpr.params.poison_epoch
            former_model=None
            current_model=deepcopy(hlpr.task.model)
            s1 = current_model.state_dict()
            s2 = None
            for epoch in range(hlpr.params.start_epoch,
                               hlpr.params.epochs + 1):
                if epoch==hlpr.params.start_epoch+1:
                    s2=hlpr.task.model.state_dict()
                if epoch>hlpr.params.start_epoch+1:
                    s1,s2=update_the_Ss(s1,s2,0.8,hlpr.task.model.state_dict())
                datafreeattack_run_fl_round(hlpr,epoch,former_model,s1,s2)
                former_model=deepcopy(current_model)
                current_model=deepcopy(hlpr.task.model)
                metric=datafreeattack_test(hlpr,epoch,backdoor=False)
                hlpr.record_accuracy(metric, datafreeattack_test(hlpr, epoch, backdoor=True), epoch)
                hlpr.save_model(hlpr.task.model, epoch, metric)
if __name__ == '__main__':
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', required=True)
    parser.add_argument('--name', dest='name', required=True)
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = args.name

    helper = Helper(params)
    logger.warning(create_table(params))

    try:
        run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. ")
        else:
            logger.error(f"Aborted training. No output generated.")
    helper.remove_update()
