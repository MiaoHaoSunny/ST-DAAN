import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PeopleDataLoader import *
from CarDataLoader import *
from Pretrain.Shared_net import SharedNet
from Pretrain.model import MMDNet, PredictNet
from loss import *

data_split = [72*24, 20*24, 20*24]
# tgt_split = [32, 20*24, 20*24]
batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# source_pretrain_dataset = PeopleDataset(mode='train', split=data_split)
# source_pretrain_loader = DataLoader(dataset=source_pretrain_dataset, batch_size=32, shuffle=True)

# source_prevalidate_dataset = PeopleDataset(mode='validate', split=data_split)
# source_prevalidate_loader = DataLoader(dataset=source_prevalidate_dataset, batch_size=32, shuffle=False)

# source_pretest_dataset = PeopleDataset(mode='test', split=data_split)
# source_pretest_loader = DataLoader(dataset=source_pretest_dataset, batch_size=32, shuffle=False)

src_train_dataset = PeopleDataset(mode='train', split=data_split)
src_train_dataloader = DataLoader(dataset=src_train_dataset, batch_size=32, shuffle=True)

src_validate_dataset = PeopleDataset(mode='validate', split=data_split)
src_validate_dataloader = DataLoader(dataset=src_validate_dataset, batch_size=32, shuffle=False)

src_test_dataset = PeopleDataset(mode='test', split=data_split)
src_test_dataloader = DataLoader(dataset=src_test_dataset, batch_size=32, shuffle=False)

tgt_train_dataset = CarDataset(mode='train', split=data_split)
tgt_train_dataloader = DataLoader(dataset=tgt_train_dataset, batch_size=32, shuffle=False)

tgt_validate_dataset = CarDataset(mode='validate', split=data_split)
tgt_validate_dataloader = DataLoader(dataset=tgt_validate_dataset, batch_size=32, shuffle=False)

tgt_test_dataset = CarDataset(mode='test', split=data_split)
tgt_test_dataloader = DataLoader(dataset=tgt_test_dataset, batch_size=32, shuffle=False)

len_src_train = len(src_train_dataloader)
len_tgt_train = len(tgt_train_dataloader)

# lr = 0.0001
lr = 0.00001
l2_decay = 5e-4
num_epoches = 500

src_loss_list = []
total_loss_list = []
tgt_val_loss_list = []

seed = 32
np.random.seed(seed=seed)
torch.manual_seed(seed)
if device == 'cuda':
    torch.cuda.manual_seed(seed)

task_criterion = nn.MSELoss()

BaseNet = SharedNet().to(device)
BaseNet.load_state_dict(torch.load('./Trained_model/best_model_normlization.pkl'))
TransferNet = MMDNet().to(device)
TaskNet = PredictNet().to(device)

for param in BaseNet.parameters():
    param.requires_grad = False

optimizer = optim.Adam([
    {'params': BaseNet.conv3d_5.parameters()},
    {'params': TransferNet.parameters()},
    {'params': TaskNet.parameters()}], lr=lr, weight_decay=l2_decay)
best_rmse = 10000

for epoch in range(num_epoches):
    t0 = time.time()
    BaseNet.train()
    TransferNet.train()
    TaskNet.train()

    src_train_aver_rmse = 0
    mmd_loss = 0
    
    iter_src = iter(src_train_dataloader)
    iter_tgt = iter(tgt_train_dataloader)

    num_iter = len_src_train

    for i in range(0, num_iter):
        src_data_x, src_data_y = next(iter_src)
        tgt_data_x, tgt_data_y = next(iter_tgt)

        if (i+1) % len_tgt_train == 0:
            iter_tgt = iter(tgt_train_dataloader)
        
        src_data_x = src_data_x.float().to(device)
        src_data_y = src_data_y.float().to(device)
        tgt_data_x = tgt_data_x.float().to(device)
        tgt_data_y = tgt_data_y.float().to(device)

        optimizer.zero_grad()
        
        inputs = torch.cat((src_data_x, tgt_data_x), dim=0)
        features = BaseNet(inputs)
        features = TransferNet(features)
        outputs = TaskNet(features)

        # print(outputs.shape, src_data_y.shape, inputs.size(0)/2)

        task_loss = torch.sqrt(task_criterion(outputs.narrow(0, 0, int(inputs.size(0)/2)), src_data_y))

        transfer_loss = DAN(features.narrow(0, 0, int(features.size(0)/2)), features.narrow(0, int(features.size(0)/2), int(features.size(0)/2)))

        total_loss = 0.1*transfer_loss + task_loss
        
        src_train_aver_rmse += total_loss.item()
        mmd_loss += transfer_loss.item()
        total_loss.backward()
        optimizer.step()
    src_train_aver_rmse /= len_src_train
    mmd_loss /= len_src_train
    src_loss_list.append(src_train_aver_rmse)
    total_loss_list.append(src_train_aver_rmse+mmd_loss)
    
    if (epoch+1) % 5 == 0 or epoch == 0:
        BaseNet.eval()
        TransferNet.eval()
        TaskNet.eval()
        tgt_validate_aver_rmse = 0
        len_tgt_validate = len(tgt_validate_dataloader)
        for i, (tgt_data_x, tgt_data_y) in enumerate(tgt_validate_dataloader):
            tgt_data_x, tgt_data_y = tgt_data_x.float().to(device), tgt_data_y.float().to(device)
            features = TransferNet(BaseNet(tgt_data_x))
            tgt_output = TaskNet(features)
            tgt_loss = torch.sqrt(task_criterion(tgt_output, tgt_data_y))
            tgt_validate_aver_rmse += tgt_loss.item()
        tgt_validate_aver_rmse /= len_tgt_validate
        tgt_val_loss_list.append(tgt_validate_aver_rmse)
        if tgt_validate_aver_rmse < best_rmse:
            best_rmse = tgt_validate_aver_rmse
            torch.save(BaseNet.state_dict(), 'best_BaseNet_transfer_gamma_0.3_500epochs_64hours.pkl')
            torch.save(TransferNet.state_dict(), 'best_TransferNet_transfer_gamma_0.3_500epochs_64hours.pkl')
            torch.save(TaskNet.state_dict(), 'best_TaskNet_transfer_gamma_0.3_500epochs_64hours.pkl')
    t1 = time.time()
    print('Epoch: [{}/{}], Source train loss: {}, MMD loss: {}, tgt_best_validate_loss: {}, Cost {}min.'.format(epoch+1, num_epoches, src_train_aver_rmse, mmd_loss, best_rmse, (t1-t0)/60))

# loss_train_list = np.array(loss_train_list)
# loss_validate_list = np.array(loss_validate_list)
# np.save('loss_train_normlization.npy', loss_train_list)
# np.save('loss_validate_normlization.npy', loss_validate_list)

src_loss_list = np.array(src_loss_list)
total_loss_list = np.array(total_loss_list)
tgt_val_loss_list = np.array(tgt_val_loss_list)

np.save('src_loss_train_64hours.npy', src_loss_list)
np.save('total_loss_train_64hours.npy', total_loss_list)
np.save('tgt_loss_validate_64hours.npy', tgt_val_loss_list)