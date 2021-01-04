#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
import argparse
import glob
import datetime
#import pybullet_envs  # PyBulletの環境をgymに登録する

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch.optim import *
from torch.utils.data import DataLoader         
from torch.utils.data import Dataset    
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision

from buffer import SimpleBuffer
from grasp_utils import *
from network import *
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, path, data_size, data_length=50, freq=1, noise=0.00):
        self.datanum = 100/2
        """
        params\n
        data_size : データセットサイズ\n
        data_length : 各データの時系列長\n
        freq : 周波数\n
        noise : ノイズの振幅\n
        returns\n
        """
        buffer_size = 5000
        rgb_shape = 128*128*4
        depth_shape = 128*128*4
        robot_state_shape = 14
        state_shape = 8
        robot_p = "robot_" + path + ".pth"
        rgb_p = "rgb_" + path + ".pth"
        depth_p = "depth_" + path + ".pth"
        robot_state_path = os.path.join("data/robot_state_data", robot_p) 
        robot_path = os.path.join("data/robot_data", robot_p) 
        rgb_path = os.path.join("data/rgb_data", rgb_p)
        depth_path = os.path.join("data/depth_data", depth_p)
        buf = SimpleBuffer(data_length, buffer_size, rgb_shape, depth_shape, state_shape, robot_state_shape, device=torch.device('cuda'))
        rgb_buffer, depth_buffer, data_buffer, robot_state_buffer = buf.load(rgb_path, depth_path, robot_path, robot_state_path)
        rgb_buffer = rgb_buffer.to('cpu').detach().numpy().copy()
        depth_buffer = depth_buffer.to('cpu').detach().numpy().copy()
        data_buffer = data_buffer.to('cpu').detach().numpy().copy()
        robot_state_buffer = robot_state_buffer.to('cpu').detach().numpy().copy()
        rgb_buffer = rgb_buffer.reshape(data_size, -1)
        depth_buffer = depth_buffer.reshape(data_size, -1)
        data_buffer = data_buffer.reshape(data_size, -1)
        robot_state_buffer = robot_state_buffer.reshape(data_size, -1)
        rgb = rgb_buffer.reshape(data_size, data_length, 4, 128, 128)
        depth = depth_buffer.reshape(data_size, data_length, 1, 128, 128)
        data = data_buffer.reshape(data_size, data_length, 8)
        robot_state = robot_state_buffer.reshape(data_size, data_length, 14)

        """
        make rgb + depth data
        """
        # depth: from 1 channel to 3 channel
        """ Fail to convert from 1 to 3 channel
        #pil_depth = np.transpose(depth[0,0,0,:,:], (1,2,0))
        gray = Image.fromarray((depth[0,0,0,:,:]*255).astype(np.uint8)).convert("RGB")
        gray = np.array(gray)
        plt.imshow(gray, vmin=0, vmax=255)
        #plt.show()
        #gray.show()
        
        gray2 = (depth[0,0,0,:,:]*255)#.astype(np.uint8)
        plt.imshow(gray2)
        #plt.show()
       
        backtorgb = cv2.cvtColor((depth[0,0,0,:,:]*255), cv2.COLOR_GRAY2RGB)
        plt.imshow(backtorgb[:,:,0])
        #plt.show()
        """
        depth = (depth*255).astype(np.uint8).reshape(data_size, data_length, 128, 128, 1)
        depth = np.tile(depth, (1,1,1,1,3))
        plt.imshow(depth[0,0,:,:,:])
        #plt.show()
        
        #rgb
        rgb = rgb.reshape(data_size, data_length, 128, 128, 4)
        rgb = rgb[:,:,:,:,:3]
        plt.imshow(rgb[0,0,:,:,:3], vmin=0, vmax=255)
        #plt.show()
        
        rgb = np.transpose(rgb, (0, 1, 4, 2, 3))
        depth = np.transpose(depth, (0, 1, 4, 2, 3))
        rgb_dataset = rgb.reshape((data_size, data_length, 3, 128, 128))
        depth_dataset = depth.reshape((data_size, data_length, 3, 128, 128))    
        self.rgbd_dataset = np.concatenate([rgb_dataset, depth_dataset], 2).astype(np.uint8)
        self.robot_dataset = np.array(robot_state, dtype=np.uint8) #.astype(np.uint8)

    def __len__(self):
        return int(self.datanum) #should be dataset size / batch size

    def __getitem__(self, idx):
        """
        return:
        x.shape(6,128,128)
        c.shape(8+6)
        """
        x = self.rgbd_dataset[idx]
        #y = self.grasp_dataset[idx]
        c = self.robot_dataset[idx]
        x = torch.from_numpy(x).int()
        #y = torch.from_numpy(y).float()
        c = torch.from_numpy(np.array(c)).int()
        return x, c

class GraspSystem():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load depth_image and grasp_pos_rot data
    def load_data(self, datasets):
        # Data loader (https://ohke.hateblo.jp/entry/2019/12/28/230000)
        train_dataloader = torch.utils.data.DataLoader(
            datasets, 
            batch_size=2, 
            shuffle=False,
            num_workers=2,
            drop_last=True
        )
        rgbd_data, robot_data = next(iter(train_dataloader))
        return train_dataloader
    
    def make_model(self):
        self.model = Predictor(8+6, hidden_size+6, 8+6).to(device)
        self.ae = Encoder6to14().to(device)
        self.criterion = nn.MSELoss()
        self.lstm_optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.encoder_optimizer = optim.Adam(self.ae.parameters(), lr=0.001)
        #summary(self.model, [(2, 128, 128), (4,)])  
   
    def save_model(self):
        now = datetime.datetime.now()   
        lstm_filename = 'data/trained_lstm_model/model_' + now.strftime('%Y%m%d_%H%M%S') + '.pth'
        encoder_filename = 'data/trained_encoder_model/model_' + now.strftime('%Y%m%d_%H%M%S') + '.pth'
        lstm_model_path = lstm_filename      
        encoder_model_path = encoder_filename      
        # GPU save   
        ## Save only parameter   
        torch.save(self.model.state_dict(), lstm_model_path) 
        torch.save(self.ae.state_dict(), encoder_model_path) 
        ## Save whole model
        #torch.save(self.model(), lstm_model_path)   
        #torch.save(self.ae(), encoder_model_path)   
        # CPU save  
        #torch.save(self.model.to('cpu').state_dict(), model_path)
        print("Finished Saving model")    

    def train(self, train_dataloader, loop_num):
        training_size = 50
        test_size = 50
        epochs_num = 100
        input_size = 1 
        hidden_size = 8
        batch_size = 2
        data_length = 50
        device=torch.device('cuda')
        #summary(ae, [(2, 128, 128), (8,)])
        now = datetime.datetime.now()
        tensorboard_cnt = 0
        log_dir = 'data/loss/loss_' + now.strftime('%Y%m%d_%H%M%S')
        for epoch in range(epochs_num):
            # training
            running_loss = 0.0
            training_accuracy = 0.0
            for i, data in enumerate(train_dataloader, 0):
                self.lstm_optimizer.zero_grad()
                self.encoder_optimizer.zero_grad()
                rgbd, robot = data 
                rgbd = rgbd.float()
                robot = robot.float().to(device)
                encoded_all = torch.tensor(np.empty((batch_size, 1, 8+6))).to(device)
                
                for j in range(data_length):
                    rgbd_j = rgbd[:,j,:,:,:].reshape(batch_size, 6, 128, 128).to(device)
                    encoded = self.ae(rgbd_j).to(device).reshape(batch_size, 1, 8+6) #.reshape(hidden.size(0), -1)
                    if j==0:
                        encoded_all = encoded
                    else:
                        encoded_all = torch.cat((encoded_all, encoded), axis=1)
                output = self.model(encoded_all.to(device)).to(device) #data shape should be (sequence_length, batch_size, vector dim) if not batch first.
                loss = self.criterion(robot, output)
                loss.backward()
                self.lstm_optimizer.step()
                self.encoder_optimizer.step()

                running_loss += loss.item()
                training_accuracy += np.sum(np.abs((robot.data.cpu() - output.data.cpu()).numpy()) < 0.1)
                writer = SummaryWriter(log_dir)
                writer.add_scalar("Loss/train", loss.item(), tensorboard_cnt) #(epoch + 1) * i)
                if i % 100 == 99: 
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
                tensorboard_cnt += 1
            print('%d loss: %.3f, training_accuracy: %.5f' % (
                epoch + 1, running_loss, training_accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', '-p', type=str, help='set loading data path', default='buffer')
    parser.add_argument('--training_size', '-n', type=int, help='set train size', default='100')
    args = parser.parse_args()
    #training_size = 50
    test_size = 50
    epochs_num = 50
    input_size = 1 
    hidden_size = 8
    batch_size = 2
    data_length = 50
    device=torch.device('cuda')
    gs = GraspSystem()
    datasets = MyDataset(args.data_path, args.training_size)
    train_dataloader = gs.load_data(datasets)
    gs.make_model()
    gs.train(train_dataloader, loop_num=100)
    gs.save_model()

