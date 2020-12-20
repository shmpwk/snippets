#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
import argparse
import glob
#import pybullet_envs  # PyBulletの環境をgymに登録する

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from buffer import SimpleBuffer
from grasp_utils import *

class Conv_AE(nn.Module):
    def  __init__(self, embedding_dimension):
       super(Convolutional_AutoEncoder, self).__init__()

       # define the network
       # encoder
       self.conv1 = nn.Sequential(nn.ZeroPad2d((1,2,1,2)),
                             nn.Conv2d(1, 32, kernel_size=5, stride=2),
                             nn.ReLU())
       self.conv2 = nn.Sequential(nn.ZeroPad2d((1,2,1,2)),
                             nn.Conv2d(32, 64, kernel_size=5, stride=2),
                             nn.ReLU())
       self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
                             nn.ReLU())
       self.fc1 = nn.Conv2d(128, 8, kernel_size=3)
       self.fc2 = nn.Linear(8 + 8, 16)
       self.fc3 = nn.Linear(16, 16)
 

       # decoder
       self.fc2 = nn.Sequential(nn.ConvTranspose2d(16, 128, kernel_size=3),
                           nn.ReLU())
       self.conv3d = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
                              nn.ReLU())
       self.conv2d = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
                              nn.ReLU())
       self.conv1d = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2)

    def forward(self, x):
        encoded = self.fc1(self.conv3(self.conv2(self.conv1(x))))

        decoded = self.fc2(encoded)
        decoded = self.conv3d(decoded)
        decoded = self.conv2d(decoded)[:,:,1:-2,1:-2]
        decoded = self.conv1d(decoded)[:,:,1:-2,1:-2]
        decoded = nn.Sigmoid()(decoded)
        return encoded, decoded   

class Encoder(nn.Module):
    """
    (4, 64, 64)の画像を(1024,)のベクトルに変換するエンコーダ
    """
    def __init__(self):
        super(Encoder, self).__init__()
        """
        self.cv1 = nn.Conv2d(4, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        """
        self.conv1 = nn.Conv2d(2, 4, 3, 2, 1)
        self.cbn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, 2, 1)
        self.cbn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3, 2, 1)
        self.cbn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, 2, 1)
        self.cbn4 = nn.BatchNorm2d(32)
        #self.conv5 = nn.Conv2d(32, 64, 3, 2, 1)
        #self.cbn5 = nn.BatchNorm2d(64)
        #self.fc1 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8 + 8, 16)
        self.fc5 = nn.Linear(16, 16) 

        self.dfc1 = nn.Linear(16, 32)
        self.dfc2 = nn.Linear(32, 64)
        self.dfc3 = nn.Linear(64, 128)
        self.dfc4 = nn.Linear(128, 256)
        self.dfc5 = nn.Linear(256, 512)
        """
        self.dcv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)
        self.dcbn1 = nn.BatchNorm2d(128)
        self.dcv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.dcbn2 = nn.BatchNorm2d(64)
        self.dcv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.dcbn3 = nn.BatchNorm2d(32)
        self.dcv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)
        """
        self.dcv0 = nn.ConvTranspose2d(516, 256, 2, 2, 0)
        self.dcbn0 = nn.BatchNorm2d(256)
        self.dcv1 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.dcbn1 = nn.BatchNorm2d(128)
        self.dcv2 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.dcbn2 = nn.BatchNorm2d(64)
        self.dcv3 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.dcbn3 = nn.BatchNorm2d(32)
        self.dcv4 = nn.ConvTranspose2d(32, 16, 2, 2, 0)
        self.dcbn4 = nn.BatchNorm2d(16)
        self.dcv5 = nn.ConvTranspose2d(16, 8, 2, 2, 0)
        self.dcbn5 = nn.BatchNorm2d(8)
        self.dcv6 = nn.ConvTranspose2d(8, 4, 2, 2, 0)
        self.dcbn6 = nn.BatchNorm2d(4)
        self.dcv7 = nn.ConvTranspose2d(4, 2, 2, 2, 0)
        self.dcbn7 = nn.BatchNorm2d(2)
        self.dcv8 = nn.ConvTranspose2d(2, 1, 2, 2, 0)
        self.dcbn8 = nn.BatchNorm2d(1)


    def forward(self, x, y):
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.cbn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.cbn2(x)
        """
        x, idx1 = F.max_pool2d(F.relu(self.conv1(x)), 2, return_indices=True)
        size1 = x.size()
        x = self.cbn1(x)
        x, idx2 = F.max_pool2d(F.relu(self.conv2(x)), 2, return_indices=True)
        size2 = x.size()
        x = self.cbn2(x)
        x = F.relu(self.conv3(x))
        x = self.cbn3(x)
        x = F.relu(self.conv4(x))
        x = self.cbn4(x)
        #x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        #x = self.cbn5(x)
        x = x.view(-1, self.num_flat_features(x))
        #depth_data =depth_data.view(depth_data.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        #return z
        obs = z
        x = F.relu(self.dfc1(obs))
        x = F.relu(self.dfc2(x))
        x = F.relu(self.dfc3(x))
        #x = F.relu(self.dfc4(x))
        #x = F.relu(self.dfc5(x))
        x = torch.reshape(x, (x.size()[0], -1, 2, 2))
        hidden = x
        #hidden = F.relu(self.dcv1(x))
        #hidden = self.dcbn1(hidden)
        #hidden = F.relu(self.dcv2(x))
        #print(hidden.shape)
        #hidden = self.dcbn2(hidden)
        #print(hidden.shape)
        #hidden = self.dcv3(hidden)
        #print(hidden.shape)
        #hidden = self.dcbn3(hidden)
        #print(hidden.shape)
        hidden = self.dcv4(hidden)
        hidden = self.dcbn4(hidden)
        hidden = self.dcv5(hidden)
        hidden = F.max_unpool2d(F.relu(hidden), idx2, 2)#, output_size=torch.Size([2, 2, 32, 32]))
        #hidden = F.max_unpool2d(F.relu(self.dcv6(hidden)), idx2, 2, output_size=torch.Size([2, 4, 32, 32]))
        hidden = self.dcbn5(hidden)
        #hidden = self.dcbn6(hidden)
        x = torch.reshape(x, (x.size()[0], -1, 2, 2))
        hidden = F.max_unpool2d(F.relu(self.dcv6(hidden)), idx1, 2)
        hidden = self.dcbn6(hidden)
        hidden = self.dcv7(hidden)
        hidden = self.dcbn7(hidden)
        embedded_obs = hidden.reshape(hidden.size(0), -1) 
        #return embedded_obs
        return hidden
   
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Predictor(nn.Module):
    """
    inputDim : 8(robot) + 8?(img)
    hiddenDim : any number is ok?
    outputDim : same as imputDim? or only robot state dim?
    """
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)

    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0) #LSTM層
        output = self.output_layer(output[:, -1, :]) #全結合層
        return output

class ObservationModel(nn.Module):
    """
    p(o_t | s_t, h_t)
    inputDim : compressed image (and robot state?)
    低次元の状態表現から画像を再構成するデコーダ (4, 64, 64)
    """
    def __init__(self, state_dim, rnn_hidden_dim):
        super(ObservationModel, self).__init__()
        self.fc = nn.Linear(state_dim + rnn_hidden_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 4, kernel_size=6, stride=2)

    def forward(self, state, rnn_hidden):
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1))
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        hidden = F.relu(self.dc2(hidden))
        hidden = F.relu(self.dc3(hidden))
        obs = self.dc4(hidden)
        return obs
"""
# モデルの宣言
state_dim = 30  # 確率的状態の次元
rnn_hidden_dim = 200  # 決定的状態（RNNの隠れ状態）の次元
encoder = Encoder().to(device)
rssm = RecurrentStateSpaceModel(state_dim,
                                   env.action_space.shape[0],
                                   rnn_hidden_dim).to(device)
obs_model = ObservationModel(state_dim, rnn_hidden_dim).to(device)
"""

def mkDataSet(path, data_size, data_length=50, freq=1, noise=0.00):
    """
    params\n
    data_size : データセットサイズ\n
    data_length : 各データの時系列長\n
    freq : 周波数\n
    noise : ノイズの振幅\n
    returns\n
    train_x : トレーニングデータ（t=1,2,...,size-1の値)\n
    train_t : トレーニングデータのラベル（t=sizeの値）\n
    """
    train_x = []
    train_t = []
    train_xrgb = []
    train_trgb = []
    train_xdepth = []
    train_tdepth = []
    buffer_size = 500
    rgb_shape = 128*128*4
    depth_shape = 128*128*4
    state_shape = 8
    robot_p = "robot_" + path + ".pth"
    rgb_p = "rgb_" + path + ".pth"
    depth_p = "depth_" + path + ".pth"
    robot_path = os.path.join("data/robot_data", robot_p) 
    rgb_path = os.path.join("data/rgb_data", rgb_p)
    depth_path = os.path.join("data/depth_data", depth_p)
    buf = SimpleBuffer(data_length, buffer_size, rgb_shape, depth_shape, state_shape, device=torch.device('cuda'))
    rgb_buffer, depth_buffer, data_buffer = buf.load(rgb_path, depth_path, robot_path)
    rgb_buffer = rgb_buffer.to('cpu').detach().numpy().copy()
    depth_buffer = depth_buffer.to('cpu').detach().numpy().copy()
    data_buffer = data_buffer.to('cpu').detach().numpy().copy()

    for offset in range(data_size):
        train_xrgb.append([rgb_buffer[int((offset + i) / freq)] for i in range(data_length)])
        train_xdepth.append([depth_buffer[int((offset + i) / freq)] for i in range(data_length)])
        train_x.append([data_buffer[int((offset + i) / freq)] for i in range(data_length)])
        train_trgb.append([rgb_buffer[int((offset + data_length) / freq)]])
        train_tdepth.append([depth_buffer[int((offset + data_length) / freq)]])
        train_t.append([data_buffer[int((offset + data_length) / freq)]])
    return train_x, train_t, train_xrgb, train_trgb, train_xdepth, train_tdepth

def mkRandomBatch(train_x, train_t, train_xrgb, train_trgb, train_xdepth, train_tdepth, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []
    batch_xrgb = []
    batch_trgb = []
    batch_xdepth = []
    batch_tdepth = []
    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
        batch_xrgb.append(train_xrgb[idx])
        batch_trgb.append(train_trgb[idx])
        batch_xdepth.append(train_xdepth[idx])
        batch_tdepth.append(train_tdepth[idx])
    
    return torch.tensor(batch_x), torch.tensor(batch_t), torch.tensor(batch_xrgb), torch.tensor(batch_trgb), torch.tensor(batch_xdepth), torch.tensor(batch_tdepth)


#class Algorithm():
#    def __init__(self):
#        self.buffer = SimpleBuffer(buffer_size, 8, state_shape, devicei=torch.device('cuda'))
#
#    def update(self):
#        self.learning_steps += 1
#        states = self.buffer.get()

def main(path):
    training_size = 20
    test_size = 20
    epochs_num = 20
    input_size = 1 
    hidden_size = 8
    batch_size = 2
    data_length = 50
    device=torch.device('cuda')
    train_x, train_t, train_xrgb, train_trgb, train_xdepth, train_tdepth = mkDataSet(path, training_size)
    test_x, test_t, test_xrgb, test_trgb,  test_xdepth, test_tdepth = mkDataSet(path, test_size)

    model = Predictor(8, hidden_size, 8)
    #encoder = Encoder(4, 128, 128)
    encoder = Encoder().to(device)
    #decoder = Decoder(4, 64, 64)

    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    #summary(encoder, [(2, 128, 128), (8,)])

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()

            data, label, rgb, rgb_label, depth, depth_label = mkRandomBatch(train_x, train_t, train_xrgb, train_trgb, train_xdepth, train_tdepth, batch_size)
            rgb = rgb.reshape(batch_size, data_length, 4, 128, 128)
            rgb_label = rgb_label.reshape(batch_size, 4, 128, 128)
            depth = depth.reshape(batch_size, data_length, 1, 128, 128)
            depth_label = depth_label.reshape(batch_size, 1, 128, 128)
 
            rgb = rgb[:,data_length-1,:3,:,:].reshape(batch_size, 3, 128, 128)
            rgb_label = rgb_label[:,:3,:,:].reshape(batch_size, 3, 128, 128)
            im_gray = 0.299 * rgb[:, 0, :, :] + 0.587 * rgb[:, 1, :, :] + 0.114 * rgb[:, 2, :, :].reshape(batch_size, 1, 128, 128)
            im_gray_label = 0.299 * rgb_label[:, 0, :, :] + 0.587 * rgb_label[:, 1, :, :] + 0.114 * rgb_label[:, 2, :, :].reshape(batch_size, 1, 128, 128)

            print(im_gray_label.shape)
            #im_gray_label = im_gray_label.reshape(batch_size, 1, 128, 128)
            encoding_gray = im_gray.to(device)
            encoding_gray_label = im_gray_label.to(device)
            encoding_depth = depth[:,data_length-1,:,:,:].reshape(batch_size, 1 , 128, 128)
            print(encoding_gray_label.shape)
            #encoding_depth_label = depth_label[:,:,:,:].reshape(batch_size, 1*128*128)
            encoding_depth = encoding_depth.to(device)
            encoding_depth_label = encoding_depth.to(device)
            output = model(data).to(device)
            encoding_img = torch.cat([encoding_gray, encoding_depth], dim=1)
            print(encoding_depth_label.shape)
            print(encoding_gray_label.shape)
            print("=============")
            encoding_img_label = torch.cat([encoding_gray_label, encoding_depth_label], dim=1).reshape(batch_size, 2*128*128)
            encoded = encoder(encoding_img, output).reshape(hidden.size(0), -1)

            #encoded = encoded.reshape()
            #decoded = decoder(encoded)
            img_label = encoding_img_label.to(device)
            loss = criterion(encoded, img_label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            training_accuracy += np.sum(np.abs((encoded.data.cpu() - img_label.data.cpu()).numpy()) < 0.1)

        #test
        test_accuracy = 0.0
        for i in range(int(test_size / batch_size)):
            offset = i * batch_size
            data, label = torch.tensor(test_x[offset:offset+batch_size]), torch.tensor(test_t[offset:offset+batch_size])
            output = model(data, None)

            test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)
        
        training_accuracy /= training_size
        test_accuracy /= test_size

        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', '-p', type=str, help='set loading data path', default='buffer')
    args = parser.parse_args()
    main(args.data_path)

