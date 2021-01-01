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
import torchvision

from buffer import SimpleBuffer
from grasp_utils import *
from network import *

def mkDataSet(path, data_size, data_length=5, freq=1, noise=0.00):
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
    buffer_size = 5000
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
    print("data_size", data_size)
    print("data_length", data_length)
    #print([rgb_buffer[i] for i in range(500)])
    #print([rgb_buffer[i][100] for i in range(data_length*data_size)])
    #print("aaaa")
    for offset in range(data_size): #data_size is 100? not sequence length.
        train_xrgb.append([rgb_buffer[int((offset + i) / freq)] for i in range(data_length)])
        #print([rgb_buffer[int((offset + i) / freq)] for i in range(data_length)])
        train_xdepth.append([depth_buffer[int((offset + i) / freq)] for i in range(data_length)])
        train_x.append([data_buffer[int((offset + i) / freq)] for i in range(data_length)])
        train_trgb.append([rgb_buffer[int((offset + data_length) / freq)]])
        #print(rgb_buffer[int((offset + data_length) / freq)])
        train_tdepth.append([depth_buffer[int((offset + data_length) / freq)]])
        train_t.append([data_buffer[int((offset + data_length) / freq)]])
        """
        for x in train_trgb:
            if x != 0 :
                print(x)
        """
    print("train x", np.array(train_x).shape)
    # Each list size is 50
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
    print("batch x ", np.array(batch_x).shape) 
    return torch.tensor(batch_x), torch.tensor(batch_t), torch.tensor(batch_xrgb), torch.tensor(batch_trgb), torch.tensor(batch_xdepth), torch.tensor(batch_tdepth)

def main(path):
    training_size = 50
    test_size = 50
    epochs_num = 100
    input_size = 1 
    hidden_size = 8
    batch_size = 2
    data_length = 50
    device=torch.device('cuda')
    train_x, train_t, train_xrgb, train_trgb, train_xdepth, train_tdepth = mkDataSet(path, training_size)
    test_x, test_t, test_xrgb, test_trgb,  test_xdepth, test_tdepth = mkDataSet(path, test_size)

    model = Predictor(8, hidden_size, 8)
    ae = AE().to(device)

    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    #summary(ae, [(2, 128, 128), (8,)])

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
            #rgb = rgb[:,data_length-10,:3,:,:].reshape(batch_size, 3, 128, 128)
            rgb = rgb[:,data_length-1,:,:,:].reshape(batch_size, 128, 128, 4)
            rgb_label = rgb_label[:,:3,:,:].reshape(batch_size, 3, 128, 128)
            #print(rgb)
            #print("=============")
            #plt.imshow(np.transpose((rgb[0,:,:,:] / 2 + 0.5).numpy(), (1, 2, 0)))
            #plt.imshow((rgb[0,:,:,:] / 2 + 0.5).numpy())
            """
            # showed strange img
            rgb = rgb.reshape(batch_size, 4, 128, 128)
            rgb = rgb[:,:3,:,:]
            rgb=rgb.reshape(batch_size, 128, 128, 3)
            #rgb[batch_size,128,128,:].shape
            plt.imshow(rgb[0,:,:,:].numpy(), vmin=0, vmax=255)
            plt.show()
            """

            """
            # succeeded to show good rgb img
            rgb = np.transpose(rgb, (0, 3, 1, 2))
            rgb = rgb[:,:3,:,:]
            rgb = np.transpose(rgb, (0, 2, 3, 1))
            #rgb[batch_size,128,128,:].shape
            plt.imshow(rgb[0,:,:,:].numpy(), vmin=0, vmax=255)
            plt.show()
            """
            rgb = np.transpose(rgb, (0, 3, 1, 2))
            im_gray = 0.299 * rgb[:, 0, :, :] + 0.587 * rgb[:, 1, :, :] + 0.114 * rgb[:, 2, :, :]
            im_gray = im_gray.reshape(batch_size, 1, 128, 128)
            im_gray_label = 0.299 * rgb_label[:, 0, :, :] + 0.587 * rgb_label[:, 1, :, :] + 0.114 * rgb_label[:, 2, :, :]
            im_gray_label = im_gray_label.reshape(batch_size, 1, 128, 128)
            encoding_gray = im_gray.to(device)
            encoding_gray_label = im_gray_label.to(device)
            encoding_depth = depth[:,data_length-1,:,:,:].reshape(batch_size, 1 , 128, 128)
            encoding_depth = encoding_depth.to(device)
            encoding_depth_label = encoding_depth.to(device)
            #img show
            #img = torchvision.utils.make_grid(im_gray)
            #img = img / 2 + 0.5  # [-1,1] を [0,1] へ戻す(正規化解除)
            #npimg = img.numpy()  # torch.Tensor から numpy へ変換
            #print(npimg.shape)
            ims = im_gray[1,:,:,:].reshape((1, 128, 128))
            #plt.imshow(ims[0,:,:])
            #plt.imshow(np.transpose(im_gray_label[0,:,:,:], (1,2,0))) # チャンネルを最後に並び変える((C,X,Y) -> (X,Y,C))
            #plt.show()
            output = model(data).to(device)
            encoding_img = torch.cat([encoding_gray, encoding_depth], dim=1)
            encoding_img_label = torch.cat([encoding_gray_label, encoding_depth_label], dim=1).reshape(batch_size, 2*128*128)
            estimated_robot_state, encoded = ae(encoding_img, output)#.reshape(hidden.size(0), -1)
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
            data, label, rgb, rgb_label, depth, depth_label = torch.tensor(test_x[offset:offset+batch_size]), torch.tensor(test_t[offset:offset+batch_size]), torch.tensor(test_xrgb[offset:offset+batch_size]), torch.tensor(test_trgb[offset:offset+batch_size]), torch.tensor(test_xdepth[offset:offset+batch_size]), torch.tensor(test_tdepth[offset:offset+batch_size])
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
