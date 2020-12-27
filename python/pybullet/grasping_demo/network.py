#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F

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
        return embedded_obs
        #return hidden
   
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


