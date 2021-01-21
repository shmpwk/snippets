#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from math import *
import argparse
import pybullet 
import pybullet as pb
import pybullet_data
import numpy as np
import time
import six
import utils
import torch

from grasp_utils import *
from buffer import *
from gripper import RGripper, LGripper
from network import *


class Simulator(object):
    def __init__(self):
        CLIENT = pybullet.connect(pybullet.GUI)
        print("client",CLIENT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
        plane = pybullet.loadURDF("plane.urdf")
        print("plane ID", plane)
        #pb.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, physicsClientId=CLIENT)
        #pb.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=CLIENT)
        pb.setGravity(0,0,-10.0)
        self.table = pb.loadURDF("table/table.urdf")
        print("table ID", self.table)
        self.plate = pb.loadURDF("dish/plate.urdf")
        print("plate ID", self.plate)
        self.rgripper = RGripper()
        self.lgripper = LGripper()
        self.box = pb.loadURDF("dish/box.urdf")
        print("cube ID", self.box)
        self.frames = [] #Movie buffer
        self.d_frames = [] #Movie buffer
        pb.resetDebugVisualizerCamera(0.4, 90, -75, (0.25, 0.0, 1.0))
        
        # For pybullet getCameraImage argument
        self.viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=[5, 5, 30],
            cameraTargetPosition=[3, 3, 3],
            cameraUpVector=[3, 1, 3])
        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1) 
 
    def get_data(self):
        """
         Right and Left gripper pos, rot, width
        """
        rx, ry, rz = self.rgripper.get_point()
        lx, ly, lz = self.lgripper.get_point()
        rtheta = self.rgripper.get_pose()
        ltheta = self.lgripper.get_pose()
        rw = self.rgripper.get_gripper_state()
        lw = self.lgripper.get_gripper_state()
        return rx, ry, lx, ly, rtheta, ltheta, rw, lw

    def load_model(self, time):
        hidden_size = 8
        device=torch.device('cuda')
        self.model = Predictor(8, hidden_size, 8).to(device)
        self.ae = ImgRobotEncoder().to(device)
        lstm_filename = 'data/trained_lstm_model/model_' + str(time) + '.pth'
        encoder_filename = 'data/trained_encoder_model/model_' + str(time) + '.pth'
        # learn GPU, load GPU
        self.model.load_state_dict(torch.load(lstm_filename))
        self.ae.load_state_dict(torch.load(encoder_filename))
        # learn CPU, load GPU
        #self.model.load_state_dict(torch.load(lstm_filename, map_location=torch.device('cpu')))
        #self.ae.load_state_dict(torch.load(encoder_filename, map_location=torch.device('cpu')))
        self.model.eval()
        self.ae.eval()

    def reset(self):
        table_pos = np.array([0.0, 0.0, 0.0])
        utils.set_point(self.table, table_pos)
        utils.set_zrot(self.table, pi*0.5)
        table_x = np.random.rand()-0.5
        table_y = np.random.rand()-0.5
        utils.set_point(self.plate, [table_x, table_y, 0.63])
        plate_pos = utils.get_point(self.plate) #Get target obj center position
        self.rgripper.set_basepose(np.array([0, 0.25, 0.78]) + np.array([plate_pos[0], plate_pos[1], 0]), [-1.54, 0.5, -1.57])
        self.rgripper.set_state([0, 0, 0])
        self.rgripper.set_angle(self.rgripper.gripper, 0)
        self.lgripper.set_basepose(np.array([0, -0.24, 0.78]) + np.array([plate_pos[0], plate_pos[1], 0]), [1.54, 0.65, 1.57])
        self.lgripper.set_state([0, 0, 0])
        self.lgripper.set_angle(self.lgripper.gripper, 0)
        self.rgripper.set_gripper_width(0.5, force=True)
        self.lgripper.set_gripper_width(0.5, force=True)
        for i in range(100):
            pb.stepSimulation()

    def rollout(self, data_length, try_num, buffer_size, rgb_shape, depth_shape, state_shape, robot_state_shape, device=torch.device('cuda')):
        """
        State:
        - rx, 
        - ry, 
        - lx, 
        - ly, 
        - rtheta, 
        - ltheta, 
        - rw, 
        - lw,
        - r1: Degree of contact on the inside of the Rgripper,
        - r2: similar to r1,
        - r3: similar to r1,
        - l1: Degree of contact on the inside of the Lgripper,
        - l2: similar to l1
        - l3: similar to l1
        """
        try:
            # Make replay buffer on GPU
            buffer = SimpleBuffer(try_num, buffer_size, rgb_shape, depth_shape, state_shape, robot_state_shape, device) 
            try_count = 0
            while (try_count != try_num):
                self.reset()
                # Get target obj state
                plate_pos = utils.get_point(self.plate) #Get target obj center position
                for i in range(data_length):
                    pb.stepSimulation()
                    #Get current img
                    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                            128,
                            128,
                            viewMatrix=self.viewMatrix)
 
                    depth = (depthImg*255).astype(np.uint8).reshape(128, 128, 1)
                    depth = np.tile(depth, (1,1,3))
                    rgb = rgbImg.reshape(128,128,4)[:,:,:3]
                    img = np.concatenate([rgb, depth], 2).astype(np.uint8)
                    img = torch.tensor(np.transpose(img, (2, 0, 1)).reshape(1,6,128,128)).float().to(device)
                                    
                    # Get current robot state
                    rx, ry, lx, ly, rtheta, ltheta, rw, lw = self.get_data()
                    r1 = len(pb.getContactPoints(2, 3, -1, 9))
                    r2 = len(pb.getContactPoints(2, 3, -1, 10))
                    r3 = len(pb.getContactPoints(2, 3, -1, 11))
                    l1 = len(pb.getContactPoints(2, 4, -1, 9))
                    l2 = len(pb.getContactPoints(2, 4, -1, 10))
                    l3 = len(pb.getContactPoints(2, 4, -1, 11))
                    robot_state = torch.tensor(np.array([rx, ry, lx, ly, rtheta, ltheta, rw, lw, r1, r2, r3, l1, l2, l3, plate_pos[0], plate_pos[1]]).reshape(1,16)).float().to(device)
                    
                    # Inference
                    encoded = self.ae(img, robot_state).to(device).reshape(1,1,8) #batch, time length, vector 
                    output = self.model(encoded.to(device)).to(device)
                    rx, ry, lx, ly, rtheta, ltheta, rw, lw = output[0,0,:].cpu().detach().numpy()
                    self.rgripper.set_state([rx, ry, 0]) #rgripper 
                    self.lgripper.set_state([lx, ly, 0]) #lgripperp
                    self.rgripper.set_pose([-1.54, rtheta, -1.57]) 
                    #self.lgripper.set_pose([-1.54, ltheta, -1.57])
                
                    self.frames.append(rgbImg)
                    self.d_frames.append(depthImg)
                    state = np.array([rx, ry, lx, ly, rtheta, ltheta, rw, lw])
                    robot_state = np.array([rx, ry, lx, ly, rtheta, ltheta, rw, lw, r1, r2, r3, l1, l2, l3, plate_pos[0], plate_pos[1]])
                    buffer.append(np.array(rgbImg).flatten(), np.array(depthImg).flatten(), state, robot_state)
                # Picking up
                self.rgripper.set_state([0.0, -0.5, 0.0]) 
                self.lgripper.set_state([0.0, -0.5, 0.0])
                self.rgripper.set_gripper_width(0) #Close gripper
                self.lgripper.set_gripper_width(0) #Close gripper
                contact_len = 0 #If gripper contact plate, contact_len increase
                for i in range(50):
                    pb.stepSimulation()
                    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                            width=128,
                            height=128,
                            viewMatrix=self.viewMatrix)
                    #self.frames.append(rgbImg) #not need for grasp learning dataset
                    #self.d_frames.append(depthImg) #not need for grasp learning dataset
                    contact_len += len(pb.getContactPoints(bodyA=1, bodyB=2)) #Judge if plate and table collision
                    contact_len += len(pb.getContactPoints(bodyA=0, bodyB=2)) #Judge if plate and table collision
                
                if contact_len > 1: #Judge if gripper contact plate
                    buffer.delete()
                    print("Failed!!!")
                else:
                    try_count += 1 
                    print("Succeeded!!!")
            
            now = datetime.datetime.now()  
            video_name = "test_" + str(try_num) + "_length_" + str(data_length) + now.strftime('_%Y%m%d_%H%M%S')+ ".mp4"
            save_video(self.frames, video_name)
            return buffer

        except KeyboardInterrupt:
            sys.exit()

if __name__ == '__main__':
    try:
        sim
    except:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--model_path', '-p', type=str, help='set model path', default='model.pth')
        args = parser.parse_args()
        model_path = (args.model_path)#Simulator loop times
        data_length = 50
        try_num = 1
        BUFFER_SIZE = try_num * data_length #10 ** 6
        rgb_shape = 128*128*4 
        depth_shape = 128*128 
        state_shape = 8
        robot_state_shape = 8+8
        sim = Simulator()
        sim.load_model(model_path)
        buffer = sim.rollout(data_length, try_num, BUFFER_SIZE, rgb_shape, depth_shape, state_shape, robot_state_shape)
        buffer.save()
        pb.disconnect()

