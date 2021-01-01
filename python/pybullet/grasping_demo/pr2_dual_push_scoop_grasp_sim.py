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
        - Right and Left gripper pos, rot, width

        """
        rgripper_pos = utils.get_point(self.rgripper)
        lgripper_pos = utils.get_point(self.lgripper)

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
        """
        Currently, 
        If plate is on the right side, grasping tactics is rotational grasping.
        If plate is on the left side, grasping tactics is pushing and grasping.
        """
        if(plate_pos[1] < 0):
            tactics = 0
        else:
            tactics = 1
        for i in range(100):
            pb.stepSimulation()
        return tactics

    def rollout(self, data_length, try_num, buffer_size, rgb_shape, depth_shape, state_shape, device=torch.device('cuda')):
        """
        State:
        rx, ry, lx, ly, rtheta, ltheta, rw, lw
        """
        try:
            # Make replay buffer on GPU
            buffer = SimpleBuffer(try_num, buffer_size, rgb_shape, depth_shape, state_shape, device) 
            try_count = 0
            while (try_count != try_num):
                tactics = self.reset()
                # Get target obj state
                plate_pos = utils.get_point(self.plate) #Get target obj center position
                if tactics:
                    print("Rotational Grasping!!!")
                    # Approaching and close right gripper
                    pitch = np.random.rand() * pi / 8 + pi / 8 
                    self.rgripper.set_state([-0.3, 0.5, 0]) #Success position
                    self.lgripper.set_state([-0.2, 0.1, 0]) #Success position
                    for i in range(data_length):
                        pb.stepSimulation()
                        rw = 1
                        lw = 1
                        lx = -0.2
                        ly = 0.1
                        rtheta = 0
                        ltheta = 0
                        if len(pb.getContactPoints(bodyA=1, bodyB=3)): #Contact Rgripper and table
                            rx = 0.3-i 
                            ry = 0.5+i
                            lx = -0.2-0.01*i
                            ly = 0.1-0.02*i
                            self.rgripper.set_state([rx, ry, 0]) #rgripper up
                            self.lgripper.set_state([lx, ly, 0]) #lgripper up
                        else:
                            rx = 0.3+i 
                            ry = 0.5+i
                            lx = -0.2+0.01*i
                            ly = 0.1-0.02*i
                            self.rgripper.set_state([rx, ry, 0]) #rgripper down
                            self.lgripper.set_state([lx, ly, 0]) #lgripper up
                        if len(pb.getContactPoints(bodyA=2, bodyB=3)): #Contact Rgripper and plate
                            rtheta = pitch
                            rw = 0
                            self.rgripper.set_pose([-1.54, pitch, -1.57]) #Random Scooping
                            #self.rgripper.set_pose([-1.54, 0.8, -1.57]) #Success Scooping
                            self.rgripper.set_gripper_width(rw) #Close gripper
                        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                                128,
                                128,
                                viewMatrix=self.viewMatrix)
                        self.frames.append(rgbImg)
                        self.d_frames.append(depthImg)
                        state = np.array([rx, ry, lx, ly, rtheta, ltheta, rw, lw])
                        buffer.append(np.array(rgbImg).flatten(), np.array(depthImg).flatten(), state)
                        #buffer.append(state, action, reward, mask, next_state)

                else:
                    print("Moving Grasping!!!")
                    pitch = np.random.rand() * pi / 8 + pi / 8 
                    rtheta = 0
                    ltheta = 0
                    self.rgripper.set_state([-0.3, 0.5, 0]) #Success position
                    self.lgripper.set_state([-0.2, 0.1, 0]) #Success position
                    for i in range(data_length):
                        pb.stepSimulation()
                        plate_pos = utils.get_point(self.plate) #Get target obj center position
                        if plate_pos[1] > -0.5:
                            rw = 1
                            lw = 1
                            if len(pb.getContactPoints(bodyA=1, bodyB=3)): #Contact Rgripper and table
                                rx = 0.3-i*0.05
                                ry = 0.5+i*0.05
                                lx = -0.2-i*0.01
                                ly = 0.1-i*0.02
                                self.rgripper.set_state([rx, ry, 0]) #rgripper up
                                self.lgripper.set_state([lx, ly, 0]) #lgripper up
                            else:
                                rx = 0.3+i*0.05
                                ry = 0.5+i*0.05
                                lx = -0.2+i*0.01
                                ly = 0.1-i*0.02
                                self.rgripper.set_state([rx, ry, 0]) #rgripper down
                                self.lgripper.set_state([lx, ly, 0]) #lgripper down
                        elif(len(pb.getContactPoints(bodyA=2, bodyB=4))): #Contact Rgripper and plate
                            lw = 0
                            rx = -0.2+i*0.01
                            ry = 0.1+i*0.02
                            lx = -i*0.01 
                            ly = -0.6-i*0.03
                            self.lgripper.set_gripper_width(lw) #close gripper
                            self.lgripper.set_state([rx, ry, 0]) #lgripper move left
                            self.rgripper.set_state([lx, ly, 0]) #rgripper move right
   
                        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                                128,
                                128,
                                viewMatrix=self.viewMatrix)
                        self.frames.append(rgbImg)
                        self.d_frames.append(depthImg)
                        state = np.array([rx, ry, lx, ly, rtheta, ltheta, rw, lw])
                        buffer.append(np.array(rgbImg).flatten(), np.array(depthImg).flatten(), state)
                        #buffer.append(state, action, reward, mask, next_state)
                        
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
                    #time.sleep(0.005)
                    contact_len += len(pb.getContactPoints(bodyA=1, bodyB=2)) #Judge if plate and table collision
                    contact_len += len(pb.getContactPoints(bodyA=0, bodyB=2)) #Judge if plate and table collision
                
                if contact_len > 1: #Judge if gripper contact plate
                    buffer.delete()
                    print("Failed!!!")
                else:
                    try_count += 1 
                    print("Succeeded!!!")
            
            now = datetime.datetime.now()  
            video_name = "try_" + str(try_num) + "_length_" + str(data_length) + now.strftime('_%Y%m%d_%H%M%S')+ ".mp4"
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
        parser.add_argument('--try_num', '-n', type=int, help='set trial number', default='3')
        args = parser.parse_args()
        try_num = (args.try_num)#Simulator loop times
        data_length = 50
        BUFFER_SIZE = try_num * data_length #10 ** 6
        rgb_shape = 128*128*4 
        depth_shape = 128*128 
        state_shape = 8
        sim = Simulator()
        buffer = sim.rollout(data_length, try_num, BUFFER_SIZE, rgb_shape, depth_shape, state_shape)
        buffer.save()
        pb.disconnect()

