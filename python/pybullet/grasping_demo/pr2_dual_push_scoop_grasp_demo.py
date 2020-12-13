#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from math import *
import pybullet 
import pybullet as pb
import pybullet_data
import numpy as np
import time
import six
import utils
from grasp_utils import *

class RGripper():
    def __init__(self):

        def construct_tables(robot):
            # construct tables
            # table (joint_name -> id) (link_name -> id)
            link_table = {pb.getBodyInfo(robot)[0].decode('UTF-8'):-1}
            joint_table = {}
            heck = lambda path: "_".join(path.split("/"))
            for _id in range(pb.getNumJoints(robot)):
                joint_info = pb.getJointInfo(robot, _id)
                joint_id = joint_info[0]

                joint_name = joint_info[1].decode('UTF-8')
                joint_table[joint_name] = joint_id
                name_ = joint_info[12].decode('UTF-8')
                name = heck(name_)
                link_table[name] = _id
            return link_table, joint_table

        self.gripper = pb.loadURDF("pr2_description/urdf/pr2_rgripper.urdf", useFixedBase=True)
        print("Rgripper ID", self.gripper)
        link_table, joint_table = construct_tables(self.gripper)
        self.link_table = link_table
        self.joint_table = joint_table

    def get_state(self):
        angle, velocity, _, _ = pb.getJointState(self.gripper, 0)
        return angle, velocity

    def set_gripper_width(self, angle, force=False):
        """
        if force, angle is set regardless of the physics
        """
        if force:
            for i in [8,9,10,11]: 
                pb.resetJointState(self.gripper, i, angle,
                        targetVelocity = 0.0)
        else:
            for i in [8,9,10,11]: 
                pb.setJointMotorControl2(self.gripper, i, 
                        pb.POSITION_CONTROL, targetPosition=angle, force=300)

    def set_state(self, state):
        for i in range(3):
            pb.setJointMotorControl2(self.gripper, i, 
                    pb.POSITION_CONTROL, targetPosition=state[i], force=300)

    def set_basepose(self, pos, rpy):
        utils.set_6dpose(self.gripper, pos, rpy)

    def set_pose(self, rpy):
        quat = utils.quat_from_euler(rpy)
        utils.set_quat(self.gripper, quat)

    def set_angle(self, angle, force=False):
        """
        if force, angle is set regardless of the physics
        """
        if force:
            for i in [8,9,10,11]:
                pb.resetJointState(self.gripper, i, angle,
                        targetVelocity = 0.0)
        else:
            for i in [8,9,10,11]:
                pb.setJointMotorControl2(self.gripper, i, 
                        pb.POSITION_CONTROL, targetPosition=angle, force=200)
class LGripper():
    def __init__(self):

        def construct_tables(robot):
            # construct tables
            # table (joint_name -> id) (link_name -> id)
            link_table = {pb.getBodyInfo(robot)[0].decode('UTF-8'):-1}
            joint_table = {}
            heck = lambda path: "_".join(path.split("/"))
            for _id in range(pb.getNumJoints(robot)):
                joint_info = pb.getJointInfo(robot, _id)
                joint_id = joint_info[0]

                joint_name = joint_info[1].decode('UTF-8')
                joint_table[joint_name] = joint_id
                name_ = joint_info[12].decode('UTF-8')
                name = heck(name_)
                link_table[name] = _id
            return link_table, joint_table

        self.gripper = pb.loadURDF("pr2_description/urdf/pr2_lgripper.urdf", useFixedBase=True)
        print("Lgripper ID", self.gripper)
        link_table, joint_table = construct_tables(self.gripper)
        self.link_table = link_table
        self.joint_table = joint_table

    def get_state(self):
        angle, velocity, _, _ = pb.getJointState(self.gripper, 0)
        return angle, velocity

    def set_gripper_width(self, angle, force=False):
        """
        if force, angle is set regardless of the physics
        """
        if force:
            for i in [8,9,10,11]:
                pb.resetJointState(self.gripper, i, angle,
                        targetVelocity = 0.0)
        else:
            for i in [8,9,10,11]:     
                pb.setJointMotorControl2(self.gripper, i, 
                        pb.POSITION_CONTROL, targetPosition=angle, force=300)

    def set_state(self, state):
        for i in range(3):
            pb.setJointMotorControl2(self.gripper, i, 
                    pb.POSITION_CONTROL, targetPosition=state[i], force=300)

    def set_basepose(self, pos, rpy):
        utils.set_6dpose(self.gripper, pos, rpy)

    def set_pose(self, rpy):
        quat = utils.quat_from_euler(rpy)
        utils.set_quat(self.gripper, quat)

    def set_angle(self, angle, force=False):
        """
        if force, angle is set regardless of the physics
        """
        if force:
            for i in [8,9,10,11]:
                pb.resetJointState(self.gripper, i, angle,
                        targetVelocity = 0.0)
        else:
            for i in [8,9,10,11]:
                pb.setJointMotorControl2(self.gripper, i, 
                        pb.POSITION_CONTROL, targetPosition=angle, force=200)

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
        self.try_num = 100 #Simulator loop times
        self.frames = [] #Movie buffer
        pb.resetDebugVisualizerCamera(2.0, 90, -0.0, (0.0, 0.0, 1.0))
        
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

    def rollout(self):
        try:
            for try_count in six.moves.range(self.try_num):
                tactics = self.reset()
                # Get target obj state
                plate_pos = utils.get_point(self.plate) #Get target obj center position
                if tactics:
                    print("Rotational Grasping!!!")
                    # Approaching and close right gripper
                    pitch = np.random.rand() * pi / 8 + pi / 8 
                    self.rgripper.set_state([-0.3, 0.5, 0]) #Success position
                    self.lgripper.set_state([-0.2, 0.1, 0]) #Success position
                    for i in range(40):
                        pb.stepSimulation()
                        
                        if len(pb.getContactPoints(bodyA=1, bodyB=3)): #Contact Rgripper and table
                            self.rgripper.set_state([0.3-i, 0.5+i, 0]) #rgripper up
                        else:
                            self.rgripper.set_state([0.3+i, 0.5+i, 0]) #rgripper down
                        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                                360,
                                240,
                                viewMatrix=self.viewMatrix)
                        self.frames.append(rgbImg)
                        if len(pb.getContactPoints(bodyA=2, bodyB=3)): #Contact Rgripper and plate
                            self.rgripper.set_pose([-1.54, pitch, -1.57]) #Random Scooping
                            #self.rgripper.set_pose([-1.54, 0.8, -1.57]) #Success Scooping
                            self.rgripper.set_gripper_width(0.0) #Close gripper

                else:
                    print("Moving Grasping!!!")
                    pitch = np.random.rand() * pi / 8 + pi / 8 
                    self.rgripper.set_state([-0.3, 0.5, 0]) #Success position
                    self.lgripper.set_state([-0.2, 0.1, 0]) #Success position
                    for i in range(100):
                        pb.stepSimulation()
                        plate_pos = utils.get_point(self.plate) #Get target obj center position
                        if plate_pos[1] > -0.5:
                            if len(pb.getContactPoints(bodyA=1, bodyB=3)): #Contact Rgripper and table
                                self.rgripper.set_state([0.3-i*0.05, 0.5+i*0.05, 0]) #rgripper up
                                self.lgripper.set_state([-0.2-i*0.01, 0.1-i*0.02, 0]) #lgripper up
                            else:
                                self.rgripper.set_state([0.3+i*0.05, 0.5+i*0.05, 0]) #rgripper down
                                self.lgripper.set_state([-0.2+i*0.01, 0.1-i*0.02, 0]) #lgripper down
                        elif(len(pb.getContactPoints(bodyA=2, bodyB=4))): #Contact Rgripper and plate
                            self.lgripper.set_gripper_width(0.0) #close gripper
                            self.lgripper.set_state([-0.2+i*0.01, 0.1+i*0.02, 0]) #lgripper move left
                            self.rgripper.set_state([-i*0.01, -0.6-i*0.03, 0]) #rgripper move right
   
                        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                                360,
                                240,
                                viewMatrix=self.viewMatrix)
                        self.frames.append(rgbImg)
                        
                # Picking up
                self.rgripper.set_state([0.0, -0.5, 0.0]) 
                self.lgripper.set_state([0.0, -0.5, 0.0])
                contact_len = 0 #If gripper contact plate, contact_len increase
                for i in range(50):
                    pb.stepSimulation()
                    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                            width=360,
                            height=240,
                            viewMatrix=self.viewMatrix)
                    self.frames.append(rgbImg)
                    #time.sleep(0.005)
                    contact_len += len(pb.getContactPoints(bodyA=1, bodyB=2)) #Judge if plate and table collision
                    contact_len += len(pb.getContactPoints(bodyA=0, bodyB=2)) #Judge if plate and table collision
                
                if contact_len > 1: #Judge if gripper contact plate
                    print("Failed!!!")
                else:
                    print("Succeeded!!!")
            save_video(self.frames, "sample.mp4")

        except KeyboardInterrupt:
            sys.exit()

if __name__ == '__main__':
    try:
        sim
    except:
        sim = Simulator()
        sim.rollout()
        pb.disconnect()

