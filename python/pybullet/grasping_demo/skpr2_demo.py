import skrobot
from skrobot.models import PR2
from skrobot.models import Fetch
from skrobot.interfaces import PybulletRobotInterface
import pybullet as pb
import pybullet_data
import numpy as np
import utils
import time
from math import *



class Sub_PR2(PR2):
    def __init__(self, *args, **kwargs):
        super(Sub_PR2, self).__init__(*args, **kwargs)
    def init_pose(self):
        #self.l_shoulder_pan_joint.joint_angle.joint_angle(np.deg2rad(299.865))
        self.l_shoulder_pan_joint.joint_angle.joint_angle(np.deg2rad(121.865))
        self.l_shoulder_lift_joint(np.deg2rad(55.4297))
        self.l_upper_arm_roll_joint.joint_angle(np.deg2rad(53.3196))
        #self.l_elbow_flex_joint.joint_angle(np.deg2rad(102.662))
        self.l_elbow_flex_joint.joint_angle(np.deg2rad(-10))
        self.l_forearm_roll_joint.joint_angle(np.deg2rad(-121.542))
        #self.l_wrist_flex_joint.joint_angle(np.deg2rad(125.071))
        self.l_wrist_flex_joint.joint_angle(np.deg2rad(-10.071))
        self.l_wrist_roll_joint.joint_angle(np.deg2rad(-87.0418))
        self.r_shoulder_pan_joint.joint_angle(np.deg2rad(-48.2131))
        self.r_shoulder_lift_joint.joint_angle(np.deg2rad(-32.0168))
        self.r_upper_arm_roll_joint.joint_angle(np.deg2rad(-20.2598))
        #self.r_elbow_flex_joint.joint_angle(np.deg2rad(-67.6931))
        self.r_elbow_flex_joint.joint_angle(np.deg2rad(-67.6931))
        self.r_forearm_roll_joint.joint_angle(np.deg2rad(-45.3044))
        #self.r_wrist_flex_joint.joint_angle(np.deg2rad(-72.9084))
        self.r_wrist_flex_joint.joint_angle(np.deg2rad(-72.9084))
        self.r_wrist_roll_joint.joint_angle(np.deg2rad(-96.2568))
        #self.torso_lift_joint.joint_angle(np.deg2rad(-100.018))
        self.torso_lift_joint.joint_angle(np.deg2rad(100.018))
        self.head_pan_joint.joint_angle(np.deg2rad(4.1047))
        self.head_tilt_joint.joint_angle(np.deg2rad(54.75))
        """
        self.torso_lift_joint.joint_angle(np.deg2rad(299.865))
        self.l_shoulder_pan_joint.joint_angle(np.deg2rad(55.4297))
        self.l_shoulder_lift_joint.joint_angle(np.deg2rad(53.3196))
        self.l_upper_arm_roll_joint.joint_angle(np.deg2rad(102.662))
        self.l_elbow_flex_joint.joint_angle(np.deg2rad(-121.542))
        self.l_forearm_roll_joint.joint_angle(np.deg2rad(125.071))
        self.l_wrist_flex_joint.joint_angle(np.deg2rad(-87.0418))
        self.l_wrist_roll_joint.joint_angle(np.deg2rad(-48.2131))
        self.r_shoulder_pan_joint.joint_angle(np.deg2rad(-32.0168))
        self.r_shoulder_lift_joint.joint_angle(np.deg2rad(-20.2598))
        self.r_upper_arm_roll_joint.joint_angle(np.deg2rad(-67.6931))
        self.r_elbow_flex_joint.joint_angle(np.deg2rad(-45.3044))
        self.r_forearm_roll_joint.joint_angle(np.deg2rad(-72.9084))
        self.r_wrist_flex_joint.joint_angle(np.deg2rad(-96.2568))
        self.r_wrist_roll_joint.joint_angle(np.deg2rad(-100.018))
        self.head_pan_joint.joint_angle(np.deg2rad(4.1047))
        self.head_tilt_joint.joint_angle(np.deg2rad(54.75))
        """
        return self.angle_vector()
#robot_model.angle_vector([299.865, 55.4297, 53.3196, 102.662, -121.542, 125.071, -87.0418, -48.2131, -32.0168, -20.2598, -67.6931, -45.3044, -72.9084, -96.2568, -100.018, 4.1047, 54.75])
    def reset_pose_2(self):
        self.torso_lift_joint.joint_angle(0.05)
        self.l_shoulder_pan_joint.joint_angle(np.deg2rad(60))
        self.l_shoulder_lift_joint.joint_angle(np.deg2rad(74))
        self.l_upper_arm_roll_joint.joint_angle(np.deg2rad(70))
        self.l_elbow_flex_joint.joint_angle(np.deg2rad(-120))
        self.l_forearm_roll_joint.joint_angle(np.deg2rad(20))
        self.l_wrist_flex_joint.joint_angle(np.deg2rad(-30))
        self.l_wrist_roll_joint.joint_angle(np.deg2rad(180))
        self.r_shoulder_pan_joint.joint_angle(np.deg2rad(-60))
        self.r_shoulder_lift_joint.joint_angle(np.deg2rad(74))
        self.r_upper_arm_roll_joint.joint_angle(np.deg2rad(-70))
        self.r_elbow_flex_joint.joint_angle(np.deg2rad(-120))
        self.r_forearm_roll_joint.joint_angle(np.deg2rad(-20))
        self.r_wrist_flex_joint.joint_angle(np.deg2rad(-30))
        self.r_wrist_roll_joint.joint_angle(np.deg2rad(180))
        self.head_pan_joint.joint_angle(0)
        self.head_tilt_joint.joint_angle(0)
        return self.angle_vector()


def set_basepose(self, pos, rpy):
    utils.set_6dpose(self.gripper, pos, rpy)

# initialize robot
client_id = pb.connect(pb.GUI)
pb.setGravity(0,0,-9.8)
robot_model = PR2()
#pr2 = pb.loadURDF(robot_model.urdf_path)
pb.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=45,
    cameraPitch=-45,
    cameraTargetPosition=(0, 0, 0.5),
)
pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
table = pb.loadURDF("table/table.urdf")
plane = pb.loadURDF("plane.urdf")
"""
robot_pos = np.array([0.0, 0.8, 0.0])
utils.set_point(pr2, robot_pos)
utils.set_zrot(pr2, pi*0.5*3)
"""

#interface = PybulletRobotInterface(robot_model, connect=client_id)
interface = PybulletRobotInterface(robot_model)
robot_pos = np.array([0.0, 1.0, 0.0])
utils.set_point(interface.robot_id, robot_pos)
utils.set_zrot(interface.robot_id, pi*0.5*3)
print('==> Initialized PR2 Robot on PyBullet')
for _ in range(100):
    pb.stepSimulation()
time.sleep(3)

#reset pose
print('==> Moving to Reset Pose')
#robot_model.reset_manip_pose()
#print(robot_model.init_pose())
#print(robot_model.reset_manip_pose())
#Sub_PR2.init_pose

#robot_model.angle_vector([299.865, 55.4297, 53.3196, 102.662, -121.542, 125.071, -87.0418, -48.2131, -32.0168, -20.2598, -67.6931, -45.3044, -72.9084, -96.2568, -100.018, 4.1047, 54.75])
#interface.angle_vector(robot_model.angle_vector(), realtime_simulation=True)
#interface.wait_interpolation()

#interface.angle_vector(robot_model.angle_vector([299.865, 55.4297, 53.3196, 102.662, -121.542, 125.071, -87.0418, -48.2131, -32.0168, -20.2598, -67.6931, -45.3044, -72.9084, -96.2568, -100.018, 4.1047, 54.75]), realtime_simulation=True)
#interface.wait_interpolation()
#interface.angle_vector(robot_model.reset_manip_pose())

"""
# ik
print('==> Solving Inverse Kinematics')
target_coords = skrobot.coordinates.Coordinates(
    pos=[-0.3, 0.4, 0.5]
).rotate(np.pi / 2.0, 'y', 'local')
skrobot.interfaces.pybullet.draw(target_coords)
robot_model.inverse_kinematics(
    target_coords,
    link_list=robot_model.rarm.link_list,
    move_target=robot_model.rarm_end_coords,
    rotation_axis=True,
    stop=1000,
)
interface.angle_vector(robot_model.angle_vector(), realtime_simulation=True)
interface.wait_interpolation()

# wait
while pb.isConnected():
    time.sleep(0.01)

pb.disconnect()
"""
