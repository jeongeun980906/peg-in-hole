import os
import time
import pdb
import pybullet as p
import pybullet_data
import utils_ur5_robotiq140
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt
from gym.spaces import Box


ACTION_RANGE = 1.0
OBJ_RANGE = 0.1
Deg2Rad = 3.141592/180.0

class UR5_robotiq():

    def __init__(self, args):

        # print(args.GUI)
        if args.GUI =='GUI':
            self.serverMode = p.GUI # GUI/DIRECT
        else:
            self.serverMode = p.DIRECT


        self.distance_threshold = args.dist_threshold
        self.reward_type = 'sparse'
        self.has_object = False
        # self.goal = np.array([0.13, -0.65, 1.02])

        self.loadURDF_all()
        self.setCamera()
        # p.setRealTimeSimulation(1)
        # p.setTimeStep(1.0/240.0)

        # environment check
        # while(True):
        #     pass

        # print(controlJoints)
        self.eefID = 7 # ee_link
        self._max_episode_steps = args.epi_step
        self.observation_space = 39     # Input size
        self.action_space = Box(-ACTION_RANGE, ACTION_RANGE, (4,))

        self.init_pose = [0.0*Deg2Rad,
                        -70.0*Deg2Rad,
                         90.0*Deg2Rad,
                        -90.0*Deg2Rad,
                        -90.0*Deg2Rad,
                         0.0*Deg2Rad] 
        # self.init_pose = [-0.15155565744358215, -0.9558248750266662, 1.2825613194480308, -1.89325623453974, -1.57109183395481, -0.15131148128391603]                               

        self.home_pose()
        p.stepSimulation()

        pose = self.getRobotPose()
        self.Orn = [pose[3], pose[4], pose[5], pose[6]]
        self.fOri=0
        # self.goal = np.array(self.targetStartPos)elf.d_oldelf.d_old
        


    def loadURDF_all(self):
        self.UR5UrdfPath = "./urdf/ur5_peg.urdf"
        # self.UR5UrdfPath = "./urdf/ur5_robotiq85.urdf"

        # connect to engine servers
        self.physicsClient = p.connect(self.serverMode)
        # add search path for loadURDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        # p.setTimeStep(0.1)

        # Gripper Setting
        self.gripper_main_control_joint_name = "robotiq_85_left_knuckle_joint"
        self.mimic_joint_name = ["robotiq_85_right_knuckle_joint",
                            "robotiq_85_left_inner_knuckle_joint",
                            "robotiq_85_right_inner_knuckle_joint",
                            "robotiq_85_left_finger_tip_joint",
                            "robotiq_85_right_finger_tip_joint"]
        self.mimic_multiplier = [1, 1, 1, -1, -1] 

        # Load URDF
        # define world
        self.planeID = p.loadURDF("plane.urdf")

        tableStartPos = [0.7, 0.0, 0.8]
        tableStartOrientation = p.getQuaternionFromEuler([0, 0, 90.0*Deg2Rad])
        self.tableID = p.loadURDF("./urdf/objects/table.urdf", tableStartPos, tableStartOrientation,useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        # define environment
        self.holePos = [0.6, 0.0, 0.87]
        self.holeOri = p.getQuaternionFromEuler([1.57079632679, 0, 0.261799333]) #.261799333
        
        self.boxId = p.loadURDF(
        "./urdf/peg_hole_gazebo/hole/urdf/hole.SLDPRT.urdf",
        self.holePos, self.holeOri,
        flags = p.URDF_USE_INERTIA_FROM_FILE)
        # p.changeDynamics(self.boxId ,-1,lateralFriction=0.5,spinningFriction=0.1)
      
        # self.holePos = [0.6, -0.1, 0.67]
        # self.holeOri = p.getQuaternionFromEuler([1.57079632679, 0, 0.261799333]) #.261799333
        
        # define environment
        # ur5standStartPos = [0.0, 0.0, 0.0]
        # ur5standStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        # self.ur5_standID = p.loadURDF("./urdf/objects/ur5_stand.urdf", ur5standStartPos, ur5standStartOrientation,useFixedBase = True)

        # setup ur5 with robotiq 85
        robotStartPos = [0.0, 0.0, 0.0]
        robotStartOrn = p.getQuaternionFromEuler([0,0,0])

        print("----------------------------------------")
        print("Loading robot from {}".format(self.UR5UrdfPath))        
        self.robotID = p.loadURDF(self.UR5UrdfPath, robotStartPos, robotStartOrn,useFixedBase = True)
                             # flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlJoints = utils_ur5_robotiq140.setup_sisbot(p, self.robotID)

    def step(self, action):


        self.moveL(action)   #move > moveL

        self.next_state_dict = self.get_state()

        
        achieved_goal = self.getRobotPoseE()
        current_pos = []
        for i in range(3):
            current_pos.append(achieved_goal[i])
        current_pos.append(achieved_goal[5])
        # print(achieved_goal[3:5])
        # print('===========')
        
        goal = []
        for i in range(3):
            goal.append(self.holePos[i])

        goal_ori = []
        for i in range(4):
            goal_ori.append(self.holeOri[i])
        goal_ori = p.getEulerFromQuaternion(goal_ori)
        goal.append(goal_ori[2])

        a = np.array(goal)
        b = np.array(current_pos)
        
        #p.addUserDebugText('.',np.array(self.holePos)[:3])
        d = np.linalg.norm(a - b, axis=-1)

        self.done = (d <= self.distance_threshold).astype(np.float32)

        reward = self.compute_reward2(b)
        
        return self.next_state_dict, reward, self.done, {}


    def reset(self):
        # while True:
        #     self.pos = np.random.uniform(-OBJ_RANGE+0.03,OBJ_RANGE+0.03,3)
        #     self.pos[1] -= 0.77
        #     self.pos[2] = 0.85
        #     self.pos2 = np.random.uniform(-OBJ_RANGE+0.03,OBJ_RANGE+0.03,3)
        #     self.pos2[1] -= 0.77
        #     self.pos2[2] += 0.92
        #     self.goal = np.array(self.pos2)
        #     if(self.goal_distance(self.pos, self.pos2) > 0.1):
        #         break
        # self.dist = 0
        self.gripper_length = 0.00374
        self.home_pose()
        
        self.state_dict = self.get_state()
        # # p.removeAllUserDebugItems()
        return self.state_dict

    def home_pose(self):

        for i, name in enumerate(self.controlJoints):
            if i > 6:
                break

            # print(i)
            self.joint = self.joints[name]

            pose1 = self.init_pose[i]
            if i < 6:
                p.resetJointState(self.robotID, self.joint.id, targetValue=pose1, targetVelocity=0)
            # else:
            #     self.gripper_opening_angle = self.Angle_Cal(pose1)  # angle calculation  action[6]
            #     p.resetJointState(self.robotID,  self.joints[self.gripper_main_control_joint_name].id, targetValue=self.gripper_opening_angle, targetVelocity=0)
                
            #     for i in range(len(self.mimic_joint_name)):
            #         joint = self.joints[self.mimic_joint_name[i]]
            #         p.resetJointState(self.robotID, joint.id, targetValue=self.gripper_opening_angle * self.mimic_multiplier[i], targetVelocity=0)
        
        # p.resetBasePositionAndOrientation(self.boxId, self.holePos, self.holeOri)
       
        p.stepSimulation()
        #p.addUserDebugText('.',p.getLinkState(self.robotID, 7)[4][:3])
    def moveL(self, targetPose, setTime = 0.1):
        stepSize = 240*setTime
        currentPose = self.getRobotPoseE()
        delta = []
        for i in range(len(currentPose)):
            delta.append((targetPose[i] - currentPose[i])/stepSize)
            
        # # p.removeAllUserDebugItems()
        #print("Delta")
        # print(delta)

        start = time.time()
        for t in range(int(stepSize)):
            stepPos = []
            stepOri = []
            for i in range(3):
                stepPos.append(currentPose[i] + (t+1)*delta[i])
            stepOri.append(0)
            stepOri.append(1.5707963)
            stepOri.append(currentPose[5] + (t+1)*delta[5])
            stepOri = p.getQuaternionFromEuler(stepOri)
        for t in range(int(stepSize)):
            
            jointPos = p.calculateInverseKinematics(self.robotID,
                                                    self.eefID,
                                                    stepPos,
                                                    stepOri)
            # pos.append(1)
            # jointPos = urk.invKine(pos)
            # print('---------------')
            # print(jointPos)
            for i, name in enumerate(self.controlJoints):
                joint = self.joints[name]
                targetJointPos = jointPos[i]

                p.setJointMotorControl2(self.robotID,
                                        joint.id,
                                        p.POSITION_CONTROL,
                                        targetPosition = targetJointPos,
                                        # targetVelocity = 5,
                                        force = joint.maxForce, 
                                        maxVelocity = joint.maxVelocity)

            # p.addUserDebugLine((0.6475237011909485, 0.6443161964416504, 0.9296525716781616),(0,0,0))
        # for i in range(10):
            
            p.stepSimulation()
            # self.getCameraImage()
        
        currentOri1 = p.getLinkState(self.robotID, 6)[5]#5
        currentOri1 = p.getEulerFromQuaternion(currentOri1)
        currentOri2 = p.getLinkState(self.robotID, 5)[5]
        currentOri2 = p.getEulerFromQuaternion(currentOri2)
        
        joint_states = p.getJointStates(self.robotID, range(p.getNumJoints(self.robotID)))
        
        joint_positions = [state[0] for state in joint_states]
        print(currentOri1,currentOri2,'\n',joint_positions)
        #print(joint_positions)
        end = time.time()

        # print(end-start)

    def move(self, action):

        ee_position = p.getLinkState(self.robotID, 7)[5]
        # l_gripper_states = p.getLinkState(self.robotID, 14)[0]
        # r_gripper_states = p.getLinkState(self.robotID, 16)[0]
        # cur_length = np.linalg.norm(np.array(l_gripper_states) - np.array(r_gripper_states), axis=-1)- 0.0156614

        # Joints states
        joint_states = p.getJointStates(self.robotID, range(p.getNumJoints(self.robotID)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        gripper_offset = 0.0156
        # print(cur_length-0.085)
        target_position=[]
        
        for i in range(3):
            target_position.append(ee_position[i]+action[i]) #0.5*action
        target_position[2] = max(target_position[2], 0.88)
        # target_position[1] = np.clip(target_position[1], -0.3,0.3)

        # Object states
        object_pos, object_ori = p.getBasePositionAndOrientation(self.boxId)
        self.object_rel_pos = []
        for i in range(3):
            self.object_rel_pos.append(object_pos[i] - ee_position[i])

        # target_length = cur_length + 0.05*action[3]

        # target_length = min(max(target_length,0.0),0.085)
        if action[2] < 0:   #3->2
            target_length =  0.085
        else:
            target_length =  0.045
        # target_length = 0.085


        jointPose = p.calculateInverseKinematics(self.robotID, self.eefID, target_position, self.Orn)
        for i, name in enumerate(self.controlJoints):
            self.joint = self.joints[name]
            
            if i == 12:
                break
            pose1 = jointPose[i]

            
            if i < 6:  
                # pose1 = joint_positions[self.joint.id] + 0.05*action[i]              
                p.setJointMotorControl2(self.robotID, self.joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose1, force=self.joint.maxForce, 
                                        maxVelocity=self.joint.maxVelocity)

        for i in range(10):
            p.stepSimulation()

    def getRobotPose(self):
        currentPos = p.getLinkState(self.robotID, 7)[4]#4
        currentOri = p.getLinkState(self.robotID, 7)[5]#5
        #currentOri = p.getEulerFromQuaternion(currentOri)
        currentPose = []
        currentPose.extend(currentPos)
        currentPose.extend(currentOri)
        return currentPose 

    def getRobotPoseE(self):
        currentPos = p.getLinkState(self.robotID, 7)[4]#4
        currentOri = p.getLinkState(self.robotID, 7)[5]#5
        currentOri = p.getEulerFromQuaternion(currentOri)
        currentPose = []
        currentPose.extend(currentPos)
        currentPose.extend(currentOri)
        return currentPose              

    def get_state(self):

        # Joints states
        joint_states = p.getJointStates(self.robotID, range(p.getNumJoints(self.robotID)))
        # >>>joint_states.shape :  (8, 4)
        # print("joint_states.shape : {}".foramt(np.array(joint_states).shape))
        joint_positions = [state[0] for state in joint_states] # (1,4) -> (1,1)
        joint_velocities = [state[1] for state in joint_states]
        
        state_list = []
        vel_list = []
        for i, name in enumerate(self.controlJoints):
            if i > 6:
                break

            
            state_list.append(joint_positions[self.joint.id])
            vel_list.append(joint_velocities[self.joint.id])

        joint_state = state_list + vel_list
        

        # End-Effector States
        ee_states = p.getLinkState(self.robotID, 7 , computeLinkVelocity = 1)
        ee_pos = ee_states[0]
        ee_ori = ee_states[1]
        ee_linear_vel = ee_states[6]
        ee_angular_vel = ee_states[7]
        ee_ori = p.getEulerFromQuaternion(ee_ori)
        # ee_state = ee_pos + ee_linear_vel + ee_angular_vel
        # 
        # Gripper States
        # l_gripper_states = p.getLinkState(self.robotID, 14 ,computeLinkVelocity = 1)
        # r_gripper_states = p.getLinkState(self.robotID, 16 ,computeLinkVelocity = 1)
        # l_gripper_pos = l_gripper_states[0]
        # r_gripper_pos = r_gripper_states[0]
        # l_gripper_vel = l_gripper_states[6]
        # r_gripper_vel = r_gripper_states[6]
        # gripper_state = l_gripper_pos + r_gripper_pos
        # gripper_vel = l_gripper_vel + r_gripper_vel
        # # print(gripper_vel)
        # gripper_length = np.linalg.norm(np.array(l_gripper_pos) - np.array(r_gripper_pos), axis=-1) - 0.0156614

        # print(gripper_length)

        # Object states
        object_pos = []
        for i in range(3):
            object_pos.append(self.holePos[i])
        #object_pos[2] += 0.3
        object_ori = []
        for i in range(4):
            object_ori.append(self.holeOri[i])

        object_ori = p.getEulerFromQuaternion(object_ori)
        
        #object_pos, object_ori = p.getBasePositionAndOrientation(self.boxId)
        #object_linear_vel, object_angular_vel = p.getBaseVelocity(self.boxId)
        #object_ori = p.getEulerFromQuaternion(object_ori)
        # print(asd)
        # print(object_ori)
        # print(tf.transformations.euler_from_quaternion(object_ori))

       # object_linear_vel = np.array(object_linear_vel) - np.array(ee_linear_vel)
        self.object_rel_pos = []
        goal_rel_pos = []
        obj_goal_rel_pos = []
        object_rel_vel = []
        for i in range(3):
            self.object_rel_pos.append(object_pos[i] - ee_pos[i])
            #object_rel_vel.append(object_linear_vel[i]-ee_linear_vel[i])
            #goal_rel_pos.append(self.goal[i] - ee_pos[i])
            #obj_goal_rel_pos.append(self.goal[i]-object_pos[i])

        # p.addUserDebugLine(l_gripper_pos,  r_gripper_pos, lineColorRGB=(255,0,255))
        # self.grasp = np.linalg.norm(np.array(ee_pos) - np.array(object_pos), axis=-1)
        self.contact = False
        self.contact1 = p.getContactPoints(bodyA=self.robotID,bodyB=self.boxId,linkIndexA = 14)
        self.contact2 = p.getContactPoints(bodyA=self.robotID,bodyB=self.boxId,linkIndexA = 16)

        if (self.contact1) and (self.contact2):
            self.contact = True
            print("CONTACT!!!")

        # Final states
        # obs = np.concatenate([
        #     ee_pos, ee_linear_vel, #gripper_state, gripper_vel,
        #     object_pos, object_ori, object_linear_vel, object_angular_vel,
        #     self.object_rel_pos, object_rel_vel,
        # ])self.object_rel_pos[0],self.object_rel_pos[1],self.object_rel_pos[2],, rel_ori
        
        rel_ori = object_ori[2] - ee_ori[2]
        obs = np.array([
            ee_pos[0], ee_pos[1], ee_pos[2], object_ori[2], ee_ori[2],
    
            self.holePos[0], self.holePos[1], self.holePos[2]
            ]) #말단과 hole의 x, y위치만 선정

        
        # print(obs.shape)
        # Final states
        # obs = np.concatenate([
            # ee_pos, ee_linear_vel, goal_rel_pos,
        # ])
        
        # self.object_pos = object_pos
        # if self.grasp < 0.05:
            # self.has_object = True
        # else:
            # self.has_object = False

        # if not self.has_object:
            # achieved_goal = np.array(ee_pos)
        # else: 
        ##achieved_goal = np.array(object_pos)
        #self.reward = self.compute_reward(achieved_goal, self.goal , self.object_rel_pos)
        #self.done = self._is_success(achieved_goal, self.goal)

        #for i,name in enumerate(ee_states):
        #    print(i, name)
        #return [obs.copy(),achieved_goal,self.goal.copy()]
        
        return obs
     
     #   return {
      #      'observation': obs.copy(),
       #     'achieved_goal': achieved_goal,
        #    'desired_goal': self.goal.copy(),
        #}


    def Angle_Cal(self, length):

        length = float(np.clip(length, 0, 0.085))
        angle = 0.715 - math.asin((length - 0.010) / 0.1143)    # angle calculation
        return angle

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def sample_action(self):
        random_action = np.random.uniform(-ACTION_RANGE,ACTION_RANGE,4)
        return random_action

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, rel_pos):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        # return -d

        reward = -(d > self.distance_threshold).astype(np.float32)

        # dist = math.sqrt(sum(map(lambda x:x*x,rel_pos)))
        # reward += -min(0.5, dist)

        # if self.contact:
            # reward += 0.5
        # print(reward)
        # if dist < 0.02:
            # reward += 0.5
        return reward

    def compute_reward2(self, achieved_goal):
        # Compute distance between goal and the achieved goal.
       # d = self.goal_distance(achieved_goal, goal)
        # return -d
        
        pose = []
        for i in range(3):
            pose.append(self.holePos[i])
        object_ori = []
        for i in range(4):
            object_ori.append(self.holeOri[i])

        object_ori = p.getEulerFromQuaternion(object_ori)
        pose.append(object_ori[2])

        #pose[2] += 0.3

        a = np.array(pose)
       
        d_pos = np.linalg.norm(a[:3] - achieved_goal[:3], axis=-1)
        d_ori = np.linalg.norm(a[3] - achieved_goal[3])

        d = 0.7*d_pos + 0.3*d_ori
        #reward =  100*(self.d_old - d)

######################################            
#람다*(x,y,z) + (1-람다)*(z_ori)
######################################

        reward = -d

        # ee_velocity = p.getLinkState(self.robotID, 7, 1)[6]
        # if self.done and ee_velocity[0] == 0 and ee_velocity[1] == 0 and ee_velocity[2] == 0:
        #     reward = 1
        #     self.real_done += 1
        if self.done:
            reward = 1
        # if d > self.d_old:
        #     reward = -1
        # elif d < self.d_old:
        #     reward = +1
        # else:
        #     reward = 0

        #reward = -(d > self.distance_threshold).astype(np.float32)
        
        return reward

    def create_state_from_observation_and_desired_goal(self, observation, desired_goal):
        return np.concatenate((observation, desired_goal))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def setCamera(self):
        self.viewMatrix1 = p.computeViewMatrix(
        cameraEyePosition = [self.holePos[0]+0.2, self.holePos[1], 1.2],
        cameraTargetPosition = self.holePos,
        cameraUpVector = [-1, 0, 0])

        self.projectionMatrix1 = p.computeProjectionMatrixFOV(
        fov = 45.0,
        aspect = 640.0/480.0,
        nearVal = 0.1,
        farVal = 3.1)

    def getCameraImage(self):
        # Camera 1        
        image1 = p.getCameraImage(
        width = 640, 
        height = 480,
        viewMatrix = self.viewMatrix1,
        projectionMatrix = self.projectionMatrix1)

        # width, height, rgbImg, depthImg, segImg
        return image1

        