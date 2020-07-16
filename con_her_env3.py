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

        self.thtorque=0.01
        self.initdis=0
        self.distance_threshold = 0.01
        self.reward_type = 'sparse'
        self.has_object = False
        # self.goal = np.array([0.13, -0.65, 1.02])

        self.loadURDF_all()
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
                       -60.0*Deg2Rad,
                        60.0*Deg2Rad,
                        -90.0*Deg2Rad,
                       -90.0*Deg2Rad,
                         0.0*Deg2Rad]                              
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
        flags = p.URDF_USE_INERTIA_FROM_FILE,useFixedBase=True)
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
        self.robotID = p.loadURDF(self.UR5UrdfPath, robotStartPos, robotStartOrn,useFixedBase = True,flags=p.URDF_USE_INERTIA_FROM_FILE)
                             # flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlJoints = utils_ur5_robotiq140.setup_sisbot(p, self.robotID)
        for i in range(p.getNumJoints(self.robotID)):
            p.enableJointForceTorqueSensor(self.robotID,i)

    def step(self, action):

        self.move(action)   #move > moveL

        self.next_state_dict = self.get_state()
        rel_pose=np.asarray([self.next_state_dict[0]-self.next_state_dict[23],self.next_state_dict[1]-self.next_state_dict[24],self.next_state_dict[2]-self.next_state_dict[25]])
        rel_ori=np.asarray([self.next_state_dict[3]-self.next_state_dict[26],self.next_state_dict[4]-self.next_state_dict[27],self.next_state_dict[5]-self.next_state_dict[28]])
        torque=[]
        for i in range(6):
            torque.append(self.next_state_dict[i+9])
        torque=np.asarray(torque)
        dis_error=np.linalg.norm(rel_pose, axis=-1, ord=2)
        ori_error=np.linalg.norm(rel_ori, axis=-1, ord=2)
        goal=False
        info=True
        self.done = self.contact
        #reward=-1
        if (dis_error<self.distance_threshold) and (ori_error<0.1):
            goal=True
        if ((dis_error)/self.initdis)>1.5:
            self.done=True
            info=False
            print('outofrange')
        reward=-0.5*(dis_error)/self.initdis
        reward-=0.5*(ori_error)
        #print(ori_error,self.next_state_dict[3:6])
        #if reward<-1:
         #   reward=-1
        #if reward>-0.5:
         #   reward=-0.5
        if self.done:
            self.contact=False
            if goal:
                reward=0.0
            #else:
             #   reward=-1.0
        return self.next_state_dict, reward, self.done, info


    def reset(self):
        print('reset')
        self.home_pose()
        self.move_init()
        self.state_dict = self.get_state()
        temp=np.asarray([self.state_dict[0]-self.state_dict[23],self.state_dict[1]-self.state_dict[24],self.state_dict[2]-self.state_dict[25]])
        self.initdis=np.linalg.norm(temp ,axis=-1, ord=2)
        #print(self.state_dict)
        # # p.removeAllUserDebugItems()
        time.sleep(1)
        return self.state_dict

    def home_pose(self):

        for i, name in enumerate(self.controlJoints):
            if i > 6:
                break
            # print(i()
            self.joint = self.joints[name]

            pose1 = self.init_pose[i]
            #p.resetJointState(self.robotID, self.joint.id, targetValue=pose1, targetVelocity=0)
            if i < 6:
                p.resetJointState(self.robotID, self.joint.id, targetValue=pose1, targetVelocity=0)
       
        p.stepSimulation()

    def move_init(self):
        target=[0.3+0.03*(np.random.rand()-0.5),0.03*(np.random.rand()-0.5),0.78+0.00*(np.random.rand()-0.5),0.0,0.0,0.0]
        self.moveL(target)
        pose=self.getRobotPose()
        fOri = p.getLinkState(self.robotID, 6)[5]
        self.fOri = p.getEulerFromQuaternion(fOri)
        print('?????',pose)

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
        
        # print(p.getLinkState(self.robotID, 7 , computeLinkVelocity = 1)[6])
        end = time.time()

        # print(end-start)

    def move(self, action,setTime=0.01):
        stepSize = 240*setTime
        currentPose = self.getRobotPoseE()
        delta = []
        for i in range(4):
            delta.append(action[i]*0.03/stepSize)

        for t in range(int(stepSize)):
            stepPos = []
            stepOri = []
            for i in range(3):
                stepPos.append(currentPose[i] + (t+1)*delta[i])
            stepOri.append(0)
            stepOri.append(1.5707963)
            stepOri.append(currentPose[5] + (t+1)*delta[3])
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
        #print(joint_positions)

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
    def process3(self,angle):
        if angle>1.5707963/2+1.5707963:
            return angle-1.5707963/2-1.5707963
        elif angle>1.5707963:
            return angle-1.5707963
        elif angle>1.5707963/2:
            return angle-1.5707963/2
        elif angle<-1.5707963-1.5707963/2:
            return angle+1.5707963+1.5707963
        elif angle<-1.5707963:
            return angle+1.5707963+1.5707963/2
        elif angle<-1.5707963/2:
            return angle+1.5707963
        elif angle<0:
            return angle+1.5707963/2
        else:
            return angle
    def process2(self,angle):
        angle %=1.5707963*2
        if angle>3*1.5707963/4+1.5707963:
            return angle-1.5707963*2
        elif angle>1.5707963/4+1.5707963:
            return angle-1.5707963/2-1.5707963
        elif angle>1.5707963-1.5707963/4:
            return angle-1.5707963
        elif angle>1.5707963/4:
            return angle-1.5707963/2
        elif angle<-1.5707963*2+1.5707963/4:
            return angle+1.5707963+1.5707963
        elif angle<-1.5707963-1.5707963/4:
            return angle+1.5707963+1.5707963/2
        elif angle<-1.5707963+1.5707963/4:
            return angle+1.5707963
        elif angle<-1.5707963/4:
            return angle+1.5707963/2
        else:
            return angle
    def process(self,angle):
        if angle>1.5707963:
            return angle-3.14
        elif angle<-1.5707963:
            return angle+3.14
        else:
            return angle
    def get_state(self):

        # Joints states
        joint_states = p.getJointStates(self.robotID, range(p.getNumJoints(self.robotID)))
        #print(joint_states,'\n')
        #print('========\n')
        # >>>joint_states.shape :  (8, 4)
        # print("joint_states.shape : {}".foramt(np.array(joint_states).shape))
        joint_positions = [state[0] for state in joint_states] #  -> (8,)
        jp=np.asarray(joint_positions)
        #joint_velocities = [state[1] for state in joint_states]
        #state_list = []
        #vel_list = []
        joint_torque=[state[3] for state in joint_states]
        torque_list=[]
        for i in range(len(self.controlJoints)):
            if i > 6:
                break
            torque_list.append(joint_torque[self.joint.id])
        torque=np.asarray(torque_list)
        # End-Effector States
        ee_states = p.getLinkState(self.robotID, 7 , computeLinkVelocity = 1)
        ee_pos = ee_states[0]
        ee_ori = ee_states[1]
        ee_linear_vel = ee_states[6]
        #ee_angular_vel = ee_states[7]
        ee_ori = p.getEulerFromQuaternion(ee_ori)
        object_pos = []
        for i in range(3):
            object_pos.append(self.holePos[i])
        #object_pos[2] += 0.3
        object_ori = []
        for i in range(4):
            object_ori.append(self.holeOri[i])

        object_ori = p.getEulerFromQuaternion(object_ori)
        print(object_ori)
        self.object_rel_pos = []
        rel_ori=[] 
        ee_ori[2]-object_pos[2]

        for i in range(3):
            self.object_rel_pos.append(-object_pos[i]+ee_pos[i])
        for i in range(3):
            rel_ori.append(ee_ori[i]-object_ori[i])    
        self.contact = False
        self.contact1 = p.getContactPoints(bodyA=self.robotID,bodyB=self.boxId)
        self.contact2 = p.getContactPoints(bodyA=self.robotID,bodyB=self.tableID)
        
        if (self.contact1) or (self.contact2):
            self.contact = True
            print("collision")

        # Final states
        # obs = np.concatenate([
        #     ee_pos, ee_linear_vel, #gripper_state, gripper_vel,
        #     object_pos, object_ori, object_linear_vel, object_angular_vel,
        #     self.object_rel_pos, object_rel_vel,
        # ])self.object_rel_pos[0],self.object_rel_pos[1],self.object_rel_pos[2],, rel_ori
        
        #rel_ori = object_ori[2] - ee_ori[2]
        obs_temp = np.array([
            ee_pos[0], ee_pos[1], ee_pos[2], abs(self.process(ee_ori[1])),self.process2(ee_ori[0]),self.process2(ee_ori[2]),
                    ee_linear_vel[0],ee_linear_vel[1],ee_linear_vel[2],
            ]) #말단과 hole의 x, y위치만 선정
        obs1=np.concatenate((obs_temp,torque),axis=None)
        obs2=np.concatenate((obs1,jp),axis=None)
        obs_temp2=np.array([object_pos[0],object_pos[1],object_pos[2],object_ori[0],object_ori[1],object_ori[2]])
        obs=np.concatenate((obs2,obs_temp2),axis=None)
        #print(ee_ori)
        return obs

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def lol(self):
        print('ll')
        while True:
            for i in range(100):
                p.setJointMotorControl2(self.robotID,
                                        5,
                                        p.POSITION_CONTROL,
                                        targetPosition = i*Deg2Rad)
                                        # targetVelocity = 5,
                                        #force = joint.maxForce, 
                                        #maxVelocity = joint.maxVelocity)
                p.stepSimulation()
                print('???')
        time.sleep(1000)