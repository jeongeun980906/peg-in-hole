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
import random

ACTION_RANGE = 1.0
OBJ_RANGE = 0.1
Deg2Rad = 3.141592/180.0
DEG2RAD = 3.141592/180.0
RAD2DEG = 180.0/3.141592

class UR5_robotiq():

    def __init__(self, args):
        
        # print(args.GUI)
        if args.GUI =='GUI':
            self.serverMode = p.GUI # GUI/DIRECT
        else:
            self.serverMode = p.DIRECT

        self.thforce=0.01
        self.initdis=0
        self.distance_threshold = 0.001
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

        self.init_pose = [-0.18293805373028393, -1.1209140643403597, 1.778449642199667, -2.2283957002700183, -1.5707963018441111, -0.4447348731959547]
        self.goal_pose=[-0.1830197380957823, -1.0908494021052788, 1.7952853703555125, -2.275321638666715, -1.5707920663516683, -0.4448227675725946]

        self.home_pose()
        self.ctrlPeriod =1.0/240.0
        p.setTimeStep(self.ctrlPeriod)

        self.controlInit = True

        pose = self.getRobotPose()
        self.Orn = [pose[3], pose[4], pose[5], pose[6]]
        self.fOri=0
        # self.goal = np.array(self.targetStartPos)elf.d_oldelf.d_old


    def loadURDF_all(self):
        self.UR5UrdfPath = "./urdf/ur5_peg_2.urdf"
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
        #tableStartPos = [-0.7, 0.0, 0.8]
        tableStartOrientation = p.getQuaternionFromEuler([0, 0, 90.0*Deg2Rad])
        self.tableID = p.loadURDF("./urdf/objects/table.urdf", tableStartPos, tableStartOrientation,useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        # define environment
        
        self.holePos = [0.6, 0.0 , 0.85]
        #self.holePos = [0.65, 0.025, 0.87]
        self.holeOri = p.getQuaternionFromEuler([0, 0, 0.261799333]) #.261799333
        #self.holeOri = p.getQuaternionFromEuler([1.57079632679, 0, 0.261799333])
        self.boxID = p.loadURDF(
        "./urdf/peg_hole_gazebo/hole/urdf/new_hole_1.urdf",
        self.holePos, self.holeOri,
        flags = p.URDF_USE_INERTIA_FROM_FILE,useFixedBase=True)

        # self.boxID = p.loadURDF(
        # "./urdf/peg_hole_gazebo/hole/urdf/hole.SLDPRT.urdf",
        # self.holePos, self.holeOri,
        # flags = p.URDF_USE_INERTIA_FROM_FILE,useFixedBase=True)
        #self.boxId = p.loadURDF(
        #"./urdf/peg_hole_gazebo/hole/urdf/new_hole.urdf",
        #self.holePos, self.holeOri,
        #flags = p.URDF_USE_INERTIA_FROM_FILE,useFixedBase=True)
        
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
        
        self.last=0

    def step(self, action):
        # if True:
        if self.controlInit:
            self.controlInit = False
            #self.move(action)
            self.AdmittanceCtrl(action)
            #self.down(action)
            self.next_state_dict = self.get_state()
            rel_pose1=np.asarray([self.next_state_dict[0]/100,self.next_state_dict[1]/100])
            rel_pose2=np.asarray([self.next_state_dict[0]/100,self.next_state_dict[1]/100,self.next_state_dict[2]/5])
            rel_ori=np.asarray([self.next_state_dict[3]/100,self.next_state_dict[4]/100,self.next_state_dict[5]/100,self.next_state_dict[6]/100])
            tot=np.concatenate((rel_pose2,rel_ori),axis=None)
            dis_error=np.linalg.norm(rel_pose2, axis=-1, ord=2)
            dis_error2=np.linalg.norm(rel_pose1, axis=-1, ord=2)
            tot_error=np.linalg.norm(tot, axis=-1, ord=2)
            #force=self.next_state_dict[7]

            self.done=False
            #self.done = self.contact
            info=tot_error
            reward=-(tot_error)/0.035+1
            if tot_error>0.035:
                self.done=True
                print('out of range')
                #print(dis_error,ori_error)
                reward=0
            
            if tot_error<0.005:
                reward=1
                #self.down(0.05)
                print('goal')
                self.done=True
            
           # if self.contact==():
                #reward-=0.5
                #self.down(0.005)
            #else:
                #reward-=temp
            #print(reward)
            return self.next_state_dict, reward, self.done, info

    def reset(self):
        print('reset')
        self.home_pose()
        self.home_pose2()
        self.state_dict = self.get_state()
        temp=self.getRobotPose()
        # # p.removeAllUserDebugItems()
        
        print('init_pose',temp)
        
        return self.state_dict

    def home_pose2(self):
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            p.setJointMotorControl2(self.robotID,
                                        joint.id,
                                        p.VELOCITY_CONTROL,
                                        #targetPosition = targetJointPos,
                                         targetVelocity = 0,
                                        force = joint.maxForce)
        for _ in range(500):
            p.stepSimulation()
        #robotTransVel, robotAngularVel = self.getRobotVel(1)
        #robotVel = robotTransVel + robotAngularVel
        #robotVel = [0.0] * 6
        #print("robotVel: {}".format(robotVel))
    
    def home_pose(self):
        for i, name in enumerate(self.controlJoints):
            if i > 6:
                break
            # print(i()
            self.joint = self.joints[name]
            seed=random.random()
            pose1 = self.init_pose[i]+0.01*(seed-0.5)
            #p.resetJointState(self.robotID, self.joint.id, targetValue=pose1, targetVelocity=0)
            if i < 6:
                p.resetJointState(self.robotID, self.joint.id, targetValue=pose1, targetVelocity=1e-5)
       
        for _ in range(5):
            p.stepSimulation()
        self.move([0.0,0.0,0.0,0.0])
    
    def move(self, action,setTime=0.01):
        stepSize = 240*setTime
        currentPose = self.getRobotPose()
            
        stepPos=[]
        stepOri=[]
        for i in range(2):
            stepPos.append(currentPose[i] + 0.0002* action[i])
        stepPos.append(currentPose[2] - 0.002* (action[2]+1))
        stepOri.append(currentPose[3])
        stepOri.append(0.7011236967207826)
        stepOri.append(-currentPose[3])
        stepOri.append(0.7011236967207826)

        jointPos = p.calculateInverseKinematics(self.robotID,
                                                    self.eefID,
                                                    stepPos,
                                                    stepOri)
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            if i==6:
                targetJointPos = jointPos[i]+action[3]*0.0005
            else:
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
            
        for _ in range(25):
            p.stepSimulation()
        #time.sleep(0.1)
            # self.getCameraImage()
        #print('step')
        
    def down(self,l,setTime=0.01):
        stepSize = 240*setTime
        currentPose = self.getRobotPose()
        print('c',currentPose)
        #time.sleep(5)
        stepPos=[]
        stepOri=[]
        stepPos.append(currentPose[0]-0.0005)
        stepPos.append(currentPose[1]-0.0005)
        stepPos.append(currentPose[2]-l)
        stepOri.append(currentPose[3])
        stepOri.append(currentPose[4])
        stepOri.append(currentPose[5])
        stepOri.append(currentPose[6])

        jointPos = p.calculateInverseKinematics(self.robotID,
                                                    self.eefID,
                                                    stepPos,
                                                    stepOri)
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            targetJointPos = jointPos[i]
            #print(targetJointPos)
            p.setJointMotorControl2(self.robotID,
                                        joint.id,
                                        p.POSITION_CONTROL,
                                        targetPosition = targetJointPos,
                                        # targetVelocity = 5,
                                        force = joint.maxForce, 
                                        maxVelocity = joint.maxVelocity)

            # p.addUserDebugLine((0.6475237011909485, 0.6443161964416504, 0.9296525716781616),(0,0,0))
        # for i in range(10):
            
        for _ in range(100):
            p.stepSimulation()
    def AdmittanceCtrl(self, action):
        # targetPos = []
        # for i in range(2):
        #     targetPos.append(currentPose[i] + 0.0002* action[i])
        # targetPos.append(currentPose[2] - 0.002* (action[2]+1))
        # targetPos.append(currentPose[3])
        # targetPos.append(0.7011236967207826)
        # targetPos.append(-currentPose[3])
        # targetPos.append(0.7011236967207826)

        # action = del[0.0002,0.0002,0.0002,0.0,0.01,0.0]
        currentPose = self.getRobotPoseE()
        targetPos = [0.0] * 6
        for i in range(2):
            targetPos[i] = currentPose[i] + 0.0005* action[i]#.item()
        targetPos[2] = currentPose[2] - 0.001*(action[2])#.item()+0.8)
        targetPos[3] = currentPose[3]   #+0.0002 * action[3].item()
        targetPos[4] = currentPose[4]   #+0.0002 * action[3].item()
        targetPos[5] = currentPose[5] + 0.0005 * action[3]#.item()

        ## Set desired force
        DForce = [0.0] * 6

        ## Set Md, Md_inv, Bd, Kd
        Md = [0.0] * 6
        Bd = [0.0] * 6
        Kd = [0.0] * 6

        Md[0] = 10.0#1.0#
        Md[1] = 10.0#1.0#
        Md[2] = 10.0#1.0
        Md[3] = 0.01* Deg2Rad#0.001 * Deg2Rad
        Md[4] = 0.01* Deg2Rad#0.001 * Deg2Rad
        Md[5] = 0.001* Deg2Rad#0.0001 * Deg2Rad

        Bd[0] = 1000.0
        Bd[1] = 1000.0
        Bd[2] = 2000.0
        Bd[3] = 0.1 * Deg2Rad
        Bd[4] = 0.1 * Deg2Rad
        Bd[5] = 0.01 * Deg2Rad

        Kd[0] = 250.0 #250.0
        Kd[1] = 250.0 #250.0
        Kd[2] = 1000.0 #1000.0
        Kd[3] = 0.25 * Deg2Rad#0.025 * Deg2Rad
        Kd[4] = 0.25 * Deg2Rad#0.025 * Deg2Rad
        Kd[5] = 0.025 * Deg2Rad#0.0025 * Deg2Rad

        ## Init Admittance

        self.CSCPos_p1 = [0.0] * 6
        self.CSCPos_p2 = [0.0] * 6

        self.CSCVel_p1 = [0.0] * 6
        self.CSCVel_p2 = [0.0] * 6

        self.CSCAcc_p1 = [0.0] * 6
        self.CSCAcc_p2 = [0.0] * 6

        self.CSCForce_p1 = [0.0] * 6
        self.CSCForce_p2 = [0.0] * 6

        ## S-time of LC sys
        STime = self.ctrlPeriod

        ## Set desired position, velocity, acceleration
        self.PathPlanner(targetPos)
       # print("Stepsize: {}".format(self.stepsize))

        for step in range(self.stepsize):

            ## Detect external force
            force = 0.0
            force_dir = [0.0] * 3
            friction1 = 0.0
            friction1_dir = [0.0] * 3
            friction2 = 0.0
            friction2_dir = [0.0] * 3
            #contactInfo = p.getContactPoints(self.robotID,self.boxID)
            contactInfo = p.getContactPoints(self.robotID)
            try:
                force = contactInfo[0][9]
                force_dir = contactInfo[0][7]
                friction1 = contactInfo[0][10]
                friction1_dir = contactInfo[0][11]
                friction2 = contactInfo[0][12]
                friction2_dir = contactInfo[0][13]
                for i, name in enumerate(self.controlJoints):
                    joint_ = self.joints[name]
                    if force[i] > joint_.maxForce:
                        force = joint_.maxForce
            except:
               pass

            normalForce = self.vecForce(force, force_dir, 3)
            # normalForce = [0.0, 0.0, 0.0]
            # normalForce = self.vecForce(force, force_dir, 3)
            lateralFriction1 = self.vecForce(friction1, friction1_dir, 3)
            lateralFriction2 = self.vecForce(friction2, friction2_dir, 3)

            MForce = [0.0] * 6
            for i in range(3):
                MForce[i] = normalForce[i] + lateralFriction1[i] + lateralFriction2[i]

            dCSForce = [0.0] * 6
            for i in range(6):
                dCSForce[i] = MForce[i] - DForce[i]

            ## Calculate desired position at current time step
            DPos, DVel, DAcc = self.calPolyPath(step)
            # d_x = self.PosInit
            # d_xd = [0.0] * 6
            # d_xdd = [0.0] * 6
            # if step * self.ctrlPeriod == math.ceil(step * self.ctrlPeriod):
            #     print("desired X = {}".format(d_x))
            #     print("desired Xd = {}".format(d_xd))
            #     print("desired Xdd = {}".format(d_xdd))

            Mpos = self.getRobotPoseE()

            ##### AdmittanceFilter #####

            A = [0.0] * 6
            dCSCPos = [0.0] * 6
            dCSCVel = [0.0] * 6
            dCSCAcc = [0.0] * 6

            for i in range(6):
                A[i] = Md[i] + Bd[i] * STime + Kd[i] * STime * STime
                dCSCPos[i] = ( (2 * Md[i] + Bd[i] * STime) * self.CSCPos_p1[i] - Md[i] * self.CSCPos_p2[i] + STime * STime * dCSForce[i] ) / A[i]
                dCSCVel[i] = (dCSCPos[i] - self.CSCPos_p1[i]) / STime
                dCSCAcc[i] = (dCSCVel[i] - self.CSCVel_p1[i]) / STime

            for i in range(6):
                self.CSCVel_p2[i] = self.CSCVel_p1[i]
                self.CSCVel_p1[i] = dCSCVel[i]
                self.CSCPos_p2[i] = self.CSCPos_p1[i]
                self.CSCPos_p1[i] = dCSCPos[i]
                self.CSCForce_p2[i] = self.CSCForce_p1[i]
                self.CSCForce_p1[i] = dCSForce[i]

            PoseSum = [0.0] * 6
            VelSum = [0.0] * 6
            AccSum = [0.0] * 6

            for i in range(6):
                PoseSum[i] = DPos[i] + dCSCPos[i]
                VelSum[i] = DVel[i] + dCSCVel[i]
                AccSum[i] = DAcc[i] + dCSCAcc[i]

            ## Rotation Sum
            Rot_dRotation1 = self.RPY2Rot(DPos)
            Rot_dRotation2 = self.RPY2Rot(dCSCPos)
            Rot_dRotationSum = self.RotSum(Rot_dRotation2, Rot_dRotation1)
            dRotationSum = self.Rot2RPY(Rot_dRotationSum)
            for i in range(3):
                PoseSum[i+3] = dRotationSum[i]

            # Ang_Vel1 = self.RPYdot2AngularVel(DPos, DVel)
            # Ang_Vel2 = self.RPYdot2AngularVel(dCSCPos, dCSCVel)
            # Ang_VelSum = [0.0] * 3
            # for i in range(3):
            #     Ang_VelSum[i] = Ang_Vel1[i] + Ang_Vel2[i]
            # dAbsVelSum = self.AngularVel2RPYdot(PoseSum, Ang_VelSum, VelSum)
            #
            # Ang_Acc1 = self.RPYAcc2AngularAcc(DPos, DVel, DAcc)
            # Ang_Acc2 = self.RPYAcc2AngularAcc(dCSCPos, dCSCVel, dCSCAcc)
            # Ang_AccSum = [0.0] * 3
            # for i in range(3):
            #     Ang_AccSum[i] = Ang_Acc1[i] + Ang_Acc2[i]
            # dAbsAccSum = self.AngularAcc2RPYAcc(PoseSum, VelSum, Ang_AccSum, AccSum)

            stepPos = []
            stepOri_Euler = []
            # stepOri = []
            for i in range(3):
                stepPos.append(PoseSum[i])
                stepOri_Euler.append(PoseSum[i+3])
            # print("Here : {}".format(stepOri_Euler))

            stepOri = p.getQuaternionFromEuler(stepOri_Euler)
            # print("HHEE : {}".format(p.getQuaternionFromEuler(stepOri_Euler)))

            # print("Desired step pose")
            # print(stepPos)
            # print(stepOri)

            jointPos = p.calculateInverseKinematics(self.robotID,
                                                    self.eefID,
                                                    stepPos,
                                                    stepOri)

            for i, name in enumerate(self.controlJoints):
                joint_ = self.joints[name]
                targetJointPos = jointPos[i]

                p.setJointMotorControl2(self.robotID,
                                        joint_.id,
                                        p.POSITION_CONTROL,
                                        targetPosition = targetJointPos,
                                        force = joint_.maxForce,
                                        maxVelocity = joint_.maxVelocity)



            p.stepSimulation()
            #time.sleep(self.ctrlPeriod)

        self.controlInit = True
    
    
    def PathPlanner(self, targetPos):
        ## Get current robot position
        robotPose = self.getRobotPoseE()
        # print("robotPos : {}".format(robotPose))
        # robotPos : [0.6468485593795776, 0.10915002971887589, 0.9787947535514832,
        #             0.0, 1.5707963267948966, -3.2679002890726813e-07]

        ## Get current robot velocity
        robotTransVel, robotAngularVel = self.getRobotVel(1)
        robotVel = robotTransVel + robotAngularVel
        #robotVel = [0.0] * 6

        ## Set MaxVel, MaxAcc
        MaxVel = [0.0] * 6
        MaxAcc = [0.0] * 6
        for i in range(3):
            MaxVel[i] = 0.01
            MaxAcc[i] = 0.01
            MaxVel[i+3] = 5.0 * Deg2Rad
            MaxAcc[i+3] = 5.0 * Deg2Rad

        ## cal Distance from current pose to target pose
        Distance = [0.0] * 6
        for i in range(6):
            Distance[i] = targetPos[i] - robotPose[i]

        for i in range(3):
            if Distance[i+3] > 180.0 * Deg2Rad:
                Distance[i+3] = Distance[i+3] - 360.0 * DEG2RADsetGravity
            elif Distance[i+3] < -180.0 * Deg2Rad:
                Distance[i+3] = Distance[i+3] + 360.0 * Deg2Rad

        # print("Distance : {}".format(Distance))

        ## Set actual acceleration of polypath
        dAcc = [0.0] * 6
        for i in range(6):
            dAcc[i] = 1/3 * MaxAcc[i]

        ## Calculate Max velocity at triangular trajectory using current velocity
        dVelMax = [0.0] * 6
        for i in range(6):
            if Distance[i] >= math.fabs(robotVel[i]) * robotVel[i] / 2.0 / dAcc[i]:
                dVelMax[i] = math.sqrt(math.fabs(robotVel[i] * robotVel[i] / 2.0 + dAcc[i] * Distance[i]))
            else:
                dVelMax[i] = -math.sqrt(math.fabs(robotVel[i] * robotVel[i] / 2.0 - dAcc[i] * Distance[i]))

        ## Decide dVelUni comparing dVelMax with MaxVel
        dVelUni = [0.0] * 6
        for i in range(6):
            if dVelMax[i] >= MaxVel[i]:
                dVelUni[i] = MaxVel[i]
            elif dVelMax[i] <= (-MaxVel[i]):
                dVelUni[i] = - MaxVel[i]
            else:
                dVelUni[i] = dVelMax[i]

        ## Calculate FinalTime using dVelUni
        act_dtime = [0.0001] * 3
        Temp = 0.0

        for i in range(6):
            ## Acc time
            Temp = math.fabs(dVelUni[i] - robotVel[i]) /  dAcc[i]
            if Temp > act_dtime[0]:
                act_dtime[0] = Temp

            ## Decel time
            Temp = math.fabs(dVelUni[i]) / dAcc[i]
            if Temp > act_dtime[2]:
                act_dtime[2] = Temp

            ## Const Vel time
            if Temp < 0.0001:
                Temp = 0.0001
            else:
                Temp = (Distance[i] - (dVelUni[i] + robotVel[i]) / 2.0 * math.fabs(dVelUni[i] - robotVel[i]) / dAcc[i] - dVelUni[i] / 2.0 * math.fabs(dVelUni[i]) / dAcc[i]) / dVelUni[i]
            if Temp > act_dtime[1]:
                act_dtime[1] = Temp

        ## Recalculate dVelUni using Max act_dtime
        for i in range(6):
            dVelUni[i] = (Distance[i] - robotVel[i] * act_dtime[0] / 2.0) / (act_dtime[0] / 2.0 + act_dtime[1] + act_dtime[2] / 2.0)

        ## Set variable for calpolypath
        self.PosInit = [0.0] * 6
        self.PosFin = [0.0] * 6
        self.VelUni = [0.0] * 6
        self.VelInit = [0.0] * 6

        for i in range(6):
            self.PosInit[i] = robotPose[i]
            self.PosFin[i] = robotPose[i] + Distance[i]
            self.VelInit[i] = robotVel[i]
            self.VelUni[i] = dVelUni[i]

        # print("PosInit: {}".format(self.PosInit))
        # print("PosFin: {}".format(self.PosFin))

        self.TimeInit = 0.0
        self.TimeFin = self.TimeInit + act_dtime[0] + act_dtime[1] + act_dtime[2]

        Time_step = self.TimeFin / self.ctrlPeriod
        self.stepsize = math.ceil(Time_step) + 1

        self.PolyPathNum = 3

        np_PolyPathTime = np.zeros([3, 2])
        self.PolyPathTime = np_PolyPathTime.tolist()
        self.PolyPathTime[0][0] = self.TimeInit
        self.PolyPathTime[0][1] = self.PolyPathTime[0][0] + act_dtime[0]

        self.PolyPathTime[1][0] = self.PolyPathTime[0][1]
        self.PolyPathTime[1][1] = self.PolyPathTime[1][0] + act_dtime[1]

        self.PolyPathTime[2][0] = self.PolyPathTime[1][1]
        self.PolyPathTime[2][1] = self.TimeFin
        # print("Time Fin : {}".format(self.TimeFin))

        # print(self.PolyPathTime)

        ## Set coefficient of 5th-polypath
        # print("dVelUni : {}".format(dVelUni))
        # print("act_dtime : {}".format(act_dtime))
        np_PolyPathCoff = np.zeros([3, 6, 6])
        self.PolyPathCoff = np_PolyPathCoff.tolist()
        for i in range(6):
            Temp0 = 0.0
            Temp1 = dVelUni[i] - robotVel[i]
            self.PolyPathCoff[0][i][0] = self.PosInit[i]
            self.PolyPathCoff[0][i][1] = robotVel[i]
            self.PolyPathCoff[0][i][2] = 0.0
            self.PolyPathCoff[0][i][3] = Temp1 / math.pow(act_dtime[0], 2)
            self.PolyPathCoff[0][i][4] = -Temp1 / (2.0 * math.pow(act_dtime[0], 3))
            self.PolyPathCoff[0][i][5] = Temp0

            Temp0 = 0.0
            Temp1 = 0.0
            self.PolyPathCoff[1][i][0] = (dVelUni[i] + robotVel[i]) * act_dtime[0] / 2.0 + self.PosInit[i]
            self.PolyPathCoff[1][i][1] = dVelUni[i]
            self.PolyPathCoff[1][i][2] = 0.0
            self.PolyPathCoff[1][i][3] = Temp0
            self.PolyPathCoff[1][i][4] = Temp0
            self.PolyPathCoff[1][i][5] = Temp0

            Temp0 = 0.0
            Temp1 = dVelUni[i]
            self.PolyPathCoff[2][i][0] = self.PosFin[i] - (dVelUni[i]) * act_dtime[2] / 2.0
            self.PolyPathCoff[2][i][1] = dVelUni[i]
            self.PolyPathCoff[2][i][2] = 0.0
            self.PolyPathCoff[2][i][3] = -Temp1 / math.pow(act_dtime[2], 2)
            self.PolyPathCoff[2][i][4] = Temp1 / (2.0 * math.pow(act_dtime[2], 3))
            self.PolyPathCoff[2][i][5] = Temp0

        # print("PolyPath coff (Accel): {}".format(self.PolyPathCoff[0]))
        # print("PolyPath coff (Const): {}".format(self.PolyPathCoff[1]))
        # print("PolyPath coff (Decel): {}".format(self.PolyPathCoff[2]))
    
    def calPolyPath(self, step_):
        ## Current time
        if step_ == (self.stepsize-1):
            t = self.TimeFin
        else:
            t = step_ * self.ctrlPeriod

        #if t == math.ceil(t):
        #    print("Time : {} sec".format(t))
            # print("Total step Size : {} step".format(self.stepsize-1))
            # print("Current step : {} step".format(step_))
        #elif t == self.TimeFin:
            # print("End step!!")
        #    print("End Time : {} sec".format(t))
            # print("Total step Size : {} step".format(self.stepsize-1))
            # print("Final step : {} step".format(step_))

        ## Calculate desired Pos, Vel, Acc at this time step
        d_x = [0.0] * 6
        d_xd = [0.0] * 6
        d_xdd = [0.0] * 6

        if t < self.TimeInit:
            for i in range(6):
                d_x[i] = self.PosInit[i]
                d_xd[i] = 0.0
                d_xdd[i] = 0.0
        elif t < self.TimeFin:
            for i in range(6):
                for k in range(self.PolyPathNum):
                    if self.PolyPathTime[k][0] <= t and t < self.PolyPathTime[k][1]:
                        d_x[i] = 0.0
                        d_xd[i] = 0.0
                        d_xdd[i] = 0.0
                        for j in range(0,6):
                            d_x[i] += self.PolyPathCoff[k][i][j] * math.pow(t - self.PolyPathTime[k][0], j)
                        for j in range(1,6):
                            d_xd[i] += j * self.PolyPathCoff[k][i][j] * math.pow(t - self.PolyPathTime[k][0], j - 1)
                        for j in range(2,6):
                            d_xdd[i] += (j * (j - 1)) * self.PolyPathCoff[k][i][j] * math.pow(t - self.PolyPathTime[k][0], j - 2)
            # if t == math.ceil(t):
            #     print("desired X : {}".format(d_x))
        else:
            for i in range(6):
                d_x[i] = self.PosFin[i]
                d_xd[i] = 0.0
                d_xdd[i] = 0.0

        return d_x, d_xd, d_xdd

    def Rot2RPY(self, dRot):
        dRPY = [0.0] * 3
        dRPY[0] = math.atan2(dRot[2][1], dRot[2][2]) * RAD2DEG
        dRPY[1] = math.atan2(-dRot[2][0], math.sqrt(dRot[2][1] * dRot[2][1] + dRot[2][2] * dRot[2][2])) * RAD2DEG
        dRPY[2] = math.atan2(dRot[1][0], dRot[0][0]) * RAD2DEG

        return dRPY

    def RPY2Rot(self, RPY_):
        dRot = [ [0.0] * 3, [0.0] * 3, [0.0] * 3 ]
        dRPY = [0.0] * 3
        for i in range(3):
            dRPY[i] = RPY_[i+3]

        dRot[0][0] =  math.cos(dRPY[2] * DEG2RAD) * math.cos(dRPY[1] * DEG2RAD)
        dRot[0][1] =  math.cos(dRPY[2] * DEG2RAD) * math.sin(dRPY[1] * DEG2RAD) * math.sin(dRPY[0] * DEG2RAD) - math.sin(dRPY[2] * DEG2RAD) * math.cos(dRPY[0] * DEG2RAD)
        dRot[0][2] =  math.cos(dRPY[2] * DEG2RAD) * math.sin(dRPY[1] * DEG2RAD) * math.cos(dRPY[0] * DEG2RAD) + math.sin(dRPY[2] * DEG2RAD) * math.sin(dRPY[0] * DEG2RAD)

        dRot[1][0] =  math.sin(dRPY[2] * DEG2RAD) * math.cos(dRPY[1] * DEG2RAD)
        dRot[1][1] =  math.sin(dRPY[2] * DEG2RAD) * math.sin(dRPY[1] * DEG2RAD) * math.sin(dRPY[0] * DEG2RAD) + math.cos(dRPY[2] * DEG2RAD) * math.cos(dRPY[0] * DEG2RAD)
        dRot[1][2] =  math.sin(dRPY[2] * DEG2RAD) * math.sin(dRPY[1] * DEG2RAD) * math.cos(dRPY[0] * DEG2RAD) - math.cos(dRPY[2] * DEG2RAD) * math.sin(dRPY[0] * DEG2RAD)

        dRot[2][0] = -math.sin(dRPY[1] * DEG2RAD)
        dRot[2][1] =  math.cos(dRPY[1] * DEG2RAD) * math.sin(dRPY[0] * DEG2RAD)
        dRot[2][2] =  math.cos(dRPY[1] * DEG2RAD) * math.cos(dRPY[0] * DEG2RAD)

        return dRot

    def RotSum(self, dRot1, dRot2):
        dRotSum = [ [0.0] * 3, [0.0] * 3, [0.0] * 3 ]
        for i in range(3):
            for j in range(3):
                dRotSum[i][j] = 0.0
                for k in range(3):
                    dRotSum[i][j] += dRot1[i][k] * dRot2[k][j]

        return dRotSum
    
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

    def getRobotVel(self, inRad = 0):
        currentPos = []
        for i in range(3):
            currentPos.append(p.getLinkState(bodyUniqueId=self.robotID,
                                             linkIndex=7,
                                             computeLinkVelocity=1)[6][i])

        currentOri = []
        for i in range(3):
            currentOri.append(p.getLinkState(bodyUniqueId=self.robotID,
                                             linkIndex=7,
                                             computeLinkVelocity=1)[7][i])

        if inRad == 0:
            return currentPos, currentOri

        elif inRad == 1:
            currentOri_rad = []
            for i in range(3):
                currentOri_rad.append(currentOri[i]*(1/Deg2Rad))
            return currentPos, currentOri_rad           

    def get_state(self):
        object_pos = [0.6, 0.0, 0.92]
        object_ori= [1.57079632679, 0, 0.261799333]
        object_ori= p.getQuaternionFromEuler(object_ori)
        joint_states = p.getJointStates(self.robotID, range(p.getNumJoints(self.robotID)))

        joint_positions = [state[0] for state in joint_states] #  -> (8,)
        jp=np.asarray(joint_positions)
        joint_torque=[state[3] for state in joint_states]
        ee_force=joint_states[7][2]
        torque_list=[]
        for i in range(len(self.controlJoints)):
            if i > 6:
                break
            torque_list.append(joint_torque[self.joint.id])
        #print(torque_list)
        #torque=np.asarray(torque_list)
        ee_force=np.asarray(ee_force)
        #print(ee_force)
        # End-Effector States
        ee_states = p.getLinkState(self.robotID, 7 , computeLinkVelocity = 1)
        ee_pos = ee_states[0]
        ee_ori = ee_states[1]
        #print(ee_ori)
        #ee_angular_vel = ee_states[7]
        ee_linear_vel=ee_states[6]
        self.contact = p.getContactPoints(bodyA=self.robotID,bodyB=self.boxID,linkIndexA=7)
        #print(self.contact)
        #self.contact2 = p.getContactPoints(bodyA=self.robotID,bodyB=self.tableID)
        #print(ee_ori,object_ori)
        # Final states
        # obs = np.concatenate([
        #     ee_pos, ee_linear_vel, #gripper_state, gripper_vel,
        #     object_pos, object_ori, object_linear_vel, object_angular_vel,
        #     self.object_rel_pos, object_rel_vel,
        # ])self.object_rel_pos[0],self.object_rel_pos[1],self.object_rel_pos[2],, rel_ori
        #rel_ori = object_ori[2] - ee_ori[2]
        obs= np.array([
            100*(ee_pos[0]-object_pos[0]),100*(ee_pos[1]-object_pos[1]), (ee_pos[2]-object_pos[2])/0.2,
            100*(ee_ori[0]+object_ori[1]),100*(ee_ori[1]-object_ori[0]),100*(ee_ori[2]-object_ori[2]),100*(ee_ori[3]-object_ori[3])
            ,ee_force[0]/50,ee_force[1]/50,ee_force[2]/50,ee_linear_vel[0],ee_linear_vel[1],ee_linear_vel[2]
            ])
        #obs1=np.concatenate((obs_temp,ee_force),axis=None)
        ##obs2=np.concatenate((obs,jp),axis=None)
        #print(obs)
        return obs
    
    def vecForce(self, F, Dir, dim):
        result = [0.0] * dim
        for i in range(dim):
            result[i] = F * Dir[i]

        return result

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def lol(self):
        object_pos=[]
        for i in range(3):
            if i==2:
                object_pos.append(self.holePos[i]+0.11)
            else:
                object_pos.append(self.holePos[i])
        currentPose = self.getRobotPoseE()
        objectOri = []
        objectOri.append(0)
        objectOri.append(1.5707963)
        objectOri.append(0.261799333)
        objectOri = p.getQuaternionFromEuler(objectOri)
        jointPos = p.calculateInverseKinematics(self.robotID,
                                                    self.eefID,
                                                    object_pos,
                                                    objectOri)
        print(jointPos)
        time.sleep(1000)

    def process(self,angle):
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