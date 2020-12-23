from env.quadruped.cheetah import Cheetah

from pybullet_utils import bullet_client
import pybullet_data
import pybullet

from copy import deepcopy
import numpy as np
import random
import time
import copy
import gym
import os

class Env:
  def __init__(self, enable_draw=False, base_fix=False):
    self.enable_draw = enable_draw
    self.time_step = 0.01
    self.num_solver_iterations = 100
    self.sub_step = 1
    self.elapsed_t = 0
    self.sub_time_step = self.time_step / self.sub_step

    if self.enable_draw:
      self.pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
      self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_GUI, 0)
    else:
      self.pybullet_client = bullet_client.BulletClient()
    self.pybullet_client.setAdditionalSearchPath('{}/urdf'.format(os.path.dirname(os.path.realpath(__file__))))
    self.pybullet_client.setPhysicsEngineParameter(numSolverIterations=self.num_solver_iterations)
    self.pybullet_client.setTimeStep(self.sub_time_step)

    #make vertical plane(bottom plane)
    plane_pos = [0, 0, 0]
    plane_orn = self.pybullet_client.getQuaternionFromEuler([0, 0, 0])
    planeId = self.pybullet_client.loadURDF("plane_implicit.urdf", plane_pos, plane_orn, useMaximalCoordinates=True)
    self.pybullet_client.setGravity(0, 0, -9.8)
    self.pybullet_client.changeDynamics(planeId, linkIndex=-1, lateralFriction=0.9)

    #make model
    self.model = Cheetah(self.pybullet_client, useFixedBase=base_fix)
    self.num_leg = self.model.num_leg


  def reset(self):
    self.elapsed_t = 0
    self.model.reset()
    state = self.model.get_state()
    #camera
    self.camera_reset(self.model.sim_model)
    self.camera_move()
    return state 

  def step(self, force_list, contact_list=[], desired_foot_list=[]):
    force_list = np.reshape(force_list, (4, 3))
    for i in range(self.sub_step):
      torque_list = self.model.get_torque_list(force_list, contact_list, desired_foot_list, self.elapsed_t)
      self.model.apply_torque(torque_list)
      self.pybullet_client.stepSimulation()
      self.elapsed_t += self.sub_time_step
    state = self.model.get_state()
    #camera
    self.camera_move()
    return state
    
  def close(self):
    pybullet.disconnect(physicsClientId=self.pybullet_client._client)
    del self.pybullet_client
    del self.model

  def camera_reset(self, target):
    self.camDist = 1.0
    self.camYaw = 30.0
    self.camPitch = -15.0
    self.camTarget = target
    self.camTargetPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.camTarget)
    self.camTargetPos = np.array(self.camTargetPos)

  def camera_move(self, alpha = 0.9):
    targetPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.camTarget)
    targetPos = np.array(targetPos)
    self.camTargetPos = alpha*self.camTargetPos + (1-alpha)*targetPos
    self.pybullet_client.resetDebugVisualizerCamera(self.camDist, self.camYaw, self.camPitch, self.camTargetPos)

  def start_mp4_logging(self, file_name):
    logging_id = self.pybullet_client.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, file_name)
    return logging_id

  def stop_mp4_logging(self, logging_id):
    self.pybullet_client.stopStateLogging(logging_id)
