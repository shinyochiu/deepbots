import math

import numpy as np
import argparse

import utilities as utils
from deepbots.supervisor.controllers.supervisor_emitter_receiver import \
    SupervisorCSV
from deepbots.supervisor.wrappers.keyboard_printer import KeyboardPrinter
from deepbots.supervisor.wrappers.tensorboard_wrapper import TensorboardLogger
from models.networks import DDPG

import gym
from gym import spaces

import tensorflow as tf
import maddpg.common.tf_util as U
from maddpg.trainer.ddpg import DDPGAgentTrainer
import tensorflow.contrib.layers as layers
import pickle
from datetime import datetime
import time

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for webot environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="formation_tracking", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=250, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=15000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--use_tf", type=bool, default=True, help="using tensorflow")
    parser.add_argument("--num_agent", type=int, default=3, help="number of agent")
    parser.add_argument("--num_obstacle", type=int, default=40, help="number of obstacle")
    parser.add_argument("--leader_observation", type=int, default=20, help="leader observation space")
    parser.add_argument("--follower_observation", type=int, default=20, help="follower observation space")
    parser.add_argument("--action", type=int, default=3, help="action space")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="target update factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--layer1-size", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--layer2-size", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--network", type=str, default="MLP", help="define neural network type")
    parser.add_argument("--trajectory_size", type=int, default=25)
    parser.add_argument("--apply_noise", type=bool, default=True)
    parser.add_argument("--noise_type", type=str, default="adaptive-param_0.2")
    parser.add_argument("--param_noise_adaption_interval", type=int, default=50)

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='Test', help="name of the experiment")
    parser.add_argument("--tf-save-dir", type=str, default='C:/Users/xinyouqiu/Anaconda3/envs/RL3.7/Lib/site-packages/deepbots/deepworlds-master/examples/formation_control/controllers/supervisor/maddpg/policy/model_formation_navigation_1.ckpt', help="directory in which training state and model should be saved")
    parser.add_argument("--save-dir", type=str,
                        default='C:/Users/xinyouqiu/Anaconda3/envs/RL3.7/Lib/site-packages/deepbots/deepworlds-master/examples/formation_control/controllers/supervisor/models/policy/',
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--manual_control", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="C:/Users/xinyouqiu/Anaconda3/envs/RL3.7/Lib/site-packages/deepbots/deepworlds-master/examples/formation_control/controllers/supervisor/maddpg/trainResult/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="C:/Users/xinyouqiu/Anaconda3/envs/RL3.7/Lib/site-packages/deepbots/deepworlds-master/examples/formation_control/controllers/supervisor/maddpg/trainResult/", help="directory where plot data is saved")
    return parser.parse_args()

arglist = parse_args()

DIST_SENSORS_MM = {'min': 0, 'max': 3}
TRACK_MM = {'min': -0.01, 'max': 0.01}
FORMATION_MM = {'min': -10, 'max': 10}
ACTION_MM = {'min': -1, 'max': 1}
ANGLE_MM = {'min': -math.pi, 'max': math.pi}
emitter_names = ['emitter_' + str(i) for i in range(arglist.num_agent)]
receiver_names = ['receiver_' + str(i) for i in range(arglist.num_agent)]

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None, is_training=True):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        def bn_layer(x, epsilon=0.001, decay=0.9, reuse=None, name=None):
            """
            Performs a batch normalization layer
            Args:
                x: input tensor
                scope: scope name
                is_training: python boolean value
                epsilon: the variance epsilon - a small float number to avoid dividing by 0
                decay: the moving average decay
            Returns:
                The ops of a batch normalization layer
            """
            shape = x.get_shape().as_list()
            # gamma: a trainable scale factor

            #gamma = tf.get_variable("gamma"+"_"+name, shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
            gamma = tf.Variable(tf.ones(shape[-1]), trainable=True, name="gamma"+"_"+name)
            # beta: a trainable shift value
            #beta = tf.get_variable("beta"+"_"+name, shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
            beta = tf.Variable(tf.zeros(shape[-1]), trainable=True, name="beta" + "_" + name)
            #avg = tf.get_variable("avg"+"_"+name, shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
            avg = tf.Variable(tf.zeros(shape[-1]), trainable=True, name="avg" + "_" + name)
            #moving_avg = tf.get_variable("moving_avg"+"_"+name, shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
            moving_avg = tf.Variable(tf.zeros(shape[-1]), trainable=True, name="moving_avg" + "_" + name)
            #var = tf.get_variable("var" + "_" + name, shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
            var = tf.Variable(tf.ones(shape[-1]), trainable=True, name="var" + "_" + name)
            #moving_var = tf.get_variable("moving_var"+"_"+name, shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
            moving_var = tf.Variable(tf.ones(shape[-1]), trainable=True, name="moving_var" + "_" + name)
            if is_training:
                # tf.nn.moments == Calculate the mean and the variance of the tensor x
                temp_avg, temp_var = tf.nn.moments(x, np.arange(len(shape) - 1), keep_dims=True)
                temp_avg = tf.reshape(temp_avg, [temp_avg.shape.as_list()[-1]])
                temp_var = tf.reshape(temp_var, [temp_var.shape.as_list()[-1]])
                update_avg = tf.assign(avg, temp_avg)
                update_var = tf.assign(var, temp_var)
                # update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
                update_moving_avg = tf.assign(moving_avg, moving_avg.value() * decay + avg.value() * (1 - decay))
                # update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
                update_moving_var = tf.assign(moving_var, moving_var.value() * decay + var.value() * (1 - decay))
                control_inputs = [update_avg, update_var, update_moving_avg, update_moving_var]
            else:
                avg = moving_avg
                var = moving_var
                control_inputs = []
            with tf.control_dependencies(control_inputs):
                output = tf.nn.relu(tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon))

            return output
        def bn_layer_top(x, epsilon=0.001, decay=0.99, name=None):
            """
            Returns a batch normalization layer that automatically switch between train and test phases based on the
            tensor is_training
            Args:
                x: input tensor
                scope: scope name
                is_training: boolean tensor or variable
                epsilon: epsilon parameter - see batch_norm_layer
                decay: epsilon parameter - see batch_norm_layer
            Returns:
                The correct batch normalization layer based on the value of is_training
            """
            # assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

            '''return tf.cond(
                is_training,
                lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
                lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
            )'''
            return bn_layer(x=x, epsilon=epsilon, decay=decay, reuse=None, name=name) if is_training else bn_layer(x=x, epsilon=epsilon, decay=decay, reuse=True, name=name)
            # out = layers.batch_norm(input, is_training=is_training)
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units*2, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def get_trainers(obs_shape_n,action_space, arglist):
    trainers = []
    if arglist.network=="MLP":
        model = mlp_model
        if arglist.good_policy=="ddpg":
            trainer = DDPGAgentTrainer
            for i in range(2):
                #first for leader second for follower
                trainers.append(trainer("agent_%d" % i, model, obs_shape_n, action_space, i, arglist))
    return trainers

class FindTargetSupervisor(SupervisorCSV):
    def __init__(self, robot, target, observation_space):
        super(FindTargetSupervisor, self).__init__(emitter_name=emitter_names,
                                                   receiver_name=receiver_names)
        self.n = len(robot)
        self.robot_name = [r for r in robot]
        self.target_name = target
        self.robot = [self.supervisor.getFromDef(robot[i]) for i in range(self.n)]

        self.leader = [('LEADER' in robot[i]) for i in range(self.n)]
        self.target = self.supervisor.getFromDef(target)
        self.obs_space = observation_space
        self.observation = [[0 for _ in range(self.obs_space[i != 0])] for i in range(self.n)]
        self.findThreshold = 2
        self.steps = 0
        self.steps_threshold = arglist.max_episode_len
        self.message = [[] for _ in range(self.n)]
        self.dis_des = None
        self.ang_des = None
        self.dis_rel = [None for _ in range(self.n)]
        self.ang_rel = [None for _ in range(self.n)]
        self.dis_prev = [None for _ in range(self.n)]
        self.ang_prev = [None for _ in range(self.n)]
        self.err = [None for _ in range(self.n)]
        self.err_prev = [None for _ in range(self.n)]
        self.should_done = False
        #self.observation_prev = [[None] for _ in range(self.n)]

    def get_observations(self):
        message = self.handle_receiver()
        observation = [[] for _ in range(self.n)]
        self.message = [[] for _ in range(self.n)]
        epsilon = 10e-6
        self.dis_prev = self.dis_rel.copy()
        self.ang_prev = self.ang_rel.copy()
        self.err_prev = self.err.copy()
        self.get_rel_pos()
        for n in range(self.n):
            if message[n][0] is not None:
                if self.dis_prev[n]:
                    if self.leader[n]:
                        #print(self.dis_rel[n] - self.dis_prev[n])
                        observation[n].append(100*(self.dis_rel[n] - self.dis_prev[n]))
                        observation[n].append(self.ang_rel[n]/np.pi)
                    else:
                        observation[n] += self.err[n]

                if self.robot[n]:
                    vel_vec = self.robot[n].getVelocity()
                    vel = np.linalg.norm(vel_vec[:3])
                    omg = vel_vec[4]/np.pi
                    observation[n].append(vel)
                    observation[n].append(omg)
                agt_dis, agt_ang = self.get_agents_pos(self.robot[n])
                if agt_dis[0]:
                    observation[n] += agt_dis
                    observation[n] += agt_ang
                else:
                    observation[n] += [10.0 for _ in range(arglist.num_agent - 1)]
                    observation[n] += [0.0 for _ in range(arglist.num_agent - 1)]
                # data from sensors
                for i in range(1, len(message[n]), 2):
                    mess = [float(message[n][i]), float(message[n][i+1])]
                    self.message[n] += mess
                    observation[n] += mess
            else:
                observation[n] = [0 for _ in range(self.obs_space[n != 0])]

            self.observation = observation
        return self.observation

    def empty_queue(self):
        self.message = [[] for _ in range(self.n)]
        self.observation = [[] for _ in range(self.n)]
        while self.supervisor.step(self.timestep) != -1:
            for i in range(self.n):
                if self.receiver[i].getQueueLength() > 0:
                    self.receiver[i].nextPacket()
                else:
                    if i == self.n-1:
                        break
                    else:
                        continue
            break

    def get_reward(self, action_n):
        reward_n = []
        for i in range(len(action_n)):
            if len(self.message[i]) == 0 or self.observation[i][0] is None or self.dis_prev[i] is None:
                reward = 0
            else:
                if self.leader[i]:
                    # todo: design leader reward
                    dis2obs = np.array(self.message[i][-16::2])
                    distance_chg = self.dis_rel[i] - self.dis_prev[i]
                    rot = utils.get_rotation(self.robot[i])
                    ang2goal = self.ang_rel[i]
                    reward = -0.3

                    idx = np.where(np.min(dis2obs))[0][0]
                    if dis2obs[idx] < 1.5:
                        reward -= (1/max(dis2obs[idx], 0.5)-1/1.5)
                    reward -= 100 * distance_chg * abs(np.cos(ang2goal))


                else:
                    # todo: design follower reward
                    dis2obs = np.array(self.message[i][-16::2])

                    reward = 0
                    idx = np.where(np.min(dis2obs))[0][0]
                    if dis2obs[idx] < 1.5:
                        reward -= (1 / max(dis2obs[idx], 0.5) - 1 / 1.5)
                    reward -= np.linalg.norm(self.err[i])

            reward_n.append(reward)
        return reward_n

    def is_done(self):
        self.steps += 1
        distance = utils.get_distance_from_target(self.robot[0], self.target)

        if distance < self.findThreshold:
            print("======== + Solved + ========")
            return True

        if self.steps > self.steps_threshold or self.should_done:
            return True

        return False

    def reset(self):
        print("Reset simulation")
        self.respawnRobot()
        self.steps = 0
        self.should_done = False
        self.message = None
        return self.observation

    def get_info(self):
        pass

    def respawnRobot(self):
        """
        This method reloads the saved CartPole robot in its initial state from the disk.
        """
        # Despawn existing robot
        for i in range(self.n):
            if self.robot[i] is not None:
                self.robot[i].remove()

        # Respawn robot in starting position and state
        rootNode = self.supervisor.getRoot(
        )  # This gets the root of the scene tree
        childrenField = rootNode.getField(
            'children'
        )  # This gets a list of all the children, ie. objects of the scene

        for i in range(self.n):
            childrenField.importMFNode(-2, self.robot_name[i]+".wbo")  # Load robots from file and add to last position

        # Get the new robot and pole endpoint references
        self.robot = [self.supervisor.getFromDef(self.robot_name[i]) for i in range(self.n)]
        self.leader = [('LEADER' in self.robot_name[i]) for i in range(self.n)]  # Assume first robot node is the leader
        self.target = self.supervisor.getFromDef(self.target_name)
        self.trans_field = [self.robot[i].getField("translation") for i in range(self.n)]
        self.rot_field = [self.robot[i].getField("rotation") for i in range(self.n)]
        # Reset the simulation physics to start over
        # getSFRotation will fail if using after getSFVec3f
        #self.rot_field[0].setSFRotation(ang_init[0])
        #leader_pos = self.trans_field[0].getSFVec3f()
        leader_pos = self.check_avail(self.robot[0], self.trans_field[0])
        target_coor = self.target.getField('translation').getSFVec3f()
        x_r = (target_coor[0] - leader_pos[0])
        z_r = (target_coor[2] - leader_pos[2])
        formation_rot = np.pi - math.atan2(x_r, z_r)
        ang_init = [0, -1, 0, formation_rot]
        self.rot_field[0].setSFRotation(ang_init)
        leader_rot = utils.get_rotation(self.robot[0])
        pos_init = self.formation_init(leader_pos, leader_rot)

        for i in range(self.n):
            ang_init = [0, -1, 0, formation_rot+np.random.uniform(-np.pi / 6, np.pi / 6)]
            self.rot_field[i].setSFRotation(ang_init)
            self.trans_field[i].setSFVec3f(pos_init[i])

        # Reset the simulation physics to start over
        self.supervisor.simulationResetPhysics()

        self._last_message = None

    def formation_init(self, leader_pos, leader_rot):
        print("Reset formation")
        formation = [[] for _ in range(self.n)]
        desire_dis = [4] * self.n
        desire_ang = [0.0] * self.n
        shape = np.random.randint(0, 2) # 0 for straight line, 1 for V shape, 2 for polygon
        if shape == 0:
            formation[0] = leader_pos
            for i in range(1, self.n):
                desire_ang[i] += 0
                pos = [i * desire_dis[i] * np.sin(desire_ang[i]), i * desire_dis[i] * np.cos(desire_ang[i])] # [x,z], x is horizontal and z is forward
                new_pos = self.rotation_2d(pos, leader_rot)
                formation[i] = [leader_pos[0] + new_pos[0],
                                leader_pos[1],
                                leader_pos[2] + new_pos[1]]
        elif shape == 1:
            formation[0] = leader_pos
            for i in range(1, self.n):
                if self.n % 2 - 1 == 0:
                    # form a V shape if even followers
                    desire_ang[i] += (np.pi / 6) * (-1) ** i
                    pos = [np.ceil(i / 2) * desire_dis[i] * np.sin(desire_ang[i]),
                           np.ceil(i / 2) * desire_dis[i] * np.cos(desire_ang[i])]
                else:
                    # form a diamond shape if odd followers
                    desire_ang[i] += np.pi + (i//2)*(np.pi / (self.n/2+1)) * (-1) ** np.ceil(i / 2)
                    pos = [-(self.n // 2) * desire_dis[i] * np.sin(desire_ang[i]),
                           -(self.n // 2) * desire_dis[i] * np.cos(desire_ang[i])]
                new_pos = self.rotation_2d(pos, leader_rot)
                formation[i] = [leader_pos[0] + new_pos[0],
                                leader_pos[1],
                                leader_pos[2] + new_pos[1]]
        elif shape == 2:
            formation[0] = leader_pos
            for i in range(1, self.n):
                desire_ang[i] += (i-1)*2*np.pi/(self.n-1)
                pos = [desire_dis[i] * np.sin(desire_ang[i]), desire_dis[i] * np.cos(desire_ang[i])]
                new_pos = self.rotation_2d(pos, leader_rot)
                formation[i] = [
                    leader_pos[0] + new_pos[0],
                    leader_pos[1],
                    leader_pos[2] + new_pos[1]]
        return formation

    def check_avail(self, robot, trans_field):
        robot_pos = trans_field.getSFVec3f()
        robot_pos = [np.random.uniform(-30, 30), robot_pos[1], np.random.uniform(-30, 30)]
        trans_field.setSFVec3f(robot_pos)

        obs = [self.supervisor.getFromDef("OBSTACLE_"+str(i+1)) for i in range(arglist.num_obstacle)]
        obs.append(self.supervisor.getFromDef(self.target_name))
        obs += [self.robot[i] for i in range(self.n) if self.robot[i] != robot]

        for i in range(len(obs)):
            if utils.get_distance_from_target(robot, obs[i]) < 1.5:
                trans = obs[i].getField("translation")
                obs_pos = trans.getSFVec3f()
                robot_pos[0] = min(max(obs_pos[0] + np.random.choice((2, -2)), -30), 30)
                robot_pos[2] = min(max(obs_pos[2] + np.random.choice((2, -2)), -30), 30)
                break
        return robot_pos

    def rotation_2d(self, pos, ang):
        # rotate by y-axis, therefore rotate matrix is [[cos,sin],[-sin,cos]]
        rot_mat = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
        pos = np.array(pos)
        new_pos = rot_mat.dot(pos)
        return new_pos.tolist()

    def get_rel_pos(self):
        goal = self.target
        leader = self.robot[0]
        for n in range(self.n):
            if n == 0:
                self.dis_rel[n] = utils.get_distance_from_target(self.robot[n], goal)
                self.ang_rel[n] = utils.get_angle_from_target(self.robot[n], goal, is_true_angle=True, is_abs=False)
            else:
                self.dis_rel[n] = utils.get_distance_from_target(leader, self.robot[n])
                self.ang_rel[n] = utils.get_angle_from_target(leader, self.robot[n], is_true_angle=True, is_abs=False)
            if self.dis_des:
                err_x = (self.dis_rel[n] * np.sin(self.ang_rel[n])) - self.dis_des[n] * np.sin(self.ang_des[n])
                err_z = (self.dis_rel[n] * np.cos(self.ang_rel[n])) - self.dis_des[n] * np.cos(self.ang_des[n])
                self.err[n] = [err_x, err_z]
            else:
                self.err[n] = [0, 0]

    def get_agents_pos(self, robot):
        dis = []
        ang = []
        for n in range(self.n):
            if self.robot[n] != robot:
                dis.append(min(utils.get_distance_from_target(robot, self.robot[n]), 10))
                ang.append(utils.get_angle_from_target(robot, self.robot[n]))
        return dis, ang

    def get_obstacle_type(self, message, robot):
        dis, ang = self.get_agents_pos(robot)
        obs_type = [0.0]
        # if obstacle detected
        if message[0] < 2.5:
            for i in range(len(dis)):
                if abs(ang[i] - message[1]) > np.pi/20:
                    obs_type = [1.0]
                else:
                    # if the distance to other agent is less than the obstacle in same direction
                    if dis[i] <= message[0]:
                        obs_type = [0.0]
                        break
        return obs_type

    def set_des_pos(self):
        self.dis_des = None
        self.ang_des = None
        if not self.dis_des:
            self.dis_des = self.dis_rel.copy()
        if not self.ang_des:
            self.ang_des = self.ang_rel.copy()

robot_dict = ['ROBOT_LEADER', 'ROBOT_FOLLOWER_1', 'ROBOT_FOLLOWER_2', 'ROBOT_FOLLOWER_3', 'ROBOT_FOLLOWER_4']
robot_name = robot_dict[:arglist.num_agent]
target_name = 'TARGET'
supervisor_pre = FindTargetSupervisor(robot_name, 'TARGET', observation_space=[arglist.leader_observation + (arglist.num_agent-1)*2, arglist.follower_observation + (arglist.num_agent-1)*2])
supervisor_env = KeyboardPrinter(supervisor_pre)
'''supervisor_env = TensorboardLogger(supervisor_key,
                                   log_dir="logs/results/ddpg",
                                   v_action=1,
                                   v_observation=1,
                                   v_reward=1,
                                   windows=[10, 100, 200])'''

score_history = []

tf.set_random_seed(0)
episode = 0
best_rew = None
if not arglist.manual_control:
    if arglist.use_tf:
        with U.single_threaded_session():
            if arglist.good_policy == "ddpg":
                obs_space = []
                obs_shape_n = []
                act_space = []
                act_space.append(spaces.Discrete(arglist.action))
                for i in range(2):
                    if i == 0:
                        obs_dim = arglist.leader_observation + (arglist.num_agent-1)*2
                    else:
                        obs_dim = arglist.follower_observation + (arglist.num_agent-1)*2
                    obs_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
                    obs_shape_n.append(obs_space[i].shape)
                trainers = get_trainers(obs_shape_n, act_space, arglist)
                saver = tf.train.Saver()

                U.initialize()

                # Load previous results, if necessary
                if arglist.load_dir == "":
                    arglist.load_dir = arglist.tf_save_dir
                if arglist.display or arglist.restore or arglist.benchmark:
                    print('Loading previous state...')
                    U.load_state(arglist.load_dir, saver)
                episode_step = [0]
                final_ep_steps = []
                episode_done = [0]
                final_ep_done = []
                train_step = 0

                episode_rewards = [0.0]  # sum of rewards for all agents
                agent_rewards = [[0.0] for _ in range(arglist.num_agent)]  # individual agent reward
                final_ep_rewards = []  # sum of rewards for training curve
                final_ep_ag_rewards = []  # agent rewards for training curve
                episode_crash = [0]  # sum of crashes for all agents
                final_ep_crash = []  # sum of crashes for training curve
                agent_info = [[[]]]  # placeholder for benchmarking info

                score = 0
                obs_ = supervisor_env.reset()
                obs_n = []
                for n in range(supervisor_pre.n):
                    obs_n.append(np.array(list(map(float, obs_[n]))))
                supervisor_pre.empty_queue()
                act_n = [np.zeros(arglist.action).tolist() for n in range(len(obs_n))]
                _, _, done, _ = supervisor_env.step(act_n)
                supervisor_pre.set_des_pos()
                print('Starting iterations...')

                while True:
                    if arglist.network == "MLP":
                        # get action
                        if arglist.good_policy == "maddpg":
                            act_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                        else:
                            act_n = [trainers[obs != 0].action(obs_n[obs], episode) for obs in range(len(obs_n))]
                        new_obs, reward_n, done, info_n = supervisor_env.step(act_n)
                        new_obs_n = []
                        for n in range(supervisor_pre.n):
                            new_obs_n.append(np.array(list(map(float, new_obs[n]))))
                        episode_step[-1] += 1
                        terminal = (episode_step[-1] >= arglist.max_episode_len)
                        # collect experience
                        if arglist.good_policy == "ddpg":
                            for i in range(len(obs_n)):
                                trainers[i != 0].experience(obs_n[i], act_n[i], reward_n[i], new_obs_n[i], done, terminal)
                        else:
                            for i, agent in enumerate(trainers):
                                agent.experience(obs_n[i], act_n[i], reward_n[i], new_obs_n[i], done, terminal)

                        obs_n = new_obs_n
                        for i, rew in enumerate(reward_n):
                            episode_rewards[-1] += rew
                            agent_rewards[i][-1] += rew

                        episode_done[-1] += done

                        if done or terminal:
                            episode += 1
                            obs_ = supervisor_env.reset()
                            obs_n = []
                            for n in range(supervisor_pre.n):
                                obs_n.append(np.array(list(map(float, obs_[n]))))
                            episode_step.append(0)
                            episode_rewards.append(0)
                            episode_crash.append(0)
                            episode_done.append(0)
                            for a in agent_rewards:
                                a.append(0)
                            agent_info.append([[]])
                    # increment global step counter
                    train_step += 1

                    # for benchmarking learned policies
                    if arglist.benchmark:
                        for i, info in enumerate(info_n):
                            agent_info[-1][i].append(info_n['n'])
                        if train_step > arglist.benchmark_iters and (done or terminal):
                            file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                            print('Finished benchmarking, now saving...')
                            with open(file_name, 'wb') as fp:
                                pickle.dump(agent_info[:-1], fp)
                            break
                        continue

                    # update all trainers, if not in display or benchmark mode
                    if not arglist.display:
                        if arglist.good_policy == "ddpg":
                            for i in range(len(obs_n)):
                                trainers[i != 0].preupdate()
                                loss = trainers[i != 0].update(trainers, train_step)
                        # save model, display training output
                        if (done or terminal) and (len(episode_rewards) % arglist.save_rate == 0):
                            final_reward = np.mean(episode_rewards[-arglist.save_rate:])
                            final_step = np.mean(episode_step[-arglist.save_rate:])
                            final_crash = np.mean(episode_crash[-arglist.save_rate:])
                            final_done = np.mean(episode_done[-arglist.save_rate:])
                            U.save_state(arglist.tf_save_dir, saver=saver)
                            best_rew = final_reward
                            now = datetime.now()
                            print("model saved at ", now)
                            '''
                            if not best_rew:
                                best_rew = final_reward
                            else:
                                if final_reward > best_rew:
                                    U.save_state(arglist.tf_save_dir, saver=saver)
                                    best_rew = final_reward
                                    now = datetime.now()
                                    print("model saved at ", now)
                            '''
                            # print statement depends on whether or not there are adversaries
                            print(
                                "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}".format(
                                    train_step, len(episode_rewards), final_reward,
                                    [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards]))
                            # Keep track of final episode reward
                            final_ep_rewards.append(final_reward)
                            final_ep_steps.append(final_step)
                            final_ep_crash.append(final_crash)
                            final_ep_done.append(final_done)
                            for rew in agent_rewards:
                                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
                        # saves final episode reward for plotting training curve later
                        if len(episode_rewards) > arglist.num_episodes:
                            rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards_formation_navigation_1.pkl'
                            with open(rew_file_name, 'wb') as fp:
                                pickle.dump(final_ep_rewards, fp)
                            agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards_formation_navigation_1.pkl'
                            with open(agrew_file_name, 'wb') as fp:
                                pickle.dump(final_ep_ag_rewards, fp)
                            step_file_name = arglist.plots_dir + arglist.exp_name + '_steps_formation_navigation_1.pkl'
                            with open(step_file_name, 'wb') as fp:
                                pickle.dump(final_ep_steps, fp)
                            done_file_name = arglist.plots_dir + arglist.exp_name + '_done_formation_navigation_1.pkl'
                            with open(done_file_name, 'wb') as fp:
                                pickle.dump(final_ep_done, fp)
                            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                            break

    else:
        agent = []

        agent.append(DDPG(name="leader",
                          lr_actor=arglist.lr_actor,
                          lr_critic=arglist.lr_critic,
                          obs_dims=arglist.leader_observation,
                          gamma=arglist.gamma,
                          tau=arglist.tau,
                          env=supervisor_env,
                          batch_size=arglist.batch_size,
                          layer1_size=arglist.layer1_size,
                          layer2_size=arglist.layer2_size,
                          n_actions=arglist.action,
                          load_models=arglist.restore,
                          save_dir=arglist.save_dir))

        agent.append(DDPG(name="follower",
                          lr_actor=arglist.lr_actor,
                          lr_critic=arglist.lr_critic,
                          obs_dims=arglist.follower_observation,
                          gamma=arglist.gamma,
                          tau=arglist.tau,
                          env=supervisor_env,
                          batch_size=arglist.batch_size,
                          layer1_size=arglist.layer1_size,
                          layer2_size=arglist.layer2_size,
                          n_actions=arglist.action,
                          load_models=arglist.restore,
                          save_dir=arglist.save_dir))

        for i in range(1, arglist.num_episodes):
            score = 0
            obs_ = supervisor_env.reset()
            obs_n = []
            for n in range(supervisor_pre.n):
                obs_n.append(list(map(float, obs_[n])))
            #obs = list(map(float, supervisor_env.reset()))
            supervisor_pre.empty_queue()
            act_n = [np.zeros(arglist.action).tolist() for n in range(len(obs_n))]
            _, _, done, _ = supervisor_env.step(act_n)
            supervisor_pre.set_des_pos()

            if arglist.display:
                print("================= TESTING =================")
                step = 0
                while not done:
                    act_n = [agent[n != 0].choose_action_train(obs_n[n]).tolist() for n in range(len(obs_n))]
                    new_obs_n, reward_n, done, info_n = supervisor_env.step(act_n)
                    for n in range(len(obs_n)):
                        obs_n[n] = list(map(float, new_obs_n[n]))
                    step += 1
            else:
                print("================= TRAINING =================")
                step = 0
                while not done:
                    act_n = [agent[n != 0].choose_action_train(obs_n[n]).tolist() for n in range(len(obs_n))]
                    new_obs_n, reward_n, done, info_n = supervisor_env.step(act_n)
                    for n in range(len(obs_n)):
                        #print(obs_n[n], act_n[n], reward_n[n], new_obs_n[n], int(done_n))
                        agent[n != 0].remember(obs_n[n], act_n[n], reward_n[n], new_obs_n[n], int(done))
                        score += reward_n[n]
                        obs_n[n] = list(map(float, new_obs_n[n]))
                    step += 1
                    total_step += 1
                    if total_step % 100 == 0:
                        for n in range(len(obs_n)):
                            loss = agent[n != 0].learn()
                            if loss:
                                pass
                                #print("critic loss", loss[0], "actor loss", loss[1])
                score_history.append(np.mean(score))
                if i % 10 == 0:
                    final_rew = np.mean(score_history[-10:])
                    if not best_rew:
                        best_rew = final_rew
                    else:
                        if best_rew < final_rew:
                            now = datetime.now()
                            print("model saved at ", now)
                            for a in agent:
                                a.save_models()
                            best_rew = final_rew
                    print("===== Episode", i, "total steps", total_step, "10 game average %.2f" % final_rew)
else:
    obs_ = supervisor_env.reset()
    obs_n = []
    for n in range(supervisor_pre.n):
        obs_n.append(list(map(float, obs_[n])))
    supervisor_pre.empty_queue()
    act_n = [np.zeros(arglist.action).tolist() for n in range(len(obs_n))]
    _, _, done, _ = supervisor_env.step(act_n)
    supervisor_pre.set_des_pos()
    while 1:
        new_state, reward, done, info = supervisor_env.manual_control()