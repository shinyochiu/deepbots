import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, target_ph_n, p_func, optimizer, grad_norm_clipping=None, scope="trainer", reuse=None, num_units=64, is_training=True):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        obs_ph_n = make_obs_ph_n
        act_pdtype_n = [make_pdtype(obs_space) for obs_space in act_space_n]

        # set up placeholders
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = tf.concat(obs_ph_n + act_ph_n, 1)
        p = p_func(p_input, int(len(p_input)), scope="pred_func", num_units=num_units, is_training=is_training)[:,0]
        p_func_vars = U.scope_vars(U.absolute_scope_name("pred_func"))

        p_loss = tf.reduce_mean(tf.square(p - target_ph_n))

        # viscosity solution to Bellman differential equation in place of an initial condition
        p_reg = tf.reduce_mean(tf.square(p))
        loss = p_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)
        #optimize_expr = U.flatgrad(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph_n], outputs=loss, updates=[optimize_expr])
        pred = U.function(inputs=[obs_ph_n[0]]+act_ph_n, outputs=p)
        pred_values = U.function(obs_ph_n + act_ph_n, pred)

        '''# target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units, is_training=is_training)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)'''

        return pred, train, {'q_values': pred_values}


class StatePredictTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args):
        # only 1 agent's observations contained
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.is_training = (args.display == False)
        self.leader = ("_0" in self.name)
        obs_ph_n = []
        target_ph_n = []
        obs_ph_n.append(U.BatchInput(obs_shape_n[self.leader != 0], name="observation" + str(0)).get())
        target_ph_n.append(U.BatchInput(obs_shape_n[self.leader != 0], name="target" + str(0)).get())

        # Create all the functions necessary to train the model
        self.pred, self.p_train, self.p_debug = p_train(
            scope=name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            target_ph_n=target_ph_n,
            p_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            is_training=self.is_training
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def pred(self, input):
        return self.pred(input[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, trainer, t):
        if len(self.replay_buffer) > self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 10 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        rew = []
        done = []
        #obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        # train q network
        num_sample = 1
        obs, act, rew, obs_next, done = trainer[-1].replay_buffer.sample_index(index)
        obs_n.append(obs)
        obs_next_n.append(obs_next)
        act_n.append(act)
        p_loss = self.p_train(*(obs_n + act_n + obs_next_n))
        # train p network
        if t % 10000 == 0:
            print("steps:{} q loss:{} p loss:{}".format(t, p_loss))
        return [p_loss]
