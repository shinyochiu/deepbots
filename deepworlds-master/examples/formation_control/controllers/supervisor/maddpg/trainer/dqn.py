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

def maxQ_act(obs, act_space_num, q_debug):
    if len(obs) == 16:
        obs = np.transpose(np.expand_dims(obs, axis=-1))
    else:
        obs = np.squeeze(obs)
    q_value = np.zeros((obs.shape[0], act_space_num))
    for act in range(act_space_num):
        temp_act = np.zeros((obs.shape[0], act_space_num))
        if obs.shape[0] > 1:
            temp_act[:][act] = 1
            q_values = q_debug['q_values'](*([obs] + [temp_act]))
            for q in range(len(q_value)):
                q_value[q][act] = q_values[q]
        else:
            temp_act[0][act] = 1
            q_value[0][act] = q_debug['q_values'](*([obs] + [temp_act]))
    q_value = np.argmax(q_value, axis=1)
    policy_act = np.zeros((obs.shape[0], act_space_num))
    for q in range(len(q_value)):
        policy_act[q][q_value[q]] = 1

    return policy_act if obs.shape[0] > 1 else np.squeeze(policy_act)

def q_train(make_obs_ph_n, act_space_n, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[0].sample_placeholder([None], name="action")]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class DQNAgentTrainer(AgentTrainer):
    def __init__(self, model, obs_shape_n, act_space_n, agent_num, args, local_q_func=False):
        # only 1 agent's observations contained
        self.n = len(obs_shape_n)
        self.act_space = act_space_n
        act_pdtype_n = [make_pdtype(act_space) for act_space in self.act_space]
        self.act_space_num = int(act_pdtype_n[0].param_shape()[0])
        self.agent_num = agent_num
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope="ddpg",
            make_obs_ph_n=obs_ph_n,
            act_space_n=self.act_space,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs, episode):
        #print(0.6+np.random.rand()*episode/25000)
        if episode < 30000:
            epsilon = np.random.rand()
        else:
            epsilon = 1
        rand_act = np.zeros(self.act_space_num)
        rand_act[np.random.randint(0, self.act_space_num-1)] = 1

        return maxQ_act(obs, self.act_space_num, self.q_debug) if epsilon > 0.6 else rand_act

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, trainer, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        obs, act, rew, obs_next, done = trainer[0].replay_buffer.sample_index(index)
        obs_n.append(obs)
        obs_next_n.append(obs_next)
        act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = maxQ_act(obs_next_n, self.act_space_num, self.q_debug)# max Q(s_t+1,a*)
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + [target_act_next_n]))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        self.q_update()

        return [q_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
