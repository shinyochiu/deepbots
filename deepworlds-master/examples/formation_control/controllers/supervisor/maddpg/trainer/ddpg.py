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

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None, is_training=True):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[0].sample_placeholder([None], name="action"+str(0))]
        p_input = obs_ph_n[0]

        p = p_func(p_input, int(act_pdtype_n[0].param_shape()[0]), scope="p_func", num_units=num_units, is_training=is_training)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        # wrap parameters in distribution
        act_pd = act_pdtype_n[0].pdfromflat(p)
        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        #p_reg = tf.reduce_mean(tf.square(p))

        act_input_n = act_ph_n + []
        act_input_n[0] = act_pd.sample()
        #act_input_n[0] = p
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units, is_training=is_training)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3
        #loss = p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)
        #optimize_expr = U.flatgrad(optimizer, loss, p_func_vars, grad_norm_clipping)
        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[0]], outputs=act_sample)
        #act = U.function(inputs=[obs_ph_n[0]], outputs=tf.nn.softmax(p, dim=-1))
        p_values = U.function([obs_ph_n[0]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[0].param_shape()[0]), scope="target_p_func", num_units=num_units, is_training=is_training)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[0].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[0]], outputs=target_act_sample)
        #target_act = U.function(inputs=[obs_ph_n[0]], outputs=tf.nn.tanh(target_p))

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64, is_training=True):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[0].sample_placeholder([None], name="action"+str(0))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units, is_training=is_training)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)
        #optimize_expr = U.flatgrad(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units, is_training=is_training)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class DDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False, param_noise=None):
        # only 1 agent's observations contained
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.is_training = (args.display == False)
        self.leader = None#("_0" in self.name)
        obs_ph_n = []
        if self.leader:
            for i in range(self.n):
                obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())
        else:
            obs_ph_n.append(U.BatchInput(obs_shape_n[agent_index!=0], name="observation" + str(agent_index!=0)).get())
            act_space_n = [act_space_n[0]]

        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        self.act_space_num = int(act_pdtype_n[0].param_shape()[0])
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            is_training=self.is_training
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr*0.1),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            is_training=self.is_training
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        #self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.max_replay_buffer_len = args.max_episode_len
        self.replay_sample_index = None

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def constraint(self, obs):
        con_val = 0
        alpha = 5
        beta = 2
        dis2obs = obs[None][0][-16::2]
        idx = 0
        for i in range(len(dis2obs)):
            if dis2obs[i] < dis2obs[idx]:
                idx = i
        # calculate distance constraint
        #con_val += alpha * np.exp(-beta * dis2obs[idx])
        return [con_val]

    def action(self, obs, episode):
        def safe_policy(obs, act):
            dis2goal = obs[None][0][0]
            ang2goal = obs[None][0][1]
            dis2obs = obs[None][0][-16::2]
            ang2obs = obs[None][0][-15::2]
            idx = np.where(dis2obs == np.min(dis2obs))[0][0]
            if dis2goal < 5 or np.min(dis2obs) > 0.5:
                return act
            else:
                # apply save potential field policy when the state is diverging
                if np.min(dis2obs) < 0.5 and abs(ang2obs[idx] - np.pi/2) < np.pi/8:
                    act[0] = 0.
                    act[1] = 1.0 - (ang2goal > 0)
                    act[2] = float(ang2goal > 0)
                else:
                    act[0] = max(0., np.cos(ang2goal))
                    act[1] = ((ang2goal > 0) - 1) * ang2goal / np.pi
                    act[2] = (ang2goal > 0) * ang2goal / np.pi
            return act

        if self.is_training:
            return self.softmax(np.random.uniform(0, 1, self.act_space_num)) if episode < 5000 else self.act(obs[None])[0]
        else:
            return self.act(obs[None])[0]
        #return self.act(obs[None])[0]

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
        rew = []
        done = []
        # train q network
        num_sample = 1
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        obs_n.append(obs)
        obs_next_n.append(obs_next)
        act_n.append(act)
        target_q = 0.0
        for j in range(num_sample):
            target_act_next_n = [self.p_debug['target_act'](obs_next_n[0])]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))
        # train p network
        p_loss = self.p_train(*(obs_n + act_n))
        if t % 10000 == 0 and self.agent_index:
            print("steps:{} q loss:{} p loss:{}".format(t, q_loss, p_loss))
        self.p_update()
        self.q_update()
    #print(q_loss,p_loss)
        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
