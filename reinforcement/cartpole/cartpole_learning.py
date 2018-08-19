import gym
import math
import collections as c
import random
import numpy as np
import tensorflow as tf

"""
    Step List Indices Lookup:
        0:  Obs
        1:  Reward
        2:  Done (termination)
        3:  info
        4:  action
        5:  cum_reward

    Action Lookup:
        1:  Left
        2:  Right
"""

# Encode Step Lookup (above)
obs = 0
reward = 1
term = 2
info = 3
prev_obs = 4
action = 5
cum_reward = 6
num_steps = 7

# Encode Action Lookup
left = 1
right = 0

# Encode tf ops
tr = 0
wr = 1
sm = 2
su = 3
sv = 4

# Debug Constants
PENALTY = -10
SEED = 742890

TRAIN_STEPS = 5000
TRAIN_SAMPLES = 100
TRAIN_INTERVAL = 10
TRAIN_THRESHOLD = 180

EPISODES = 15000
RECENCY = TRAIN_INTERVAL

GAMMA = 0.85
EPSILON_DECAY = 0.99

MODEL_TYPE='ed_ts'
LOGFILE='./tensorboard/model_' + MODEL_TYPE
MODEL_SAVE='./model/model_' + MODEL_TYPE

#Globals
stop_train = False
num_episodes = 0
epsilon = 0.99
train_examples = []
recent_episodes = c.deque(maxlen = RECENCY)

# Run Simulations
def run_simulation(env, output_layer, max_steps, rand):
    steps = []
    state = env.reset()

    for i in range(max_steps):
        env.render()
        (next_action, action_val) = get_next_action(env, state, output_layer, rand)

        step = list(env.step(action_val))

        step.append(state)
        step.append(next_action)
        steps.append(step)

        done = step[term]
        state = step[obs]

        if done:
            break

    return steps

def update_epsilon():
    global epsilon
    epsilon *= EPSILON_DECAY
    print("Updated epsilon:", epsilon)

def steps_per_ep():
    global stop_train
    step_lens = []

    for steps in recent_episodes:
        step_lens.append(len(steps))

    mean = sum(step_lens) / float(len(step_lens))
    stop_train = mean > TRAIN_THRESHOLD

    return step_lens

def run_and_store_simulation(env, output_layer, max_steps, tf_objs):
    global num_episodes
    global recent_episodes

    num_episodes += 1
    steps = []

    writer = tf_objs[wr]
    avg_steps_summ = tf_objs[sm]

    print("Running Trained Simulation")
    steps = run_simulation(env, output_layer, max_steps, False)

    if num_episodes % TRAIN_INTERVAL == 0:
        update_epsilon()
        step_lens = steps_per_ep()
        summ = avg_steps_summ.eval(feed_dict = {'S:0':step_lens})
        writer.add_summary(summ, num_episodes)

    return steps

# Train Model Routine
def train_model(env, output_layer, tf_objs, max_steps=1000):
    print("Running Simulations")
    for i in range(EPISODES):
        print("")
        print("Episode", i + 1)

        steps = run_and_store_simulation(env, output_layer, max_steps, tf_objs)
        calculate_cum_reward(steps)
        update_coef(steps, tf_objs)

def neural_layer(input_mat, input_neurons, output_neurons, hidden=True):
    weight = weight_variable([input_neurons, output_neurons])
    bias = bias_variable([output_neurons])

    tf.summary.histogram('Weight', weight)
    tf.summary.histogram('Bias', bias)

    output = tf.matmul(input_mat, weight) + bias
    if hidden:
        output = tf.sigmoid(output)
    return output


# Set up model
def setup_model(alpha=0.01, lamda=1.0, hparam=''):
    print("Setting Up Model ...")

    # input layer
    input_ph = tf.placeholder("float", [None, 4], 'X')
    action_ph = tf.placeholder("float", [None, 2], 'A')
    reward_ph = tf.placeholder("float", [None, 1], 'R')
    step_lens_ph = tf.placeholder("float", [None], 'S')

    # Writer
    writer = tf.summary.FileWriter(LOGFILE + hparam)

    with tf.name_scope('output'):
        output_layer = neural_layer(input_ph, 4, 2, hidden=False)

    # Regularization/Loss
    #with tf.name_scope('regularization'):
    #    regularizer = tf.contrib.layers.l2_regularizer(scale=lamda)
    #    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

    with tf.name_scope('loss'):
        q_pred = tf.reduce_sum(tf.multiply(output_layer, action_ph), axis=[1])
        loss = tf.reduce_mean(tf.squared_difference(reward_ph, q_pred))

    #Train/Test Data
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(alpha).minimize(loss)

    with tf.name_scope('accuracy'):
        error_ts = tf.reduce_mean(q_pred - reward_ph)
        error = tf.summary.scalar("error", error_ts)

    with tf.name_scope('step_len'):
        avg_steps = tf.reduce_mean(step_lens_ph)
        avg_steps_summ = tf.summary.scalar("avg_steps", avg_steps, collections=["STEPS"])

    #Housekeeping
    writer.add_graph(sess.graph)
    summ_vars = tf.summary.merge_all()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    return (output_layer, [train_step, writer, avg_steps_summ, summ_vars, saver])


# Helpers
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    collections = [tf.GraphKeys.REGULARIZATION_LOSSES, tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.Variable(initial, collections=collections, name='weights')

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    collections = [tf.GraphKeys.REGULARIZATION_LOSSES, tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.Variable(initial, name='bias')

def expected_reward(state, output_layer):
    state_rshp = np.reshape(state, [1, 4])
    return output_layer.eval(feed_dict={'X:0':state_rshp})[0]

def reshape_examples(s, n):
    input_ = np.reshape([b[0] for b in s], [n, -1])
    action = np.reshape([b[1] for b in s], [n, -1])
    reward = np.reshape([b[2] for b in s], [n, -1])

    return (input_, action, reward)

def run_train_step(train_step, writer, summ_vars):
    global train_examples
    print("Training Model ... ")
    for i in range(TRAIN_STEPS):
        num_samples = min(len(train_examples), TRAIN_SAMPLES)
        batch = list(random.sample(train_examples, num_samples))

        (input_b, action_b, reward_b) = reshape_examples(batch, len(batch))
        feed_dict = {'X:0':input_b, 'A:0':action_b, 'R:0':reward_b}

        train_step.run(feed_dict = feed_dict)

    summ = summ_vars.eval(feed_dict = feed_dict)
    writer.add_summary(summ, num_episodes)

    print("Done Training Model")

def update_coef(steps, tf_objs):
    # Global examples variables
    global train_examples
    global recent_episodes
    global stop_train

    # Unpack tf_objs
    train_step = tf_objs[tr]
    writer = tf_objs[wr]
    summ_vars = tf_objs[su]
    saver = tf_objs[sv]

    for step in steps:
        train_examples.append((step[prev_obs], step[action], step[cum_reward]))
    recent_episodes.append(steps)

    if not(stop_train) and num_episodes % TRAIN_INTERVAL == 0:
        run_train_step(train_step, writer, summ_vars)

        path = saver.save(sess, MODEL_SAVE)
        print('Saved at path \'%s\'' % path)


def train_examples_ordered_insert(step):
    global train_examples
    i = 0
    if len(train_examples) > 0:
        while step[num_steps] <= train_examples[i][3]:
            i += 1
            if i == len(train_examples):
                break

def get_next_action(env, state, output_layer, rand):
    global epsilon

    rand = rand or (epsilon > np.random.uniform())
    next_action = np.zeros([1, 2])
    action_val = 0

    if rand:
        action_val = env.action_space.sample()
    else:
        y = expected_reward(state, output_layer)
        action_val = np.argmax(y)

    next_action[0][action_val] = 1
    return (next_action, action_val)

def get_adj_action(curr_idx, last_idx, steps):
    if curr_idx >= len(steps) or steps[curr_idx][cum_reward] <= 0:
        if not(last_idx):
            last_idx = curr_idx
        next_action = (steps[last_idx][action] + 1) % 2
    else:
        next_action = steps[curr_idx][action]

    action_val = right
    if(next_action[0][left]):
        action_val = left

    return (next_action, action_val, last_idx)

def calculate_cum_reward(steps):
    tot_reward = 0
    for i,step in enumerate(reversed(steps)):
        j = len(steps) - 1 - i
        if tot_reward == 0:
            tot_reward = (PENALTY * math.pow(GAMMA, j))
        else:
            tot_reward += (step[reward] * math.pow(GAMMA, j))
        step.append(tot_reward / 100)

def make_hparam(alpha, lamda):
    return 'lr_%.0e,rg_%.0e' % (alpha, lamda)

# Run and Print Simulation
def run_and_print_sim(env, max_steps=100):
    print("Random Simulation")
    steps = run_simulation(env, [], max_steps, True)
    print_state_reward(steps)

def print_state_reward(steps):
    for (i, step) in enumerate(steps):
        print("State: ", step[obs])
        print("Next Action: ", step[action])
        print("Reward: ", step[reward])
        print("Cumulative: ", step[cum_reward])
        print("")


# Main Run
env = gym.make('CartPole-v0')
#run_and_print_sim(env)

lamda = 10.0
alpha = math.pow(10,-4)

print("Starting Training Session for alpha = %.0e, lamda = %.0e, model type: %s" % (alpha, lamda, MODEL_TYPE))
sess = tf.InteractiveSession()

tf.set_random_seed(SEED)

(output_layer, tf_objs) = setup_model(alpha, lamda)
train_model(env, output_layer, tf_objs)

sess.close()
tf.reset_default_graph()
num_episodes = 0

env.close()
