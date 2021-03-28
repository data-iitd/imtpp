import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os, pdb, pickle
import decorated_options as Deco
from utils import MAE, ACC
from scipy.integrate import quad
import multiprocessing as MP
import logging
tf.get_logger().setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__EMBED_SIZE = 16
__HIDDEN_LAYER_SIZE = 64 

def_opts = Deco.Options(
    momentum=0.9,
    decay_steps=100,
    decay_rate=0.001,
    l2_penalty=0.001,
    float_type=tf.float32,
    seed=1234,
    scope='IMTPP',
    device_gpu='/gpu:0',
    device_cpu='/cpu:0',
    embed_size=__EMBED_SIZE,
    # Common
    Wem=lambda num_categories: np.random.RandomState(42).randn(num_categories, __EMBED_SIZE) * 0.01,
    Wy=np.ones((__EMBED_SIZE, __HIDDEN_LAYER_SIZE)) * 0.0,
    Wt=np.ones((1, __HIDDEN_LAYER_SIZE)) * 1e-3,

    # Observed
    Wh=np.eye(__HIDDEN_LAYER_SIZE),
    bh=np.ones((1, __HIDDEN_LAYER_SIZE)),
    Vy=lambda num_categories: np.ones((__HIDDEN_LAYER_SIZE, num_categories)) * 0.001,
    Vt=np.ones((__HIDDEN_LAYER_SIZE, 1)) * 0.001,
    Vomt=np.ones((__HIDDEN_LAYER_SIZE, 1)) * 0.001,
    bk=lambda num_categories: np.ones((1, num_categories)) * 0.0,
    # PP
    bt=np.log(1.0),
    wt=1.0,

    # Missing
    Wmh=np.eye(__HIDDEN_LAYER_SIZE),
    bmh=np.ones((1, __HIDDEN_LAYER_SIZE)),
    Wmt=np.ones((1, __HIDDEN_LAYER_SIZE)) * 1e-3,
    Vmy=lambda num_categories: np.ones((__HIDDEN_LAYER_SIZE, num_categories)) * 0.001,
    # PP
    Vat=np.ones((__EMBED_SIZE, 1)) * 0.0,
    Vmot=np.ones((__HIDDEN_LAYER_SIZE, 1)) * 0.001,
    Vmt=np.ones((__HIDDEN_LAYER_SIZE, 1)) * 0.001,
    bmt=np.log(1.0),
    wmt=1.0,

    #Prior
    Wph=np.eye(__HIDDEN_LAYER_SIZE),
    bph=np.ones((1, __HIDDEN_LAYER_SIZE)),
    Wpt=np.ones((1, __HIDDEN_LAYER_SIZE)) * 1e-3,
    Vpt=np.ones((__HIDDEN_LAYER_SIZE, 1)) * 0.001,
    bpt=np.log(1.0),
    wpt=1.0,
)

def softplus(x):
    return np.log1p(np.exp(x))

def quad_func(t, c, w):
    return c * t * np.exp(-w * t + (c / w) * (np.exp(-w * t) - 1))

class IMTPP:
    @Deco.optioned()
    def __init__(self, sess, num_categories, batch_size,
                 learning_rate, momentum, l2_penalty, embed_size,
                 float_type, bptt, seed, scope, decay_steps, decay_rate,
                 device_gpu, device_cpu, cpu_only,
                 Wt, Wem, Wh, bh, wt, Wy, Vy, Vt, Vomt, bk, bt, Wmh, bmh, Wmt, Vmy, Vat, Vmot, Vmt, bmt, wmt, Wph, bph, Wpt, Vpt, wpt, bpt):
        self.HIDDEN_LAYER_SIZE = Wh.shape[0]
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.MOMENTUM = momentum
        self.L2_PENALTY = l2_penalty
        self.EMBED_SIZE = embed_size
        self.BPTT = bptt

        self.NUM_CATEGORIES = num_categories
        self.FLOAT_TYPE = float_type

        self.DEVICE_CPU = device_cpu
        self.DEVICE_GPU = device_gpu

        self.sess = sess
        self.seed = seed
        self.last_epoch = 0

        self.rs = np.random.RandomState(seed + 42)

        with tf.variable_scope(scope):
            with tf.device(device_gpu if not cpu_only else device_cpu):
                self.events_in = tf.placeholder(tf.int32, [None, self.BPTT], name='events_in')
                self.times_in = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_in')
                self.times_miss = tf.placeholder(tf.int32, [None, self.BPTT], name='times_miss')

                self.events_out = tf.placeholder(tf.int32, [None, self.BPTT], name='events_out')
                self.times_out = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_out')

                self.batch_num_events = tf.placeholder(self.FLOAT_TYPE, [], name='bptt_events')

                self.inf_batch_size = tf.shape(self.events_in)[0]

                # Make variables
                with tf.variable_scope('hidden_state'):
                    self.Wt = tf.get_variable(name='Wt',
                                              shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wt))
                    self.Wem = tf.get_variable(name='Wem', shape=(self.NUM_CATEGORIES, self.EMBED_SIZE),
                                               dtype=self.FLOAT_TYPE,
                                               initializer=tf.constant_initializer(Wem(self.NUM_CATEGORIES)))

                    # Observer RNN
                    self.Wh = tf.get_variable(name='Wh', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wh))
                    self.bh = tf.get_variable(name='bh', shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bh))

                    # Missing RNN
                    self.Wmh = tf.get_variable(name='Wmh', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wmh))
                    self.bmh = tf.get_variable(name='bmh', shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bmh))

                    # Prior RNN
                    self.Wph = tf.get_variable(name='Wph', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wph))
                    self.bph = tf.get_variable(name='bph', shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bph))

                with tf.variable_scope('output'):
                    self.wt = tf.get_variable(name='wt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(wt))

                    self.Wy = tf.get_variable(name='Wy', shape=(self.EMBED_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wy))

                    self.Vy = tf.get_variable(name='Vy', shape=(self.HIDDEN_LAYER_SIZE, self.NUM_CATEGORIES),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vy(self.NUM_CATEGORIES)))
                    self.Vt = tf.get_variable(name='Vt', shape=(self.HIDDEN_LAYER_SIZE, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vt))

                    self.Vomt = tf.get_variable(name='Vomt', shape=(self.HIDDEN_LAYER_SIZE, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vomt))

                    self.bt = tf.get_variable(name='bt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bt))
                    self.bk = tf.get_variable(name='bk', shape=(1, self.NUM_CATEGORIES),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bk(num_categories)))

                    # Missing RNN
                    self.Wmt = tf.get_variable(name='Wmt',
                                              shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wmt))

                    self.Vmy = tf.get_variable(name='Vmy', shape=(self.HIDDEN_LAYER_SIZE, self.NUM_CATEGORIES),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vmy(self.NUM_CATEGORIES)))

                    self.wmt = tf.get_variable(name='wmt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(wmt))
                    self.Vmt = tf.get_variable(name='Vmt', shape=(self.HIDDEN_LAYER_SIZE, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vmt))
                    self.Vmot = tf.get_variable(name='Vmot', shape=(self.HIDDEN_LAYER_SIZE, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vmot))
                    self.bmt = tf.get_variable(name='bmt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bmt))
                    self.Vat = tf.get_variable(name='Vat', shape=(self.EMBED_SIZE, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vat))

                    # Prior RNN
                    self.Wpt = tf.get_variable(name='Wpt',
                                              shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wpt))

                    self.Vpt = tf.get_variable(name='Vpt', shape=(self.HIDDEN_LAYER_SIZE, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vpt))

                    self.wpt = tf.get_variable(name='wpt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(wpt))

                    self.bpt = tf.get_variable(name='bpt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bpt))

                self.all_vars = [self.Wt, self.Wem, self.Wh, self.bh,
                                 self.Wmh, self.bmh,
                                 self.wt, self.Wy, self.Vy, self.Vt,  self.Vomt, self.bt, self.bk,
                                 self.wmt, self.Vmt, self.Vmot, self.bmt]

                self.observed_initial_state = obs_state = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE],
                                                      dtype=self.FLOAT_TYPE,
                                                      name='observed_initial_state')
                self.observed_initial_time = obs_last_time_in = tf.zeros((self.inf_batch_size,),
                                                         dtype=self.FLOAT_TYPE,
                                                         name='observed_initial_time')
                self.initial_missing_state = miss_state = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE],
                                                      dtype=self.FLOAT_TYPE,
                                                      name='initial_missing_state')

                self.initial_prior_state = prior_state = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE],
                                                      dtype=self.FLOAT_TYPE,
                                                      name='initial_missing_state')

                self.initial_missing_time = miss_last_time = tf.zeros((self.inf_batch_size,),
                                                         dtype=self.FLOAT_TYPE,
                                                         name='initial_missing_time')

                self.loss = 0.0
                ones_2d = tf.ones((self.inf_batch_size, 1), dtype=self.FLOAT_TYPE)

                self.hidden_states = []
                self.miss_hidden_states = []

                self.event_preds = []

                self.time_LLs = []
                self.mark_LLs = []
                self.log_lambdas = []
                self.times = []
                self.miss_times = []
                mod_miss_time = tf.expand_dims(miss_last_time, axis=-1)
                obs_last_time = tf.expand_dims(obs_last_time_in, axis=-1)
                print("Checking dimensions")
                with tf.name_scope('BPTT'):
                    for i in range(self.BPTT):
                        self.iter_miss = []
                        self.kl_div = []

                        events_embedded = tf.nn.embedding_lookup(self.Wem, tf.mod(self.events_in[:, i] - 1, self.NUM_CATEGORIES))
                        time = self.times_in[:, i]
                        time_next = self.times_out[:, i]
                        nn_time = self.times_miss[:, i]
                        time_2d = tf.expand_dims(time, axis=-1)
                        mod_nn_time = tf.expand_dims(tf.to_float(nn_time), axis=-1)
                        mod_time_next = tf.expand_dims(time_next, axis=-1)
                        delta_t_prev = time_2d - obs_last_time
                        delta_t_next = mod_time_next - time_2d
                        
                        cons_time = tf.maximum(time_2d, mod_miss_time)
                        delta_t_cons = cons_time - obs_last_time
                        
                        obs_last_time = cons_time
                        type_delta_t = True

                        with tf.name_scope('state_recursion'):
                            new_obs_state = tf.tanh(
                                tf.matmul(obs_state, self.Wh) +
                                tf.matmul(events_embedded, self.Wy) +
                                (tf.matmul(delta_t_cons, self.Wt) if type_delta_t else tf.matmul(time_2d, self.Wt)) +
                                tf.matmul(ones_2d, self.bh),
                                name='oh_t')

                            obs_state = tf.where(self.events_in[:, i] > 0, new_obs_state, obs_state)
                            base_intensity = tf.matmul(ones_2d, self.bt)
                            wt_soft_plus = tf.nn.softplus(self.wt)

                            # Sampling from observed
                            lambda_part1 = tf.minimum(50.0, (tf.matmul(obs_state, self.Vt) + (tf.matmul(miss_state, self.Vomt) + base_intensity))) 
                            c1 = tf.exp(lambda_part1)
                            u = tf.random.uniform((self.inf_batch_size, 1), minval=0,maxval=1)
                            c1_sp = tf.nn.softplus(c1)

                            pred_t_delta = delta_t_cons - tf.matmul(tf.log(1 + tf.matmul(1/c1_sp, wt_soft_plus)*tf.log(1-u)), (1.0 / wt_soft_plus))
                            pred_t = tf.add(cons_time,pred_t_delta)

                            # RNN For missing data
                            delta_t_miss = mod_time_next - mod_miss_time
                            base_intensity_d = tf.matmul(ones_2d, self.bmt)
                            wmt_soft_plus = tf.nn.softplus(self.wmt)

                            new_embed = tf.nn.embedding_lookup(self.Wem, tf.mod(self.events_out[:, i] - 1, self.NUM_CATEGORIES))
                            lambda_d_part1 = tf.minimum(50.0, (tf.matmul(obs_state, self.Vmot) + tf.matmul(new_embed, self.Vat) + (tf.matmul(miss_state, self.Vmt) + base_intensity_d)))

                            u_d = tf.random.uniform((self.inf_batch_size, 1), minval=0,maxval=1)
                            c1_miss = tf.exp(lambda_d_part1)
                            c1_miss_sp = tf.nn.softplus(c1_miss)

                            # Sampling for missing 
                            # miss_delta_t = delta_t_miss - tf.matmul(tf.log(tf.maximum(0.007,1 + tf.matmul(1/c1_miss_sp, wmt_soft_plus)*tf.log(1 - u_d))), (1.0 / wmt_soft_plus))
                            miss_delta_t = delta_t_miss - tf.matmul(tf.log(1 + tf.matmul(1/c1_miss_sp, wmt_soft_plus)), (1.0 / wmt_soft_plus))
                            mod_miss_time = tf.add(mod_time_next,miss_delta_t) #Check the dimensions here
                            events_miss = tf.nn.softmax(tf.minimum(50.0, tf.matmul(obs_state, self.Vy) + tf.matmul(miss_state, self.Vmy) + ones_2d * self.bk),name='Pr_events')


                            iter_miss = miss_delta_t
                            iter_event_miss = events_miss
                            mod_i = tf.convert_to_tensor(i)
                            result = tf.while_loop(self.loop_condition, self.loop_body, [mod_nn_time, mod_miss_time, miss_state, obs_state, 
                                    ones_2d, wmt_soft_plus, prior_state, new_embed, base_intensity_d, i, iter_miss, iter_event_miss], maximum_iterations=10, 
                                    shape_invariants=[mod_nn_time.get_shape(), mod_miss_time.get_shape(), miss_state.get_shape(), obs_state.get_shape(), 
                                    ones_2d.get_shape(), wmt_soft_plus.get_shape(), prior_state.get_shape(), new_embed.get_shape(), base_intensity_d.get_shape(), mod_i.shape,tf.TensorShape([None, None]),tf.TensorShape([None, None])])
                            mod_miss_time = result[1]
                            miss_state = result[2]
                            prior_state = result[6]
                            iter_miss = result[10]
                            events_miss = result[11]

                        with tf.name_scope('loss_calc'):
                            wpt_soft_plus = tf.nn.softplus(self.wpt)
                            base_intensity_dd = tf.matmul(ones_2d, self.bpt)

                            log_obs_lambda = tf.minimum(50.0, (tf.matmul(obs_state, self.Vt) + tf.matmul(miss_state, self.Vomt)) + base_intensity + (-delta_t_cons * wt_soft_plus))
                            obs_lambda = tf.exp(tf.minimum(50.0, log_obs_lambda), name='obs_lambda')
                            log_p = (log_obs_lambda -
                                          (1.0 / wt_soft_plus) * tf.exp(tf.minimum(50.0, tf.matmul(obs_state, self.Vt) + tf.matmul(miss_state, self.Vomt) + base_intensity)) +
                                          (1.0 / wt_soft_plus) * obs_lambda)

                            log_miss_lambda = tf.minimum(50.0, (tf.matmul(obs_state, self.Vmot) + tf.matmul(new_embed, self.Vat) + tf.matmul(miss_state, self.Vmt) + base_intensity_d + (-delta_t_miss * wmt_soft_plus))) 
                            miss_lambda = tf.exp(tf.minimum(50.0, log_miss_lambda), name='miss_lambda')
                            posterior_q = (log_miss_lambda -
                                          (1.0 / wmt_soft_plus) * tf.exp(tf.minimum(50.0, tf.matmul(obs_state, self.Vmot) + tf.matmul(new_embed, self.Vat) + tf.matmul(miss_state, self.Vmt) + base_intensity_d)) +
                                          (1.0 / wmt_soft_plus) * miss_lambda)
                            
                            log_prior_lambda = tf.minimum(50.0, (tf.matmul(prior_state, self.Vpt) + base_intensity_dd + (-miss_delta_t * wpt_soft_plus)))
                            prior_lambda = tf.exp(tf.minimum(50.0, log_prior_lambda), name='prior_lambda')
                            prior_p = (log_prior_lambda -
                                          (1.0 / wpt_soft_plus) * tf.exp(tf.minimum(50.0, tf.matmul(prior_state, self.Vpt) + base_intensity_dd + (-miss_delta_t * wpt_soft_plus))) +
                                          (1.0 / wpt_soft_plus) * prior_lambda)

                            events_pred = tf.nn.softmax(tf.minimum(50.0, tf.matmul(obs_state, self.Vy) + tf.matmul(miss_state, self.Vmy) + ones_2d * self.bk),name='Pr_events')

                            ll_part_1 = log_p
                            
                            [iter_miss  , obs_state, new_embed, miss_state, base_intensity_d, wmt_soft_plus,prior_state, 
                            base_intensity_dd, wpt_soft_plus, kl] = self.calc_func(iter_miss, obs_state, new_embed, 
                              miss_state, base_intensity_d, wmt_soft_plus, prior_state, base_intensity_dd, wpt_soft_plus)

                            approx_kl = tfp.monte_carlo.expectation(f = lambda x: kl, samples=iter_miss, use_reparametrization = True)

                            classify_ll = tf.expand_dims(
                                tf.log(tf.maximum(1e-6, tf.gather_nd(events_pred, tf.concat([
                                                tf.expand_dims(tf.range(self.inf_batch_size), -1),
                                                tf.expand_dims(tf.mod(self.events_out[:, i] - 1, self.NUM_CATEGORIES), -1)
                                            ], axis=1, name='Pr_next_event')))), axis=-1, name='log_Pr_next_event')
                            
                            step_LL = classify_ll - ll_part_1
                            num_events = tf.reduce_sum(tf.where(self.events_in[:, i] > 0,
                                                       tf.ones(shape=(self.inf_batch_size,), dtype=self.FLOAT_TYPE),
                                                       tf.zeros(shape=(self.inf_batch_size,), dtype=self.FLOAT_TYPE)),
                                                       name='num_events') 
                          
                            var = tf.trainable_variables() 
                            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in var if 'b' not in v.name ]) * 0.001

                            kl_loss = tf.maximum(-50.0, tf.add(lossL2, tf.reduce_sum(approx_kl))/self.batch_num_events)
                            kl_loss = tf.minimum(50.0, kl_loss)
                            self.loss -= tf.reduce_sum(tf.where(self.events_in[:, i] > 0,
                            tf.squeeze(step_LL)/self.batch_num_events,tf.zeros(shape=(self.inf_batch_size,)))) - kl_loss

                        self.time_LLs.append(ll_part_1)
                        self.mark_LLs.append(classify_ll)
                        self.log_lambdas.append(log_obs_lambda)

                        self.hidden_states.append(obs_state)

                        self.event_preds.append(events_pred)
                        self.times.append(time)
                    
                    print("Done!")
                self.final_state = self.hidden_states[-1]

                with tf.device(device_cpu):
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.learning_rate = tf.train.inverse_time_decay(self.LEARNING_RATE,
                                                                 global_step=self.global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate)
              
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.MOMENTUM)
                self.gvs = self.optimizer.compute_gradients(self.loss)
                grads, vars_ = list(zip(*self.gvs))

                self.norm_grads, self.global_norm = tf.clip_by_global_norm(grads, 10.0)
                capped_gvs = list(zip(self.norm_grads, vars_))

                self.update = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
                self.tf_init = tf.global_variables_initializer()
    
    def calc_func(self, x, obs_state, new_embed, miss_state, base_intensity_d, wmt_soft_plus,prior_state, base_intensity_dd, wpt_soft_plus):
        kl = tf.reshape(tf.constant(0.0), [1,1])
        it = tf.convert_to_tensor(0)

        output = tf.while_loop(self.func_cond, self.func, [x, obs_state, new_embed, miss_state, base_intensity_d,
          wmt_soft_plus,prior_state, base_intensity_dd, wpt_soft_plus, kl, it], maximum_iterations=5,
          shape_invariants=[x.get_shape(), obs_state.get_shape(), new_embed.get_shape(), miss_state.get_shape(), base_intensity_d.get_shape(),
          wmt_soft_plus.get_shape(),prior_state.get_shape(), base_intensity_dd.get_shape(), wpt_soft_plus.get_shape(), tf.TensorShape([None, 1]), it.get_shape()])
        
        [x, obs_state, new_embed, miss_state, base_intensity_d, wmt_soft_plus,prior_state, base_intensity_dd, wpt_soft_plus, kl, it] = output
        
        return [x, obs_state, new_embed, miss_state, base_intensity_d, wmt_soft_plus,prior_state, base_intensity_dd, wpt_soft_plus, kl] 

    def func_cond(self, x, obs_state, new_embed, miss_state, base_intensity_d, wmt_soft_plus, prior_state, base_intensity_dd, wpt_soft_plus, kl, it):
        return tf.reduce_sum(it) <= 10

    def func(self, x, obs_state, new_embed, miss_state, base_intensity_d, wmt_soft_plus, prior_state, base_intensity_dd, wpt_soft_plus, kl, it):
        del_t = x[it][:]
        mod_del_t = del_t
        log_miss_lambda = tf.minimum(50.0, (tf.matmul(obs_state, self.Vmot) + tf.matmul(new_embed, self.Vat) + tf.matmul(miss_state, self.Vmt) + base_intensity_d + (-mod_del_t * wmt_soft_plus))) 
        miss_lambda = tf.exp(tf.minimum(50.0, log_miss_lambda), name='miss_lambda')
        posterior_q = (log_miss_lambda - (1.0 / wmt_soft_plus) * tf.exp(tf.minimum(50.0, tf.matmul(obs_state, self.Vmot) + 
                      tf.matmul(new_embed, self.Vat) + tf.matmul(miss_state, self.Vmt) + base_intensity_d)) +
                      (1.0 / wmt_soft_plus) * miss_lambda)
        
        log_prior_lambda = tf.minimum(50.0, (tf.matmul(prior_state, self.Vpt) + base_intensity_dd + (-mod_del_t * wpt_soft_plus)))
        prior_lambda = tf.exp(tf.minimum(50.0, log_prior_lambda), name='prior_lambda')

        prior_p = (log_prior_lambda - (1.0 / wpt_soft_plus) * tf.exp(tf.minimum(50.0, tf.matmul(prior_state, self.Vpt) + 
                  base_intensity_dd + (-mod_del_t * wpt_soft_plus))) + (1.0 / wpt_soft_plus) * prior_lambda)
        
        kl_div = tf.reduce_sum(tf.multiply(tf.nn.softplus(posterior_q),tf.log(tf.divide(tf.nn.softplus(posterior_q), 1e-4*tf.nn.softplus(prior_p)))))
        mod_kl_div = tf.reshape(kl_div, [1,1])

        kl = tf.concat([kl, mod_kl_div], axis=0)
        it += 1
        return [x, obs_state, new_embed, miss_state, base_intensity_d, wmt_soft_plus, prior_state, base_intensity_dd, wpt_soft_plus, kl, it]

    def loop_condition(self, mod_nn_time, mod_miss_time, miss_state, obs_state, ones_2d, wmt_soft_plus, prior_state, new_embed, base_intensity_d, i, iter_miss, iter_event_miss):
        return tf.reduce_sum(mod_miss_time) < tf.reduce_sum(mod_nn_time)

    def loop_body(self, mod_nn_time, mod_miss_time, miss_state, obs_state, ones_2d, wmt_soft_plus, prior_state, new_embed, base_intensity_d, i, iter_miss, iter_event_miss):
        # tf.print(mod_miss_time)
        lambda_d_part1 = tf.minimum(50.0, (tf.matmul(obs_state, self.Vmot) + tf.matmul(new_embed, self.Vat) + (tf.matmul(miss_state, self.Vmt) + base_intensity_d)))
        u_d = tf.random.uniform((self.inf_batch_size, 1), minval=0,maxval=1, dtype=tf.dtypes.float32)
        c1_miss = tf.exp(lambda_d_part1)
        c1_miss_sp = tf.nn.softplus(c1_miss)
        miss_rnn_dt = mod_nn_time - mod_miss_time
        new_miss_state = tf.tanh(tf.matmul(miss_state, self.Wmh) + tf.matmul(miss_rnn_dt, self.Wmt) + tf.matmul(ones_2d, self.bmh),name='mh_t')
        miss_state = tf.where(self.events_out[:, i] > 0, new_miss_state, miss_state)

        # miss_delta_t = miss_rnn_dt - tf.matmul(tf.log(1 + tf.matmul(1/c1_miss_sp, wmt_soft_plus) * tf.log(1 - u_d)), (1.0 / wmt_soft_plus))
        # miss_delta_t = miss_rnn_dt - tf.matmul(tf.log(tf.maximum(0.007,1 + tf.matmul(1/c1_miss_sp, wmt_soft_plus)*tf.log(1 - u_d))), (1.0 / wmt_soft_plus))
        miss_delta_t = miss_rnn_dt - tf.matmul(tf.log(1 + tf.matmul(1/c1_miss_sp, wmt_soft_plus)), (1.0 / wmt_soft_plus))
        events_miss = tf.nn.softmax(tf.minimum(50.0, tf.matmul(obs_state, self.Vy) + tf.matmul(miss_state, self.Vmy) + ones_2d * self.bk),name='Pr_events')        

        miss_time_new = tf.add(mod_miss_time,miss_delta_t)
        mod_miss_time = miss_time_new
        iter_miss = tf.concat([iter_miss, miss_delta_t], 1)
        iter_event_miss = tf.concat([iter_event_miss, events_miss], 1)

        # For prior
        new_prior_state = tf.tanh(tf.matmul(prior_state, self.Wph) + tf.matmul(miss_delta_t, self.Wpt) + tf.matmul(ones_2d, self.bph), name='ph_t')
        prior_state = tf.where(self.events_out[:, i] > 0, new_prior_state, prior_state)

        return [mod_nn_time, mod_miss_time, miss_state, obs_state, ones_2d, wmt_soft_plus, prior_state, new_embed, base_intensity_d, i, iter_miss, iter_event_miss]

    def initialize(self, finalize=False):
        self.sess.run(self.tf_init)

        if finalize:
            self.sess.graph.finalize()

    def train(self, training_data):
        num_epochs = 1
        train_event_in_seq = training_data['train_event_in_seq']
        train_time_in_seq = training_data['train_time_in_seq']
        train_time_miss_seq = training_data['train_time_miss_seq']
        train_event_out_seq = training_data['train_event_out_seq']
        train_time_out_seq = training_data['train_time_out_seq']

        idxes = list(range(len(train_event_in_seq)))
        n_batches = len(idxes) // self.BATCH_SIZE

        for epoch in range(self.last_epoch, self.last_epoch + num_epochs):
            self.rs.shuffle(idxes)
            total_loss = 0.0

            for batch_idx in range(n_batches):
                batch_idxes = idxes[batch_idx * self.BATCH_SIZE:(batch_idx + 1) * self.BATCH_SIZE]
                batch_event_train_in = train_event_in_seq[batch_idxes, :]
                batch_event_train_out = train_event_out_seq[batch_idxes, :]
                batch_time_train_in = train_time_in_seq[batch_idxes, :]
                batch_time_train_miss = train_time_miss_seq[batch_idxes, :]
                batch_time_train_out = train_time_out_seq[batch_idxes, :]

                cur_state = np.zeros((self.BATCH_SIZE, self.HIDDEN_LAYER_SIZE))
                batch_loss = 0.0

                batch_num_events = np.sum(batch_event_train_in > 0)
                for bptt_idx in range(0, len(batch_event_train_in[0]) - self.BPTT, self.BPTT):
                    bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
                    bptt_event_in = batch_event_train_in[:, bptt_range]
                    bptt_event_out = batch_event_train_out[:, bptt_range]
                    bptt_time_in = batch_time_train_in[:, bptt_range]
                    bptt_time_miss = batch_time_train_miss[:, bptt_range]
                    bptt_time_out = batch_time_train_out[:, bptt_range]

                    if np.all(bptt_event_in[:, 0] == 0):
                        break

                    if bptt_idx > 0:
                        initial_time = batch_time_train_in[:, bptt_idx - 1]
                    else:
                        initial_time = np.zeros(batch_time_train_in.shape[0])

                    feed_dict = {
                        self.observed_initial_state: cur_state,
                        self.observed_initial_time: initial_time,
                        self.events_in: bptt_event_in,
                        self.events_out: bptt_event_out,
                        self.times_in: bptt_time_in,
                        self.times_miss: bptt_time_miss,
                        self.times_out: bptt_time_out,
                        self.batch_num_events: batch_num_events
                    }

                    _, cur_state, loss_ = \
                        self.sess.run([self.update,
                                       self.final_state, self.loss],
                                      feed_dict=feed_dict)
                    batch_loss += loss_

                total_loss += batch_loss
            print('Loss after epoch {:.4f}'.format(total_loss / n_batches))

        self.last_epoch += num_epochs

    def predict(self, event_in_seq, event_out_seq, time_in_seq, time_n_seq, time_nn_seq, pred_event_in_seq, pred_time_in_seq, pred_event_out_seq, pred_time_out_seq, single_threaded=False):
        [pWt, pWem, pWh, pbh] = self.sess.run([self.Wt, self.Wem, self.Wh, self.bh])
        [pWpt, pbph, pWmh, pbmh] = self.sess.run([self.Wpt, self.bph, self.Wmh, self.bmh])
        [pwt, pWy, pVy, pVt] = self.sess.run([self.wt, self.Wy, self.Vy, self.Vt])
        [pVomt, pbt, pbk] = self.sess.run([self.Vomt, self.bt, self.bk])
        [pWmt, pVmy, pwmt] = self.sess.run([self.Wmt, self.Vmy, self.wmt])
        [pVmot, pbmt, pVat, pVmt] = self.sess.run([self.Vmot, self.bmt, self.Vat, self.Vmt])
        [pWpt, pVpt, pwpt, pbpt] = self.sess.run([self.Wpt, self.Vpt, self.wpt, self.bpt])

        graph = tf.Graph()
        with graph.as_default():
            [self.pWt, self.pWem, self.pWh, self.pbh] = [tf.convert_to_tensor(pWt), tf.convert_to_tensor(pWem), tf.convert_to_tensor(pWh), tf.convert_to_tensor(pbh)]
            [self.pWpt, self.pbph, self.pWmh, self.pbmh] = [tf.convert_to_tensor(pWpt), tf.convert_to_tensor(pbph), tf.convert_to_tensor(pWmh), tf.convert_to_tensor(pbmh)]
            [self.pwt, self.pWy, self.pVy, self.pVt] = [tf.convert_to_tensor(pwt), tf.convert_to_tensor(pWy), tf.convert_to_tensor(pVy), tf.convert_to_tensor(pVt)]
            [self.pVomt, self.pbt, self.pbk] = [tf.convert_to_tensor(pVomt), tf.convert_to_tensor(pbt), tf.convert_to_tensor(pbk)]
            [self.pWmt, self.pVmy, self.pwmt] = [tf.convert_to_tensor(pWmt), tf.convert_to_tensor(pVmy), tf.convert_to_tensor(pwmt)]
            [self.pVmot, self.pbmt, self.pVat, self.pVmt] = [tf.convert_to_tensor(pVomt), tf.convert_to_tensor(pbmt), tf.convert_to_tensor(pVat), tf.convert_to_tensor(pVmt)]
            [self.pWpt, self.pVpt, self.pwpt, self.pbpt] = [tf.convert_to_tensor(pWpt), tf.convert_to_tensor(pVpt), tf.convert_to_tensor(pwpt), tf.convert_to_tensor(pbpt)]

            self.pevents_in = tf.placeholder(tf.int32, [None, self.BPTT], name='events_in')
            self.ptimes_in = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_in')
            self.pinf_batch_size = tf.shape(self.pevents_in)[0]

            self.pinitial_state = obs_state = tf.zeros([self.pinf_batch_size, self.HIDDEN_LAYER_SIZE], dtype=self.FLOAT_TYPE, name='observed_initial_state')
            self.pinitial_time = obs_last_time_in = tf.zeros((self.pinf_batch_size,), dtype=self.FLOAT_TYPE, name='observed_initial_time')
            miss_state = tf.zeros([self.pinf_batch_size, self.HIDDEN_LAYER_SIZE],dtype=self.FLOAT_TYPE,name='initial_missing_state')
            
            last_time = tf.zeros((self.pinf_batch_size,), dtype=self.FLOAT_TYPE, name='initial_time')
            mod_last_time = tf.expand_dims(last_time, axis=-1)
            miss_last_time = tf.zeros((self.pinf_batch_size,), dtype=self.FLOAT_TYPE, name='initial_time')
            mod_last_miss_time = tf.expand_dims(last_time, axis=-1)
            ones_2d = tf.ones((self.pinf_batch_size, 1), dtype=self.FLOAT_TYPE)

            self.phidden_states = []
            self.pevent_preds = []

            for i in range(self.BPTT):
                events_embedded = tf.nn.embedding_lookup(self.pWem, tf.mod(self.pevents_in[:, i] - 1, self.NUM_CATEGORIES))
                time = self.ptimes_in[:, i]
                
                time_2d = tf.expand_dims(time, axis=-1)
                delta_t_prev = time_2d - mod_last_time
                type_delta_t = True
                mod_last_time = time_2d
                delta_t_miss = time_2d - mod_last_miss_time

                with tf.name_scope('state_recursion'):
                    new_obs_state = tf.tanh(tf.matmul(obs_state, self.pWh) +
                    tf.matmul(events_embedded, self.pWy) +
                    (tf.matmul(delta_t_prev, self.pWt) if type_delta_t else tf.matmul(time_2d, self.pWt)) +
                    tf.matmul(ones_2d, self.pbh), name='oh_t')
                    
                    obs_state = tf.where(self.pevents_in[:, i] > 0, new_obs_state, obs_state)
                    base_intensity = tf.matmul(ones_2d, self.pbt)
                    wt_soft_plus = tf.nn.softplus(self.pwt)
                    lambda_part1 = tf.minimum(50.0, (tf.matmul(obs_state, self.pVt) + (tf.matmul(miss_state, self.pVomt) + base_intensity))) 
                    c1 = tf.exp(lambda_part1)
                    u = tf.random.uniform((self.pinf_batch_size, 1), minval=0,maxval=1)
                    c1_sp = tf.nn.softplus(c1)

                    pred_t_delta = delta_t_prev - tf.matmul(tf.log(1 + tf.matmul(1/c1_sp, wt_soft_plus)*tf.log(1-u)), (1.0 / wt_soft_plus))
                    pred_t = tf.add(time_2d,pred_t_delta)
                    
                    events_pred = tf.nn.softmax(tf.minimum(50.0, tf.matmul(obs_state, self.pVy) + tf.matmul(miss_state, self.pVmy) + ones_2d * self.pbk),name='Pr_events')
                    self.phidden_states.append(obs_state)
                    self.pevent_preds.append(events_pred)

                    events_miss = events_pred

                    iter_miss = delta_t_miss
                    iter_event_miss = events_miss
                    base_intensity_d = tf.matmul(ones_2d, self.pbmt)
                    wmt_soft_plus = tf.nn.softplus(self.pwmt)
                    mod_i = tf.convert_to_tensor(i)

                    new_embed = events_embedded
                    result = tf.while_loop(self.predict_cond, self.predict_loop, [i, mod_last_miss_time, time_2d, miss_state, obs_state, new_embed, 
                            ones_2d, wmt_soft_plus, base_intensity_d, iter_miss, iter_event_miss], maximum_iterations=10, 
                            shape_invariants=[mod_i.get_shape(), mod_last_miss_time.get_shape(), time_2d.get_shape(), miss_state.get_shape(), obs_state.get_shape(), new_embed.get_shape(),
                            ones_2d.get_shape(), wmt_soft_plus.get_shape(), base_intensity_d.get_shape(), tf.TensorShape([None, None]),tf.TensorShape([None, None])])
                    mod_miss_time = result[1]
                    miss_state = result[3]
                    iter_miss = result[9]
                    events_miss = result[10]

            self.pfinal_state = obs_state

        # Prediction graph
        all_hidden_states = []
        all_event_preds = []
        cur_state = np.zeros((len(pred_event_in_seq), self.HIDDEN_LAYER_SIZE))
        for bptt_idx in range(0, len(pred_event_in_seq[0]) - self.BPTT, self.BPTT):
            bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
            bptt_event_in = pred_event_in_seq[:, bptt_range]
            bptt_time_in = pred_time_in_seq[:, bptt_range]

            if bptt_idx > 0:
                initial_time = pred_event_in_seq[:, bptt_idx - 1]
            else:
                initial_time = np.zeros(bptt_time_in.shape[0])

            feed_dict_p = {
              self.pinitial_state: cur_state,
              self.pinitial_time: initial_time,
              self.pevents_in: bptt_event_in,
              self.ptimes_in: bptt_time_in,
            }

            with tf.Session(graph=graph) as sess:
                bptt_hidden_states, bptt_events_pred, cur_state = sess.run(
                    [self.phidden_states, self.pevent_preds, self.pfinal_state],
                    feed_dict=feed_dict_p)
        
            all_hidden_states.extend(bptt_hidden_states)
            all_event_preds.extend(bptt_events_pred)

        [Vt, bt, wt]  = self.sess.run([self.Vt, self.bt, self.wt])
        wt = softplus(wt)

        pickle.dump([Vt, bt, wt, all_hidden_states, pred_time_in_seq, pred_time_out_seq, pred_event_in_seq, pred_event_out_seq], open('Our.p','wb'))
        pickle.dump([pWem, pVy, pbk, pWh, pWy, pWt, pbh, cur_state, pred_time_in_seq, pred_time_out_seq, pred_event_in_seq, pred_event_out_seq], open('Event.p','wb'))

        global _quad_worker
        def _quad_worker(params):
            idx, h_i = params
            preds_i = []
            C = np.exp(np.dot(h_i, Vt) + bt).reshape(-1)

            for c_, t_last in zip(C, pred_time_in_seq[:, idx]):
                args = (c_, wt)
                val, _err = quad(quad_func, 0, np.inf, args=args)
                preds_i.append(t_last + val)

            return preds_i

        if single_threaded:
            all_time_preds = [_quad_worker((idx, x)) for idx, x in enumerate(all_hidden_states)]
        else:
            with MP.Pool() as pool:
                all_time_preds = pool.map(_quad_worker, enumerate(all_hidden_states))

        return np.asarray(all_time_preds).T, np.asarray(all_event_preds).swapaxes(0, 1)

    def predict_cond(self, i, mod_last_miss_time, time_2d, miss_state, obs_state, new_embed, ones_2d, wmt_soft_plus, base_intensity_d, iter_miss, iter_event_miss):
        return tf.reduce_sum(mod_last_miss_time) < tf.reduce_sum(time_2d)

    def predict_loop(self, i, mod_last_miss_time, time_2d, miss_state, obs_state, new_embed, ones_2d, wmt_soft_plus, base_intensity_d, iter_miss, iter_event_miss):
        lambda_d_part1 = tf.minimum(50.0, (tf.matmul(obs_state, self.pVmot) + tf.matmul(new_embed, self.pVat) + (tf.matmul(miss_state, self.pVmt) + base_intensity_d)))
        u_d = tf.random.uniform((self.pinf_batch_size, 1), minval=0,maxval=1, dtype=tf.dtypes.float32)
        c1_miss = tf.exp(lambda_d_part1)
        c1_miss_sp = tf.nn.softplus(c1_miss)

        miss_rnn_dt = time_2d - mod_last_miss_time
        new_miss_state = tf.tanh(tf.matmul(miss_state, self.pWmh) + tf.matmul(miss_rnn_dt, self.pWmt) + tf.matmul(ones_2d, self.pbmh),name='mh_t')
        miss_state = tf.where(self.pevents_in[:, i] > 0, new_miss_state, miss_state)

        # miss_delta_t = miss_rnn_dt - tf.matmul(tf.log(1 + tf.matmul(1/c1_miss_sp, wmt_soft_plus) * tf.log(1 - u_d)), (1.0 / wmt_soft_plus))
        # miss_delta_t = miss_rnn_dt - tf.matmul(tf.log(tf.maximum(0.007,1 + tf.matmul(1/c1_miss_sp, wmt_soft_plus)*tf.log(1 - u_d))), (1.0 / wmt_soft_plus))
        miss_delta_t = miss_rnn_dt - tf.matmul(tf.log(1 + tf.matmul(1/c1_miss_sp, wmt_soft_plus)), (1.0 / wmt_soft_plus))
        events_miss = tf.nn.softmax(tf.minimum(50.0, tf.matmul(obs_state, self.pVy) + tf.matmul(miss_state, self.pVmy) + ones_2d * self.pbk),name='Pr_events')        
        miss_time_new = tf.add(mod_last_miss_time,miss_delta_t)
        mod_last_miss_time = miss_time_new
        iter_miss = tf.concat([iter_miss, miss_delta_t], 1)
        iter_event_miss = tf.concat([iter_event_miss, events_miss], 1)

        return [i, mod_last_miss_time, time_2d, miss_state, obs_state, new_embed, ones_2d, wmt_soft_plus, base_intensity_d, iter_miss, iter_event_miss]

    def eval(self, time_preds, time_true, event_preds, event_true):
        mae, _ = MAE(time_preds, time_true, event_true)
        print('** MAE = {:.4f}; ACC = {:.4f}'.format(
            mae, ACC(event_preds, event_true)))

    def predict_test(self, data, single_threaded=False):
        return self.predict(event_in_seq=data['train_event_in_seq'],
               event_out_seq=data['train_event_out_seq'],
               time_in_seq=data['train_time_in_seq'],
               time_n_seq = data['train_time_out_seq'],
               time_nn_seq = data['train_time_miss_seq'],
               pred_event_in_seq = data['test_event_in_seq'],
               pred_time_in_seq = data['test_time_in_seq'],
               pred_event_out_seq = data['test_event_out_seq'],
               pred_time_out_seq = data['test_time_out_seq'],
               single_threaded=single_threaded)
