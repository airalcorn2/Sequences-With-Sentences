# Michael A. Alcorn (malcorn@redhat.com)
# See: https://github.com/dennybritz/rnn-tutorial-gru-lstm.

import numpy as np
import theano

from theano import tensor as T

INPUT_DIM = 50
WORD_WINDOW = 5
WORD_DEPTH = 100
WORD_DIM = WORD_WINDOW * WORD_DEPTH
HIDDEN_DIM = 128
N_FILTERS = 100

input_dim = INPUT_DIM
word_dim = WORD_DIM
hidden_dim = HIDDEN_DIM
n_filters = N_FILTERS
bptt_truncate = -1
step_dim = input_dim + n_filters

# Initialize CNN parameters.
F = np.array(np.random.randn(n_filters, word_dim), dtype = theano.config.floatX) * 0.01

# Initialize RNN parameters.
U_1 = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (3, hidden_dim, step_dim))
U_2 = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (3, hidden_dim, hidden_dim))
W = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (6, hidden_dim, hidden_dim))
V = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (1, hidden_dim))
b = np.zeros((6, hidden_dim))
c = 0.0
h0_l1 = np.zeros(hidden_dim)
h0_l2 = np.zeros(hidden_dim)

# Initialize CNN gradients.
mF = theano.shared(name = "mF", value = np.zeros(F.shape).astype(theano.config.floatX))

# Initialize RNN gradients.
mU_1 = theano.shared(name = "mU_1", value = np.zeros(U_1.shape).astype(theano.config.floatX))
mU_2 = theano.shared(name = "mU_2", value = np.zeros(U_2.shape).astype(theano.config.floatX))
mV = theano.shared(name = "mV", value = np.zeros(V.shape).astype(theano.config.floatX))
mW = theano.shared(name = "mW", value = np.zeros(W.shape).astype(theano.config.floatX))
mb = theano.shared(name = "mb", value = np.zeros(b.shape).astype(theano.config.floatX))
mc = theano.shared(name = "mc", value = 0.0)

# Create CNN shared variables.
F = theano.shared(name = "F", value = F.astype(theano.config.floatX))

# Create RNN shared variables.
U_1 = theano.shared(name = "U_1", value = U_1.astype(theano.config.floatX))
U_2 = theano.shared(name = "U_2", value = U_2.astype(theano.config.floatX))
W = theano.shared(name = "W", value = W.astype(theano.config.floatX))
V = theano.shared(name = "V", value = V.astype(theano.config.floatX))
b = theano.shared(name = "b", value = b.astype(theano.config.floatX))
c = theano.shared(name = "c", value = c)
h0_l1 = theano.shared(name = "h0_l1", value = h0_l1.astype(theano.config.floatX))
h0_l2 = theano.shared(name = "h0_l2", value = h0_l2.astype(theano.config.floatX))

x = T.fmatrix("x")
sentences = T.ftensor3("sentences")
y = T.bmatrix("y")


def forward_prop_step(x_t, sentence_t, s_t1_prev, s_t2_prev):
    
    filtered_words = T.tanh(F.dot(sentence_t))
    pooled_words = filtered_words.max(axis = 1)
    
    x_e = T.concatenate([x_t, pooled_words])
    
    # GRU Layer 1.
    z_t1 = T.nnet.hard_sigmoid(U_1[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
    r_t1 = T.nnet.hard_sigmoid(U_1[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
    c_t1 = T.tanh(U_1[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
    s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
    
    # GRU Layer 2.
    z_t2 = T.nnet.hard_sigmoid(U_2[0].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
    r_t2 = T.nnet.hard_sigmoid(U_2[1].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
    c_t2 = T.tanh(U_2[2].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
    s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
    
    # Final output calculation.
    o_t = T.nnet.sigmoid(V.dot(s_t2) + c)
    
    return [o_t, s_t1, s_t2]


[o, s, s2], updates = theano.scan(forward_prop_step,
                                  sequences = [x, sentences],
                                  truncate_gradient = bptt_truncate,
                                  outputs_info = [None,
                                                  dict(initial = T.zeros(hidden_dim)),
                                                  dict(initial = T.zeros(hidden_dim))])


X = []
Y = []
text = []
MAX_TEXT_LEN = 50
MAX_STEPS = 20
N = 20

for i in range(N):
    steps = np.random.randint(1, MAX_STEPS + 1)
    X.append(np.random.rand(steps, INPUT_DIM).astype("float32"))
    Y.append(np.array([np.random.randint(1 + 1) for j in range(steps)], dtype = "int8").reshape(steps, 1))

LEARNING_RATE = 1e-3
EPOCHS = 20
test_text = np.random.rand(X[0].shape[0], WORD_DIM, MAX_TEXT_LEN).astype("float32")

# Assign functions.
predict = theano.function([x, sentences], o, mode = "DebugMode")
predict(X[0], test_text)

o_error = T.sum(T.nnet.binary_crossentropy(o, y))

# Total cost (could add regularization here).
cost = o_error

get_cost = theano.function([x, sentences, y], cost)
get_cost(X[0], test_text, Y[0])

# Gradients.
dU_1 = T.grad(cost, U_1)
dU_2 = T.grad(cost, U_2)
dW = T.grad(cost, W)
db = T.grad(cost, b)
dV = T.grad(cost, V)
dc = T.grad(cost, c)
dF = T.grad(cost, F)

learning_rate = T.scalar("learning_rate")
decay = T.scalar("decay")

# rmsprop cache updates.
cache_mU_1 = decay * mU_1 + (1 - decay) * dU_1 ** 2
cache_mU_2 = decay * mU_2 + (1 - decay) * dU_2 ** 2
cache_mW = decay * mW + (1 - decay) * dW ** 2
cache_mV = decay * mV + (1 - decay) * dV ** 2
cache_mb = decay * mb + (1 - decay) * db ** 2
cache_mc = decay * mc + (1 - decay) * dc ** 2
cache_mF = decay * mF + (1 - decay) * dc ** 2

sgd_step = theano.function([x, sentences, y, learning_rate,
                            theano.In(decay, value = 0.9)],
                            [], updates = [(U_1, U_1 - learning_rate * dU_1 / T.sqrt(mU_1 + 1e-6)),
                                           (U_2, U_2 - learning_rate * dU_2 / T.sqrt(mU_2 + 1e-6)),
                                           (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                                           (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                                           (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                                           (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                                           (F, F - learning_rate * dF / T.sqrt(mF + 1e-6)),
                                           (mU_1, cache_mU_1),
                                           (mU_2, cache_mU_2),
                                           (mW, cache_mW),
                                           (mV, cache_mV),
                                           (mb, cache_mb),
                                           (mc, cache_mc),
                                           (mF, cache_mF)])

sgd_step(X[0], test_text, Y[0], LEARNING_RATE)
