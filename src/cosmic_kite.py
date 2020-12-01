import os
import pkg_resources

location = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(location, 'data')

# Decoder loss
def dec_loss(true, pred):

  factor   = tf.range(lmin, lmax, dtype = 'float32')
  factor   = 2*factor + 1
  multiply = tf.reshape(tf.shape(true)[0], (1,))
  factor   = tf.reshape(tf.tile(factor, multiply), [ multiply[0], tf.shape(factor)[0]])

  aux0 = K.log(K.abs(true/pred))
  aux1 = (pred/true)
  aux  = factor*(aux0 + aux1 - 1)

  loss = K.sum(aux, axis = 0)

  return K.mean(loss)

# Let's load the scaler
from pickle import dump
from pickle import load
from sklearn import preprocessing

scaler_x = load(open(data_path + '/scaler_x.pkl', 'rb'))
scaler_y = load(open(data_path + '/scaler_y.pkl', 'rb'))

# Let's load some neccesary libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU

# Let's define the architecture and load the trained values

# Armo los tensores input
lmin = 50
lmax = 2500
actFun = 'LeakyReLU'
batch_size        = 256
original_dim      = lmax-lmin
latent_dim        = 6
intermediate_dim  = 1000
intermediate_dim2 = 500
intermediate_dim3 = 100
intermediate_dim4 = 50
epochs            = 100
epsilon_std       = 1.0
input_shape = (original_dim,)

# VAE model = encoder + decoder
# build encoder model
inputs    = Input(shape=input_shape, name='encoder_input')
x         = Dense(intermediate_dim)(inputs)
x         = LeakyReLU(alpha=0.1)(x)
x         = Dense(intermediate_dim3)(x)
x         = LeakyReLU(alpha=0.1)(x)
x         = Dense(intermediate_dim4)(x)
x         = LeakyReLU(alpha=0.1)(x)
latent    = Dense(latent_dim, name='z_mean', activation='linear')(x)

# instantiate encoder model
encoder = Model(inputs, latent, name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x             = Dense(intermediate_dim4)(latent_inputs)
x             = LeakyReLU(alpha=0.1)(x)
x             = Dense(intermediate_dim3)(x)
x             = LeakyReLU(alpha=0.1)(x)
x             = Dense(intermediate_dim)(x)
x             = LeakyReLU(alpha=0.1)(x)
outputs       = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
dec_output = decoder(latent)
enc_output = encoder(inputs)
vae        = Model(inputs, [enc_output, dec_output], name='vae_mlp')

vae.load_weights(data_path + '/vae_model.h5')

# Let's define de main functions

def pars2ps(pars):
  pars_scaled    = scaler_y.transform(pars)
  ps_pred_scaled = decoder.predict(pars_scaled)
  ps_pred        = scaler_x.inverse_transform(ps_pred_scaled)
  return ps_pred

def ps2pars(ps):
  ps_scaled        = scaler_x.transform(ps)
  pars_pred_scaled = encoder.predict(ps_scaled)
  pars_pred        = scaler_y.inverse_transform(pars_pred_scaled)
  return pars_pred  
#def pars2ps(H0, ombh2, omch2, n, tau, As):
#  params  = scaler_y.transform([(ombh2, omch2, H0, n, tau, As)])
#  pred_cl = scaler_x.inverse_transform(decoder.predict(params))[0,:]
#  return pred_cl
#
#def ps2pars(ps):
#  pars = scaler_y.inverse_transform(encoder.predict(scaler_x.transform(ps.reshape(1,2450)))[0][0:6])
#  return pars
