import os
import pkg_resources
import numpy as np

location = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(location, 'data')

# Let's load the scaler
from pickle import load
from sklearn import preprocessing

scaler_x = load(open(data_path + '/scaler_x_3ch.pkl', 'rb'))
scaler_y = load(open(data_path + '/scaler_y_3ch.pkl', 'rb'))

# Let's load some neccesary libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU

# Let's define the architecture and load the trained values

# Armo los tensores input
lmin = 2
lmax = 2650
actFun = 'LeakyReLU'
batch_size        = 256
original_dim      = 3*(lmax-lmin)
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

vae.load_weights(data_path + '/3ch_model.h5')

# Let's define de main functions

def pars2ps(pars):
  if ((np.max(pars[:,0]) > 0.023482) or (np.min(pars[:,0]) < 0.021282)): print('WARNING! Some values of omega_b*h2 are outside the training range')
  if ((np.max(pars[:,1]) > 0.130607) or (np.min(pars[:,1]) < 0.109607)): print('WARNING! Some values of omega_c*h2 are outside the training range')
  if ((np.max(pars[:,2]) > 76.52) or (np.min(pars[:,2]) < 58.121)): print('WARNING! Some values of H0 are outside the training range')
  if ((np.max(pars[:,3]) > 0.99454) or (np.min(pars[:,3]) < 0.937550)): print('WARNING! Some values of n are outside the training range')
  if ((np.max(pars[:,4]) > 0.094) or (np.min(pars[:,4]) < 0.01430)): print('WARNING! Some values of tau are outside the training range')
  if ((np.max(pars[:,5]) > 2.2705e-9) or (np.min(pars[:,5]) < 1.9305e-9)): print('WARNING! Some values of As are outside the training range')
  pars_scaled    = scaler_y.transform(pars)
  ps_pred_scaled = decoder.predict(pars_scaled)
  ps_pred        = scaler_x.inverse_transform(ps_pred_scaled)
  tt_pred        = ps_pred[:,0:(lmax-lmin)]
  ee_pred        = ps_pred[:,(lmax-lmin):2*(lmax-lmin)]
  te_pred        = ps_pred[:,2*(lmax-lmin):3*(lmax-lmin)]
  ell_array      = np.arange(lmin, lmax)
  return [tt_pred, ee_pred, te_pred, ell_array]


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
