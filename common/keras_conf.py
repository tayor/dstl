# Keras common configuration

import os
from keras.backend import set_image_dim_ordering

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'
set_image_dim_ordering('th')

print("Keras user configuration is setup")