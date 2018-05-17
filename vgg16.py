from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import six
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape


def VGG16(include_top=True, input_shape=(3, 32, 32), classes=10):
    """Instantiates the VGG16 architecture.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        input_shape: optional shape tuple
        classes: optional number of classes to classify images
            into
    # Returns
        A Keras model instance.

    """
    # Determine proper input shape

    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape = input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(2048, activation='relu', name='fc1'))
    model.add(Dense(2048, activation='relu', name='fc2'))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    return model
