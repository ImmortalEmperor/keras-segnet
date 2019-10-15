from keras.models import Model
from keras.layers import Input, MaxPool2D, Concatenate
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D, Conv2DTranspose, UpSampling2D
from keras.layers.normalization import BatchNormalization

def FastSegNet(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode='softmax'):
    """
    SegNet perpixel image classifier
    """
    # encoder
    inputs = Input(shape=input_shape)
    
    conv_1 = Convolution2D(32, (kernel, kernel), padding='same')(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)
    
    pool_1 = MaxPool2D(pool_size, name='Pool_1')(conv_1)
    
    conv_2 = Convolution2D(64, (kernel, kernel), padding='same')(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)
    
    pool_2 = MaxPool2D(pool_size, name='Pool_2')(conv_2)
    
    conv_3 = Convolution2D(128, (kernel, kernel), padding='same')(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)
    
    pool_3 = MaxPool2D(pool_size, name='Pool_3')(conv_3)
    
    conv_4 = Convolution2D(256, (kernel, kernel), padding='same')(pool_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)

    pool_4 = MaxPool2D(pool_size, name='Pool_4')(conv_4)
    
    conv_5 = Convolution2D(256, (kernel, kernel), padding='same')(pool_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation('relu')(conv_5)
    
    pool_5 = MaxPool2D(pool_size, name='Pool_5')(conv_5)
    
    # decoder
    
    unpool_1 = UpSampling2D(pool_size, name='Un-Pool_1')(pool_5)
    
    deconv_1 = Conv2DTranspose(256, (kernel, kernel), padding='same')(unpool_1)
    deconv_1 = BatchNormalization()(deconv_1)
    deconv_1 = Activation('relu')(deconv_1)
    
    concat_1 = Concatenate()([deconv_1, pool_4])
    unpool_2 = UpSampling2D(pool_size, name='Un-Pool_2')(concat_1)
    
    deconv_2 = Conv2DTranspose(256, (kernel, kernel), padding='same')(unpool_2)
    deconv_2 = BatchNormalization()(deconv_2)
    deconv_2 = Activation('relu')(deconv_2)
    
    concat_2 = Concatenate()([deconv_2, pool_3])
    unpool_3 = UpSampling2D(pool_size, name='Un-Pool_3')(concat_2)
    
    deconv_3 = Conv2DTranspose(128, (kernel, kernel), padding='same')(unpool_3)
    deconv_3 = BatchNormalization()(deconv_3)
    deconv_3 = Activation('relu')(deconv_3)
    
    concat_3 = Concatenate()([deconv_3, pool_2])
    unpool_4 = UpSampling2D(pool_size, name='Un-Pool_4')(concat_3)    
    
    deconv_4 = Conv2DTranspose(64, (kernel, kernel), padding='same')(unpool_4)
    deconv_4 = BatchNormalization()(deconv_4)
    deconv_4 = Activation('relu')(deconv_4)

    concat_4 = Concatenate()([deconv_4, pool_1])
    unpool_5 = UpSampling2D(pool_size, name='Un-Pool_5')(concat_4)
    
    deconv_5 = Conv2DTranspose(32, (kernel, kernel), padding='same')(unpool_5)
    deconv_5 = BatchNormalization()(deconv_5)
    deconv_5 = Activation('relu')(deconv_5)
    
    deconv_6 = Conv2DTranspose(n_labels, (1, 1), padding='valid')(deconv_5)
    deconv_6 = BatchNormalization()(deconv_6)
    deconv_6 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(deconv_6)
    
    outputs = Activation(output_mode)(deconv_6)
    
    return Model(inputs=inputs, outputs=outputs, name='FastSegNet')
