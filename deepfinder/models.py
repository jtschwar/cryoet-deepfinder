# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.models import Model

def my_model(dim_in, Ncl):
    
    input = Input(shape=(dim_in,dim_in,dim_in,1))
    
    x    = Conv3D(32, (3,3,3), padding='same', activation='relu')(input)
    high = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    
    x = MaxPooling3D((2,2,2), strides=None)(high)
    
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    mid = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    
    x = MaxPooling3D((2,2,2), strides=None)(mid)
    
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2,2,2), data_format='channels_last')(x)
    x = Conv3D(64, (2,2,2), padding='same', activation='relu')(x)
    
    x = concatenate([x, mid])
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    
    x = UpSampling3D(size=(2,2,2), data_format='channels_last')(x)
    x = Conv3D(48, (2,2,2), padding='same', activation='relu')(x)
    
    x = concatenate([x, high])
    x = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    
    output = Conv3D(Ncl, (1,1,1), padding='same', activation='softmax')(x)
    
    model = Model(input, output)
    return model

def residual_block(input, filters, kernel_size=(3,3,3)):
    x = Conv3D(filters, kernel_size, padding='same')(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv3D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    if input.shape[-1] != filters:
        input = Conv3D(filters, (1,1,1), padding='same')(input)

    x = Add()([input, x])
    x = LeakyReLU()(x)
    return x

def my_res_unet_model(dim_in, Ncl):
    input = Input(shape=(dim_in,dim_in,dim_in,1))

    x = Conv3D(32, (3,3,3), padding='same')(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv3D(32, (3,3,3), padding='same')(x)
    x = BatchNormalization()(x)
    high = LeakyReLU()(x)

    x = MaxPooling3D((2,2,2))(high)

    x = Conv3D(48, (3,3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv3D(48, (3,3,3), padding='same')(x)
    x = BatchNormalization()(x)
    mid = LeakyReLU()(x)

    x = MaxPooling3D((2,2,2))(mid)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = UpSampling3D(size=(2,2,2))(x)
    x = Conv3D(64, (2,2,2), padding='same', activation='relu')(x)

    x = concatenate([x, mid])
    x = residual_block(x, 48)
    x = residual_block(x, 48)

    x = UpSampling3D(size=(2,2,2))(x)
    x = Conv3D(48, (2,2,2), padding='same', activation='relu')(x)

    x = concatenate([x, high])
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    output = Conv3D(Ncl, (1,1,1), padding='same', activation='softmax')(x)

    model = Model(input, output)
    return model
