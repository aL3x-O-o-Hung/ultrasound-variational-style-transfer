from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.ops import math_ops
import numpy as np

BASE_NUM_KERNELS = 64

print(tf.__version__)


class BatchNormRelu(tf.keras.layers.Layer):
    """Batch normalization + ReLu"""

    def __init__(self, name=None, dtype=None):
        super(BatchNormRelu, self).__init__(name=name)
        self.bnorm = tf.keras.layers.experimental.SyncBatchNormalization(momentum=0.9, dtype=dtype)

    def call(self, inputs, is_training=True):
        x = self.bnorm(inputs, training=is_training)
        x = tf.keras.activations.swish(x)
        return x




class Conv2DFixedPadding(tf.keras.layers.Layer):
    """Conv2D Fixed Padding layer"""

    def __init__(self, filters, kernel_size, stride, name=None, dtype=None):
        super(Conv2DFixedPadding, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=1,
            dilation_rate=1,
            padding=('same' if stride == 1 else 'valid'),
            activation=None,
            dtype=dtype
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        return x

class Conv2DTranspose(tf.keras.layers.Layer):
    """Conv2DTranspose layer"""

    def __init__(self, output_channels, kernel_size, name=None, dtype=None):
        super(Conv2DTranspose, self).__init__(name=name)
        '''
        self.tconv1 = tf.keras.layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            activation=None,
            dtype=dtype
        )
        '''
        self.upsample=tf.keras.layers.UpSampling2D(interpolation='bilinear')
        self.conv=Conv2DFixedPadding(filters=output_channels,kernel_size=kernel_size,stride=1)
        #self.brelu=BatchNormRelu()

    def call(self, inputs,is_training=True):
        #x = self.tconv1(inputs)
        x=self.upsample(inputs)
        x=self.conv(x)
        #x=self.brelu(x,is_training=is_training)
        return x

class ConvBlock(tf.keras.layers.Layer):
    """Downsampling ConvBlock on Encoder side"""

    def __init__(self, filters, kernel_size=3, do_max_pool=True, name=None):
        super(ConvBlock, self).__init__(name=name)
        self.do_max_pool = do_max_pool
        self.conv1 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=kernel_size,
                                        stride=1)
        self.brelu1 = BatchNormRelu()
        self.conv2 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=kernel_size,
                                        stride=1)
        self.brelu2 = BatchNormRelu()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2,
                                                  strides=2,
                                                  padding='valid')

    def call(self, inputs, is_training=True):
        x = self.conv1(inputs)
        x = self.brelu1(x, is_training)
        x = self.conv2(x)
        x = self.brelu2(x, is_training)
        output_b = x
        if self.do_max_pool:
            x = self.max_pool(x)
        return x, output_b

class SEBlock(tf.keras.layers.Layer):
    def __init__(self,channel_original,channel_new,name=None):
        super(SEBlock,self).__init__(name=name)
        self.pool=tf.keras.layers.GlobalAveragePooling2D()
        self.conv1=Conv2DFixedPadding(filters=channel_new,kernel_size=1,stride=1)
        self.conv2=Conv2DFixedPadding(filters=channel_original,kernel_size=1,stride=1)
    def call(self,inputs):
        x=self.pool(inputs)
        x=tf.expand_dims(x,axis=1)
        x=tf.expand_dims(x,axis=1)
        x=self.conv1(x)
        x=tf.keras.activations.relu(x)
        x=self.conv2(x)
        x=tf.keras.activations.sigmoid(x)
        x=x*inputs
        return x

class FirstConvBlock(tf.keras.layers.Layer):
    """Downsampling ConvBlock on Encoder side"""

    def __init__(self, filters, kernel_size=3, name=None):
        super(FirstConvBlock, self).__init__(name=name)
        self.conv1 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=kernel_size,
                                        stride=1)
        self.brelu1 = BatchNormRelu()
        self.conv2 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=kernel_size,
                                        stride=1)

    def call(self, inputs, is_training=True):
        x = self.conv1(inputs)
        x = self.brelu1(x, is_training)
        x = self.conv2(x)
        return x


class ResNetConvBlock(tf.keras.layers.Layer):
    """Downsampling ConvBlock on Encoder side"""

    def __init__(self, filters,r,kernel_size=3, name=None):
        super(ResNetConvBlock, self).__init__(name=name)
        self.filters = filters
        if self.filters <= 128:
            self.conv1 = Conv2DFixedPadding(filters=filters,
                                            kernel_size=kernel_size,
                                            stride=1)
            self.brelu1 = BatchNormRelu()
            self.conv2 = Conv2DFixedPadding(filters=filters,
                                            kernel_size=kernel_size,
                                            stride=1)
            self.brelu2 = BatchNormRelu()
        else:
            self.conv1 = Conv2DFixedPadding(filters=filters // 4,
                                            kernel_size=1,
                                            stride=1)
            self.brelu1 = BatchNormRelu()
            self.conv2 = Conv2DFixedPadding(filters=filters // 4,
                                            kernel_size=kernel_size,
                                            stride=1)
            self.brelu2 = BatchNormRelu()
            self.conv3 = Conv2DFixedPadding(filters=filters,
                                            kernel_size=1,
                                            stride=1)
            self.brelu3 = BatchNormRelu()
        self.se=SEBlock(filters,filters//r)

    def call(self, inputs, is_training):
        if self.filters <= 128:
            x = self.brelu1(inputs, is_training=True)
            x = self.conv1(x)
            x = self.brelu2(x, is_training)
            x = self.conv2(x)
        else:
            x = self.brelu1(inputs, is_training=True)
            x = self.conv1(x)
            x = self.brelu2(x, is_training)
            x = self.conv2(x)
            x = self.brelu3(x, is_training)
            x = self.conv3(x)
        x=self.se(x)
        x = inputs + x
        return x


class DownSampleBlock(tf.keras.layers.Layer):
    """Downsampling ConvBlock on Encoder side"""

    def __init__(self, filters, kernel_size=3, name=None):
        super(DownSampleBlock, self).__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=2,
            dilation_rate=1,
            padding='same',
            activation=None,
        )
        self.brelu = BatchNormRelu()

    def call(self, inputs, is_training=True):
        output_b = inputs
        x = self.brelu(inputs, is_training)
        x = self.conv(x)
        return x, output_b


class FakeDownSampleBlock(tf.keras.layers.Layer):
    """Downsampling ConvBlock on Encoder side"""

    def __init__(self, filters, kernel_size=3, name=None):
        super(FakeDownSampleBlock, self).__init__(name=name)
        self.brelu = BatchNormRelu()

    def call(self, inputs, is_training=True):
        output_b = inputs
        x = self.brelu(inputs, is_training)
        return x, output_b


class DeConvBlock(tf.keras.layers.Layer):
    """Upsampling DeConvBlock on Decoder side"""

    def __init__(self, filters, kernel_size=2, name=None):
        super(DeConvBlock, self).__init__(name=name)
        self.tconv1 = Conv2DTranspose(output_channels=filters,
                                      kernel_size=kernel_size)
        self.conv1 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=3,
                                        stride=1)
        self.brelu1 = BatchNormRelu()
        self.conv2 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=3,
                                        stride=1)
        self.brelu2 = BatchNormRelu()

    def call(self, inputs, output_b, is_training=True):
        x = self.tconv1(inputs,is_training=is_training)
        x = tf.keras.activations.swish(x)

        """Cropping is only used when convolution padding is 'valid'"""
        src_shape = output_b.shape[1]
        tgt_shape = x.shape[1]
        start_pixel = int((src_shape - tgt_shape) / 2)
        end_pixel = start_pixel + tgt_shape

        cropped_b = output_b[:, start_pixel:end_pixel, start_pixel:end_pixel, :]
        """Assumes that data format is NHWC"""
        x = tf.concat([cropped_b, x], axis=-1)

        x = self.conv1(x)
        x = self.brelu1(x, is_training)
        x = self.conv2(x)
        x = self.brelu2(x, is_training)
        return x


class ResNetDeConvBlock(tf.keras.layers.Layer):
    """Upsampling DeConvBlock on Decoder side"""

    def __init__(self, filters,r,kernel_size=2, name=None):
        super(ResNetDeConvBlock, self).__init__(name=name)
        self.tconv1 = Conv2DTranspose(output_channels=filters,
                                      kernel_size=kernel_size)
        self.conv1 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=3,
                                        stride=1)
        self.brelu1 = BatchNormRelu()
        self.brelu2 = BatchNormRelu()
        self.se_list=[]
        self.brelu_list = []
        self.conv_list = []

        for i in range(3):
            self.brelu_list.append([])
            self.conv_list.append([])
            self.se_list.append(SEBlock(filters,filters//r))
            if filters <= 128:
                tmp_brelu = BatchNormRelu(name=name + '_brelu' + str(i + 1) + '_1')
                tmp_conv = Conv2DFixedPadding(filters=filters,
                                              kernel_size=3,
                                              stride=1,
                                              name=name + '_conv' + str(i + 1) + '_1')
                self.brelu_list[-1].append(tmp_brelu)
                self.conv_list[-1].append(tmp_conv)
                tmp_brelu = BatchNormRelu(name=name + '_brelu' + str(i + 1) + '_2')
                tmp_conv = Conv2DFixedPadding(filters=filters,
                                              kernel_size=3,
                                              stride=1,
                                              name=name + '_conv' + str(i + 1) + '_2')
                self.brelu_list[-1].append(tmp_brelu)
                self.conv_list[-1].append(tmp_conv)
            else:
                tmp_brelu = BatchNormRelu(name=name + '_brelu' + str(i + 1) + '_1')
                tmp_conv = Conv2DFixedPadding(filters=filters // 4,
                                              kernel_size=1,
                                              stride=1,
                                              name=name + '_conv' + str(i + 1) + '_1')
                self.brelu_list[-1].append(tmp_brelu)
                self.conv_list[-1].append(tmp_conv)
                tmp_brelu = BatchNormRelu(name=name + '_brelu' + str(i + 1) + '_3')
                tmp_conv = Conv2DFixedPadding(filters=filters // 4,
                                              kernel_size=3,
                                              stride=1,
                                              name=name + '_conv' + str(i + 1) + '_3')
                self.brelu_list[-1].append(tmp_brelu)
                self.conv_list[-1].append(tmp_conv)
                tmp_brelu = BatchNormRelu(name=name + '_brelu' + str(i + 1) + '_3')
                tmp_conv = Conv2DFixedPadding(filters=filters,
                                              kernel_size=1,
                                              stride=1,
                                              name=name + '_conv' + str(i + 1) + '_3')
                self.brelu_list[-1].append(tmp_brelu)
                self.conv_list[-1].append(tmp_conv)

    def call(self, inputs, output_b, is_training):
        x = self.tconv1(inputs,is_training=is_training)

        """Cropping is only used when convolution padding is 'valid'"""
        src_shape = output_b.shape[1]
        tgt_shape = x.shape[1]
        start_pixel = int((src_shape - tgt_shape) / 2)
        end_pixel = start_pixel + tgt_shape

        cropped_b = output_b[:, start_pixel:end_pixel, start_pixel:end_pixel, :]
        """Assumes that data format is NHWC"""
        x = tf.concat([cropped_b, x], axis=-1)
        x = self.brelu1(x, is_training)
        x = self.conv1(x)

        for i in range(len(self.brelu_list)):
            y = x
            for j in range(len(self.brelu_list[i])):
                y = self.brelu_list[i][j](y, is_training=is_training)
                y = self.conv_list[i][j](y)
            x = x + y
            x=self.se_list[i](x)
        x = self.brelu2(x, is_training)

        return x


class PriorBlock(tf.keras.layers.Layer):
    """calculating Prior Block"""

    def __init__(self, filters, name=None):  # filters: number of the layers incorporated into the decoder
        super(PriorBlock, self).__init__(name=name)
        self.conv = Conv2DFixedPadding(filters=filters * 2, kernel_size=1, stride=1)

    def call(self, inputs, is_training):
        x = 0.1 * self.conv(inputs)
        s = x.get_shape().as_list()[3]
        mean = x[:, :, :, :s // 2]
        # mean =tf.keras.activations.tanh(mean)
        logstd = x[:, :, :, s // 2:]
        logstd = 3.0 * tf.keras.activations.tanh(logstd)
        std = K.exp(logstd)
        # var = K.abs(logvar)
        return tf.concat([mean, std], axis=-1)


@tf.function
def prob_function(inputs):
    # For sample method
    ts = inputs.get_shape()
    s = ts.as_list()
    s[3] = int(s[3] / 2)
    dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
    if ts[0] is None:
        samp = dist.sample([1, s[1], s[2], s[3]])
    else:
        samp = dist.sample([ts[0], s[1], s[2], s[3]])
    dis = 0.5*tf.math.multiply(samp, inputs[:, :, :, s[3]:])
    dis = tf.math.add(dis, inputs[:, :, :, 0:s[3]])
    return dis

@tf.function
def prob_function_(inputs,k):
    # For sample method
    ts = inputs.get_shape()
    s = ts.as_list()
    s[3] = int(s[3] / 2)
    dis=inputs[:, :, :, 0:s[3]]+k*inputs[:,:,:,s[3]:]
    return dis

@tf.function
def prob_given_variance(inputs,var):
    ts=inputs.get_shape()
    s=ts.as_list()
    s[3]=int(s[3]/2)
    dis=0.5*tf.math.multiply(var,inputs[:,:,:,s[3]:])
    dis=tf.math.add(dis,inputs[:,:,:,0:s[3]])
    return dis

@tf.function
def transform(x):
    s=x.get_shape()
    s=s.as_list()[3]//2
    x0=tf.keras.activations.tanh(x[:,:,:,:s])
    x1=tf.keras.activations.sigmoid(x[:,:,:,s:])
    x=tf.concat([x0,x1],axis=-1)
    return x


class Prob(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(Prob, self).__init__(name=name)

    def call(self, inputs):
        ts = inputs.get_shape()
        s = ts.as_list()
        s[3] = int(s[3] / 2)
        dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        if ts[0] is None:
            samp = dist.sample([1, s[1], s[2], s[3]])
        else:
            samp = dist.sample([ts[0], s[1], s[2], s[3]])
        dis = tf.math.multiply(samp, inputs[:, :, :, s[3]:])
        dis = tf.math.add(dis, inputs[:, :, :, 0:s[3]])
        return dis

class Encoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,num_filters,name=None):
        super(Encoder,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.conv=[]
        for i in range(num_layers):
            if i!=num_layers-1:
                self.conv.append(ConvBlock(num_filters[i],kernel_size=3,do_max_pool=True,name=name+'conv_'+str(i)))
            else:
                self.conv.append(ConvBlock(num_filters[i],kernel_size=3,do_max_pool=False,name=name+'conv_'+str(i)))
    def call(self,inputs,is_training=True):
        x=inputs
        b_list=[]
        for conv in self.conv:
            x,b=conv(x,is_training=is_training)
            b_list.append(b)
        return x,b_list

class VariationalEncoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,num_filters,name=None):
        super(VariationalEncoder,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.conv=[]
        self.fix_conv=[]
        self.conv0=[]
        for i in range(num_layers):
            if i!=num_layers-1:
                self.conv.append(ConvBlock(num_filters[i],kernel_size=3,do_max_pool=True,name=name+'conv_'+str(i)))
                self.fix_conv.append(Conv2DFixedPadding(filters=num_filters[i]*2,kernel_size=1,stride=1,name=name+'conv_fix_'+str(i)))
            else:
                self.conv.append(ConvBlock(num_filters[i],kernel_size=3,do_max_pool=False,name=name+'conv_'+str(i)))
                self.fix_conv.append(Conv2DFixedPadding(filters=num_filters[i]*2,kernel_size=1,stride=1,name=name+'conv_fix_'+str(i)))
            self.conv0.append(ConvBlock(num_filters[i],kernel_size=3,do_max_pool=False,name=name+'conv_after_'+str(i)))
    def call(self,inputs,is_training=True):
        prob=[]
        x=inputs
        b_list=[]
        for i in range(self.num_layers):
            x,b=self.conv[i](x,is_training=is_training)
            b=self.fix_conv[i](b)
            b=transform(b)
            prob.append(b)
            b=prob_function(b)
            b,_=self.conv0[i](b,is_training=is_training)
            b_list.append(b)
        return b_list,prob


    def given_known_dist(self,inputs,is_training=False):
        b_list=[]
        for i in range(self.num_layers):
            b,_=self.conv0[i](inputs[i],is_training=is_training)
            b_list.append(b)
        return b_list

    def get_style_representation(self,inputs,is_training=False):
        x=inputs
        b_list=[]
        for i in range(self.num_layers):
            x,b=self.conv[i](x,is_training=is_training)
            b=self.fix_conv[i](b)
            b=transform(b)
            s=b.get_shape().as_list()
            b_list.append(b[:,:,:,0:s[3]//2])
        return b_list




def expand_moments_dim(moment):
    return tf.reshape(moment, [-1, 1, 1, tf.shape(moment)[-1]])

@tf.function
def adain(content_feature, style_feature, eps=1e-5):
    content_mean, content_var = tf.nn.moments(content_feature, axes=[1, 2])
    style_mean, style_var = tf.nn.moments(style_feature, axes=[1, 2])

    content_std = tf.sqrt(content_var)
    style_std = tf.sqrt(style_var)

    content_mean = expand_moments_dim(content_mean)
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # content_mean = tf.broadcast_to(content_mean, tf.shape(content_feature))

    content_std = expand_moments_dim(content_std) + eps
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # content_std = tf.broadcast_to(content_std, tf.shape(content_feature))

    style_mean = expand_moments_dim(style_mean)
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # style_mean = tf.broadcast_to(style_mean, tf.shape(content_feature))

    style_std = expand_moments_dim(style_std) + eps
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # style_std = tf.broadcast_to(style_std, tf.shape(content_feature))

    normalized_content = tf.divide(content_feature - content_mean, content_std)
    return tf.multiply(normalized_content, style_std) + style_mean

class NormalDecoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,num_filters,s,name=None):
        super(NormalDecoder,self).__init__(name=name)
        self.num_layers=num_layers
        self.s=s
        self.num_filters=num_filters
        self.tconv=[]
        for i in range(num_layers):
            self.tconv.append(DeConvBlock(num_filters[i],name=name+'deconv_'+str(i)))
        self.conv=Conv2DFixedPadding(filters=s,kernel_size=1,stride=1,name=name+'conv_final')

    def call(self,inputs,b,is_training=True):
        x=inputs
        for i in range(self.num_layers):
            x=self.tconv[i](x,b[i],is_training=is_training)
        x=self.conv(x)
        x=tf.keras.activations.sigmoid(x)
        return x

class UNet(tf.keras.Model):
    def __init__(self,num_layers,num_filters,s,name=None):
        super(UNet,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.encoder=Encoder(num_layers,num_filters,name=name+'_content_encoder_')
        num_filters=num_filters[:-1]
        num_filters=num_filters[::-1]
        self.decoder=NormalDecoder(num_layers-1,num_filters,s,name=name+'_normal_decoder_')
    def call(self,inputs,is_training=True):
        x=inputs
        x,b=self.encoder(x,is_training=is_training)
        b=b[:-1]
        b=b[::-1]
        x=self.decoder(x,b,is_training=is_training)
        mse=K.square(x-inputs)
        mse=tf.reduce_mean(mse)
        self.add_loss(mse)
        return x

class StyleUNet(tf.keras.Model):
    def __init__(self,num_layers,num_filters,s,name=None):
        super(StyleUNet,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.s=s
        self.encoder=VariationalEncoder(num_layers,num_filters,name=name+'_style_encoder_')
        num_filters=num_filters[:-1]
        num_filters=num_filters[::-1]
        self.decoder=NormalDecoder(num_layers-1,num_filters,s,name=name+'_normal_decoder0_')
    def residual_kl_normal(self,y_prior):
        s=y_prior.get_shape().as_list()[3]
        mean_prior=y_prior[:,:,:,0:s//2]
        std_prior=y_prior[:,:,:,s//2:]
        mean_delta=mean_prior
        std_delta=std_prior
        first=math_ops.log(std_delta)
        second=0.5*math_ops.divide(K.square(mean_delta),K.square(std_prior))
        third=0.5*K.square(std_delta)
        loss=second+third-first-0.5
        loss=tf.reduce_mean(loss)
        return loss
    def call(self,inputs,is_training=True):
        x=inputs
        b,prob=self.encoder(x,is_training=is_training)
        x=b[-1]
        b=b[:-1]
        b=b[::-1]
        x=self.decoder(x,b,is_training=is_training)
        mse=K.square(x-inputs)
        mse=tf.reduce_mean(mse)
        loss=mse
        self.add_metric(mse,aggregation='mean',name='mse')
        i=0
        for p in prob:
            temp=self.residual_kl_normal(p)
            self.add_metric(temp,aggregation='mean',name='kl'+str(i))
            i+=1
            loss+=temp
        self.add_loss(loss)
        return x



class Decoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,num_filters,s,name=None):
        super(Decoder,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.s=s
        self.tconv=[]
        for i in range(num_layers):
            self.tconv.append(DeConvBlock(num_filters[i],name=name+'deconv_'+str(i)))
        self.conv=Conv2DFixedPadding(filters=s,kernel_size=1,stride=1,name=name+'conv_final_')

    def call(self,inputs,b_content,b_style,is_training=True):
        x=inputs
        x=adain(x,b_style[0])
        for i in range(self.num_layers):
            x=self.tconv[i](x,b_content[i],is_training=is_training)
            x=adain(x,b_style[i+1])
        x=self.conv(x)
        x=tf.keras.activations.sigmoid(x)
        return x

class StyleTransfer(tf.keras.Model):
    def __init__(self,num_layers,num_filters,sh,rec,p,s,tv,name=None):
        super(StyleTransfer,self).__init__(name=name)
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.content_encoder=Encoder(num_layers,num_filters,name=name+'_content_encoder_')
        self.style_encoder=VariationalEncoder(num_layers,num_filters,name=name+'_style_encoder_')
        num_layers-=1
        num_filters=num_filters[:-1]
        num_filters=num_filters[::-1]
        self.decoder=Decoder(num_layers,num_filters,sh,name=name+'_decoder_')
        VGG=VGG16()
        for layer in VGG.layers:
            layer.trainable=False
        self.VGGs=[]
        for i in range(1,6):
            if p[i-1]>0 or s[i-1]>0:
                name='block'+str(i)+'_conv1'
                vgg_input=VGG.input
                vgg_out=VGG.get_layer(name).output
                self.VGGs.append(tf.keras.Model(inputs=vgg_input,outputs=vgg_out))
                for layer in self.VGGs[i-1].layers:
                    layer.trainable=False
            else:
                self.VGGs.append(None)
        self.rec=rec
        self.p=p
        self.s=s
        self.tv=tv
        self.sh=sh

    def residual_kl_normal(self,y_prior):
        s=y_prior.get_shape().as_list()[3]
        mean_prior=y_prior[:,:,:,0:s//2]
        std_prior=y_prior[:,:,:,s//2:]
        mean_delta=mean_prior
        std_delta=std_prior
        first=math_ops.log(std_delta+0.00000001)
        second=0.5*math_ops.divide(K.square(mean_delta+0.00000001),K.square(std_prior+0.00000001))
        third=0.5*K.square(std_delta)
        loss=second+third-first-0.5
        loss=tf.reduce_mean(loss)
        return loss

    def reconstruction_loss(self,y_true,y_pred,weight):
        loss=K.square(y_true-y_pred)
        loss=weight*tf.reduce_mean(loss)
        self.add_metric(loss,name='reconstruction loss',aggregation='mean')
        return loss

    def perceptual_loss(self,y_true,y_pred,l,weight):
        loss=K.square(y_true-y_pred)
        loss=weight*tf.reduce_mean(loss)
        self.add_metric(loss,name='perceptual loss'+str(l),aggregation='mean')
        return loss

    def gram_matrix(self,x):
        x=K.permute_dimensions(x,(0,3,1,2))
        shape=K.shape(x)
        B,C,H,W=shape[0],shape[1],shape[2],shape[3]
        features=K.reshape(x,K.stack([B,C,H*W]))
        gram=K.batch_dot(features,features,axes=2)
        denominator=C*H*W
        gram=gram/K.cast(denominator,x.dtype)
        return gram

    def style_loss(self,y_true,y_pred,l,weight):
        y_true=self.gram_matrix(y_true)
        y_pred=self.gram_matrix(y_pred)
        loss=0.5*K.square(y_true-y_pred)
        loss=weight*tf.reduce_mean(loss)
        self.add_metric(loss,name='style loss'+str(l),aggregation='mean')
        return loss

    def deep_loss(self,y_true_content,y_true_style,y_pred,VGG_model,p,s):
        if self.sh==1:
            y_true_content=tf.concat([y_true_content,y_true_content,y_true_content],axis=-1)
            y_true_style=tf.concat([y_true_style,y_true_style,y_true_style],axis=-1)
            y_pred=tf.concat([y_pred,y_pred,y_pred],axis=-1)
        y_true_content=tf.image.resize(y_true_content,[224,224])
        y_true_style=tf.image.resize(y_true_style,[224,224])
        y_pred=tf.image.resize(y_pred,[224,224])
        y_true_content=preprocess_input(y_true_content*255)
        y_true_style=preprocess_input(y_true_style*255)
        y_pred=preprocess_input(y_pred*255)
        loss=0
        for i in range(len(VGG_model)):
            if p[i]>0 or s[i]>0:
                y_true_content_=VGG_model[i](y_true_content)
                y_true_style_=VGG_model[i](y_true_style)
                y_pred_=VGG_model[i](y_pred)
                loss+=self.perceptual_loss(y_true_content_,y_pred_,i,p[i])+self.style_loss(y_true_style_,y_pred_,i,s[i])
        return loss

    def total_variation_loss(self,y_pred,weight):
        loss=weight*tf.reduce_mean(tf.image.total_variation(y_pred))
        self.add_metric(loss,name='tv loss',aggregation='mean')
        return loss

    def total_loss(self,y_true_content,y_true_style,y_pred,VGG_model,rec,p,s,tv):
        loss=self.deep_loss(y_true_content,y_true_style,y_pred,VGG_model,p,s)
        if rec!=0:
            loss+=self.reconstruction_loss(y_true_content,y_pred,rec)
        if tv>0:
            loss+=self.total_variation_loss(y_pred,tv)
        return loss

    def training_loss(self,y_true_content,y_true_style,y_pred,VGG_model,rec,p,s,tv):
        loss=self.total_loss(y_true_content,y_true_style,y_pred,VGG_model,rec,p,s,tv)
        return loss

    def call(self,inputs,is_training=True):
        s=self.sh
        x_content=inputs[:,:,:,0:s]
        x_style=inputs[:,:,:,s:]
        x,b_content=self.content_encoder(x_content,is_training=is_training)
        b_style,prob=self.style_encoder(x_style,is_training=is_training)
        b_style=b_style[::-1]
        b_content=b_content[:-1]
        b_content=b_content[::-1]
        x=self.decoder(x,b_content,b_style,is_training=is_training)
        loss=0
        i=0
        for p in prob:
            temp=self.residual_kl_normal(p)
            self.add_metric(temp,aggregation='mean',name='kl'+str(i))
            loss+=temp
            i+=1
        loss+=self.total_loss(x_content,x_style,x,self.VGGs,self.rec,self.p,self.s,self.tv)
        self.add_loss(loss)
        return x

    def get_style_representation(self,inputs,is_training=False):
        b=self.style_encoder.get_style_representation(inputs,is_training=is_training)
        return b


    def sample(self,inputs,p,is_training=False):
        temp=[]
        s=self.sh
        x_content=inputs[:,:,:,0:s]
        x_style=inputs[:,:,:,s:]
        _,prob=self.style_encoder(x_style,is_training=is_training)
        for i in range(self.num_layers):
            if i==0:
                temp.append(prob_function_(prob[i],p))
            else:
                temp.append(prob_function_(prob[i],0))
        b_style=self.style_encoder.given_known_dist(temp,is_training=is_training)
        b_style=b_style[::-1]
        x,b_content=self.content_encoder(x_content,is_training=is_training)
        b_content=b_content[:-1]
        b_content=b_content[::-1]
        x=self.decoder(x,b_content,b_style,is_training=is_training)
        return x


    def given_known_dist(self,inputs,b,is_training=False):
        s=self.sh
        x_content=inputs[:,:,:,0:s]
        x,b_content=self.content_encoder(x_content,is_training=is_training)
        b_style=self.style_encoder.given_known_dist(b,is_training=is_training)
        b_style=b_style[::-1]
        b_content=b_content[:-1]
        b_content=b_content[::-1]
        x=self.decoder(x,b_content,b_style,is_training=is_training)
        return x



    def pseudo_sample(self,inputs,p,is_training=False):
        temp=[]
        s=inputs.get_shape().as_list()
        for i in range(self.num_layers):
            x=np.ones((s[0],s[1],s[2],self.num_filters[i]))*p
            x=tf.convert_to_tensor(x,tf.float32)
            temp.append(x)
        b_style=self.style_encoder.given_known_dist(temp,is_training=is_training)
        b_style=b_style[::-1]
        x,b_content=self.content_encoder(inputs,is_training=is_training)
        b_content=b_content[:-1]
        b_content=b_content[::-1]
        x=self.decoder(x,b_content,b_style,is_training=is_training)
        return x







