import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


def feature_extraction_net(input_shape):
    resnet_inception = tf.keras.applications.InceptionResNetV2(pooling='avg', include_top=False)
    resnet_inception.trainable = False
    x = x_in = Input(shape=input_shape)
    x = resnet_inception(x)
    model = Model(inputs=x_in, outputs=x)
    return model