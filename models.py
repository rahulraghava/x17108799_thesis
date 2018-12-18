# import the necessary packages
from keras.models import Model
from keras.layers import Input, Activation, merge, Flatten, Dropout, MaxPooling2D, AveragePooling2D, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras import backend as K
from keras.applications.densenet import DenseNet121
 
class lenet5:
    def __init__(self, classes, height = 32, width = 32, depth = 3):
        self.height = height
        self.width = width
        self.depth = depth
        self.classes = classes
    def build(self):
        inputShape = (self.height, self.width, self.depth)
        if K.image_data_format() == "channels_first":
            inputShape = (self.depth, self.height, self.width)
        input_img=Input(shape=inputShape, name="input_img")
        conv_1 = Conv2D(20,(5,5), padding='same', name='conv_1')(input_img)
        relu_1 = Activation("relu", name='relu_1')(conv_1)
        maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name="maxpool_1")(relu_1)
        conv_2 = Conv2D(50,(5,5), padding='same', name='conv_2')(maxpool_1)
        relu_2 = Activation("relu", name='relu_2')(conv_2)
        maxpool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name="maxpool_2")(relu_2)
        flatten = Flatten(name='flatten')(maxpool_2)
        full_connected_1 = Dense(500, name='full_connected_1')(flatten)
        relu_3 = Activation("relu", name="relu_3")(full_connected_1)
        full_connected_2 = Dense(self.classes, name='full_connected_2')(relu_3)
        softmax = Activation("softmax", name='softmax')(full_connected_2)
        return Model(inputs=input_img,outputs=softmax)

class lenet5_64:
    def __init__(self, classes, height = 64, width = 64, depth = 3):
        self.height = height
        self.width = width
        self.depth = depth
        self.classes = classes
    def build(self):
        inputShape = (self.height, self.width, self.depth)
        if K.image_data_format() == "channels_first":
            inputShape = (self.depth, self.height, self.width)
        input_img=Input(shape=inputShape, name="input_img")
        conv_1 = Conv2D(20,(5,5), padding='same', name='conv_1')(input_img)
        relu_1 = Activation("relu", name='relu_1')(conv_1)
        maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name="maxpool_1")(relu_1)
        conv_2 = Conv2D(50,(5,5), padding='same', name='conv_2')(maxpool_1)
        relu_2 = Activation("relu", name='relu_2')(conv_2)
        dropout = Dropout(0.5, name='dropout')(relu_2)
        maxpool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name="maxpool_2")(dropout)
        flatten = Flatten(name='flatten')(maxpool_2)
        full_connected_1 = Dense(500, name='full_connected_1')(flatten)
        relu_3 = Activation("relu", name="relu_3")(full_connected_1)
        full_connected_2 = Dense(self.classes, name='full_connected_2')(relu_3)
        softmax = Activation("softmax", name='softmax')(full_connected_2)
        return Model(inputs=input_img,outputs=softmax)


class squeezenet:
    def __init__(self, classes, height = 224, width = 224, depth = 3):
        self.height = height
        self.width = width
        self.depth = depth
        self.classes = classes
    def build(self):
        inputShape = (self.height, self.width, self.depth)
        if K.image_data_format() == "channels_first":
            inputShape = (self.depth, self.height, self.width)
        input_img = Input(shape=inputShape)
        conv_1 = Conv2D(96,(7,7), padding = "same", activation='relu', strides=(2,2), name='conv_1')(input_img)
        max_1 = MaxPooling2D((3,3), strides=(2,2), name='maxpool_1')(conv_1)
        fire1 = self.firemodule(16,max_1, 1)
        fire2 = self.firemodule(16,fire1, 2)
        fire3 = self.firemodule(16, fire2,3)
        maxpool_2 = MaxPooling2D((3,3), strides=(2,2), name= 'maxpool_2')(fire3)
        fire4 = self.firemodule(32, maxpool_2, 4)
        fire5 = self.firemodule(48, fire4,5)
        fire6 = self.firemodule(48, fire5,6)
        fire7 = self.firemodule(64, fire6, 7)
        maxpool_3 =  MaxPooling2D((3,3), strides=(2,2), name= 'maxpool_3')(fire7)
        fire8 = self.firemodule(64, maxpool_3, 8)
        fire_dropout = Dropout(0.5,name='dropout')(fire8)
        conv_2 = Conv2D(self.classes, (1,1), name= 'conv_2')(fire_dropout)
        avgpool = AveragePooling2D(pool_size=(13,13), name='avgpool')(conv_2)
        flatten = Flatten(name='flatten')(avgpool)
        softmax = Activation("softmax", name='softmax')(flatten)
        return Model(inputs=input_img, outputs= softmax)
    def firemodule(self, n , inp , series):
        fi_name = 'fire' + str(series)
        fire_squeeze = Conv2D(n,(1,1), activation='relu', padding='same', name= fi_name+'_squeeze')(inp)
        fire_expand_1 = Conv2D(4*n,(1,1), activation='relu', padding='same', name=fi_name + '_expand_1')(fire_squeeze)
        fire_expand_2 = Conv2D(4*n,(3,3), activation= 'relu', padding='same',name = fi_name + '_expand_2')(fire_squeeze)
        return concatenate([fire_expand_1,fire_expand_2],axis=-1 )

class densenet:
    def __init__(self, classes, height = 224, width = 224, depth = 3):
        self.height = height
        self.width = width
        self.depth = depth
        self.classes = classes
    def build(self):
        return DenseNet121(include_top=True, weights=None, input_shape=(self.height, self.width, self.depth), pooling=None, classes= self.classes)