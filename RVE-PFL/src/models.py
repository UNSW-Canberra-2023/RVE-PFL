#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import math 
import torch

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, BatchNormalization, Dropout,Reshape

import numpy as np
from tensorflow.keras.models import Model

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
    
    




# Global Classifier   
class Classifier(nn.Module):
    def __init__(self,args):
        super(Classifier, self).__init__()
        # Set Generic Variables
        self.latent_size = args.latent_size
        self.filter_shape = args.filter_shape
        self.num_classes = args.num_classes
        
        
        #For Classifier
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.filter_shape**2*32, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(0.25)
        self.maxpool = nn.MaxPool2d(2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)

    def forward(self,x):
        #print(x.shape)
        #x = x.view(-1, 1, int(math.sqrt(self.latent_size)), int(math.sqrt(self.latent_size))) # reshape to (batch_size, 1, 28, 28)
        
        x = self.maxpool(self.batchnorm1(torch.relu(self.conv1(x)))) # 2D convolution, max pooling and batch normalization
        x = self.maxpool(self.batchnorm2(torch.relu(self.conv2(x))))
        x = x.view(-1, self.filter_shape**2*32) # flatten
        x = self.dropout(torch.relu(self.fc1(x)))
        logits = self.fc2(x)
        #probas = F.softmax(logits, dim=1)
        return logits



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class VAE_CNN:
    def __init__(self, latent_size=784, filter_shape=7, num_classes=10, **kwargs):
        super(VAE_CNN, self).__init__(**kwargs)
        self.latent_size = latent_size
        self.filter_shape = filter_shape
        self.num_classes = num_classes
     
        

        

    def reparameterize(self, mu, logvar):
        z_mean, z_log_var = mu,logvar
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build(self,latent_size, n_dim):
        self.latent_size = latent_size
        x= Input(shape=(int(np.sqrt(latent_size/n_dim)), int(np.sqrt(latent_size/n_dim)),n_dim))
        conv_encode_1 = Conv2D(32, (3, 3), strides=(1, 1), input_shape=(np.sqrt(latent_size/n_dim), np.sqrt(latent_size/n_dim),n_dim), name="encoder1_layer1", activation='relu')(x)
        conv_encode_2 = Conv2D(64, (3, 3), strides=(1, 1), name="encoder1_layer2", activation='relu') (conv_encode_1)
        flatten_encode_3 =   Flatten(name="encoder1_layer3")(conv_encode_2)
        linear1_encode_4 = Dense(512, name="encoder1_layer4")(flatten_encode_3)
        droup_out = Dropout(0.25,  name="encoder1_layer5")(linear1_encode_4)
        mu = Dense(self.latent_size, name="encoder1_layer6")(droup_out)
        logvar = Dense(self.latent_size, name="encoder1_layer7")(droup_out)

        z = self.reparameterize(mu, logvar)

        input_classifier = Reshape((int(np.sqrt(self.latent_size/n_dim)), int(np.sqrt(self.latent_size/n_dim)),n_dim) )(z)
        conv1 = Conv2D(32, (3, 3), padding='same', input_shape=(int(np.sqrt(self.latent_size/n_dim)), int(np.sqrt(self.latent_size/n_dim)),n_dim),activation='relu', name="classifier_layer1")(input_classifier)
        maxpool_1 = MaxPool2D((2, 2), name="classifier_layer2")(conv1)
        #batchnorm1 = BatchNormalization( name="classifier_layer3")(maxpool_1)
        conv2 = Conv2D(64, (3, 3), padding='same',activation='relu',  name="classifier_layer4" )(maxpool_1)
        maxpool_2 = MaxPool2D((2, 2), name="classifier_layer5")(conv2)
        #batchnorm2 = BatchNormalization(name="classifier_layer6")(maxpool_2)
        flatten =   Flatten(name="classifier_layer7")(maxpool_2)
        fc1 = Dense(256, activation='relu', name="classifier_layer8")(flatten)
        droup_out = Dropout(0.25,  name="classifier_layer9")(fc1)
        fc2 = Dense(self.num_classes, activation='softmax', name="classifier_layer10")(droup_out)

        model =  Model(inputs=x, outputs=fc2)
        
        
        #model.add_loss(get_loss(mu, logvar))
        
        return model


class CNN:
    def __init__(self, latent_size=784, filter_shape=7, num_classes=10, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.latent_size = latent_size
        self.filter_shape = filter_shape
        self.num_classes = num_classes
     
        

        

    def reparameterize(self, mu, logvar):
        z_mean, z_log_var = mu,logvar
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build(self):

        x= Input(shape=(int(np.sqrt(self.latent_size)), int(np.sqrt(self.latent_size)),1))
        conv1 = Conv2D(32, (3, 3), padding='same', input_shape=(int(np.sqrt(self.latent_size)), int(np.sqrt(self.latent_size)),1),activation='relu', name="classifier_layer1")(x)
        maxpool_1 = MaxPool2D((2, 2), name="classifier_layer2")(conv1)
        batchnorm1 = BatchNormalization( name="classifier_layer3")(maxpool_1)
        conv2 = Conv2D(64, (3, 3), padding='same',activation='relu',  name="classifier_layer4" )(batchnorm1)
        maxpool_2 = MaxPool2D((2, 2), name="classifier_layer5")(conv2)
        batchnorm2 = BatchNormalization(name="classifier_layer6")(maxpool_2)
        flatten =   Flatten(name="encoder1_layer7")(batchnorm2)
        fc1 = Dense(256, activation='relu', name="classifier_layer8")(flatten)
        droup_out = Dropout(0.25,  name="classifier_layer9")(fc1)
        fc2 = Dense(self.num_classes, activation='softmax', name="classifier_layer10")(droup_out)

        model =  Model(inputs=x, outputs=fc2)

        return model
