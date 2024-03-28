from torch import nn
import torch.nn.functional as F
import torch

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, BatchNormalization, Dropout,Reshape

import numpy as np
from keras.models import Model

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from keras import layers
from keras import backend as K
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import os 
import copy
import tensorflow_datasets as tfds




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
        #batchnorm1 = BatchNormalization( name="classifier_layer3")(maxpool_1)
        conv2 = Conv2D(64, (3, 3), padding='same',activation='relu',  name="classifier_layer4" )(maxpool_1)
        maxpool_2 = MaxPool2D((2, 2), name="classifier_layer5")(conv2)
        #batchnorm2 = BatchNormalization(name="classifier_layer6")(maxpool_2)
        flatten =   Flatten(name="classifier_layer7")(maxpool_2)
        fc1 = Dense(256, activation='relu', name="classifier_layer8")(flatten)
        droup_out = Dropout(0.25,  name="classifier_layer9")(fc1)
        fc2 = Dense(self.num_classes, activation='softmax', name="classifier_layer10")(droup_out)

        model =  Model(inputs=x, outputs=fc2)

        return model
      


# Create ToTensor and ToPILImage equivalent functions
def to_tensor(image):
    return tf.image.convert_image_dtype(image, dtype=tf.float32)

def to_pil_image(image):
    return tf.keras.preprocessing.image.array_to_img(image)

def label_to_onehot(target, num_classes=10):
    onehot_target = tf.one_hot(target, depth=num_classes)
    return onehot_target


    

index = 0


CNN = CNN()
model = CNN.build()


for class_index in range(10) :

        read_from_dict =1  
        weights_dict ={}
        L = ['classifier_layer1', 'classifier_layer4', 'classifier_layer8', 'classifier_layer10']
        if read_from_dict ==1:
            with open('classifier_weights_dict_6.pkl', 'rb') as f:
                weights_dict = pickle.load(f)
            i=0
            for layer_name, weights in weights_dict.items():
                    l_name = L[i]
                    i=i+1
                    model.get_layer(l_name).set_weights(weights)
        else:
            model.load_weights('personalized_model_user_0_epoch_0.h5')

        index +=  1
        

        gt_label =class_index
        gt_onehot_label =to_tensor( label_to_onehot(gt_label))
        gt_onehot_label = tf.reshape(gt_onehot_label, (1, -1))

        criterion = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.SGD()

        batch_size = 1 

        ds, ds_info = tfds.load('mnist', split='test', shuffle_files=True, with_info=True)
        ds = ds.batch(batch_size)
        gradients_accumulated = [tf.zeros_like(var) for var in model.trainable_variables]
        
        images_folder = 'Results'

        # Specify the file format (e.g., 'png', 'jpg', etc.)
        file_format = 'png'
      
        count = 0
        for gt_data,label in ds:            
            if label[0] == class_index:
                    print(label[0])
                    count +=1
                    gt_onehot_label = to_tensor( label_to_onehot( label))  # Adjust this based on your dataset structure
                    with tf.GradientTape(persistent=True) as tape:
                        pred = model(gt_data)
                        y = criterion(gt_onehot_label, pred)

                    # Get gradients
                    gradients = tape.gradient(y, model.trainable_variables)

                    # Accumulate gradients;
                    gradients_accumulated = [grad_accum + grad for grad_accum, grad in zip(gradients_accumulated, gradients)]
                    # Clean up the tape
                    break
                    del tape

        # Optionally, you can calculate the average gradient if needed
        original_dy_dx =gradients_accumulated #gradients # [grad_accum    for grad_accum in gradients_accumulated]

        # Generate dummy data and label
        dummy_data = tf.Variable(tf.random.normal(shape=(1,28,28,1)), trainable=True)
        dummy_label = tf.Variable(tf.random.normal(shape=(1,10)), trainable=True)


 
        
        history = []
        # Run the optimization loop
        for iters in range(2000):
            with tf.GradientTape(persistent=True) as tape:
                dummy_pred = model(dummy_data)
                dummy_onehot_label = tf.nn.softmax(dummy_label, axis=-1)
                dummy_loss = criterion(dummy_pred, dummy_onehot_label)

                dummy_dy_dx = tape.gradient(dummy_loss,  model.trainable_variables)

                # Compute the gradient difference
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += tf.reduce_sum(tf.square(gx - gy))

                # Update the dummy data and label
                gradients = tape.gradient(grad_diff, [dummy_data, dummy_label])
                optimizer.apply_gradients(zip(gradients, [dummy_data, dummy_label]))
                #print(dummy_label)

                if iters % 10 == 0:
                    print(iters, "%.4f" % grad_diff.numpy())
                    history.append(dummy_data.numpy()[0, :, :, 0])
                    img = dummy_data.numpy()[0, :, :, :]
                    #print(img.shape)
                    plt.imsave(os.path.join(images_folder, f'{index}_iter_{iters * 10}.{file_format}'), to_pil_image(img),cmap = 'coolwarm')    

   
   
   

