#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf 
from keras.utils import to_categorical
import numpy as np 
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import precision, recall, f1, accuracy#, categorical_crossentropy
from keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from numpy.random import RandomState

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


 


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image , label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger     #, self.testloader 
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        
        rng = RandomState()
        rng.shuffle(idxs)
        #idxs_train = idxs.sample(frac=0.8, random_state=rng)
        #idxs_test = idxs.loc[~idxs.isin(idxs_train)]
        idxs_train = idxs[int(0.1*len(idxs)):]
        #idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[0:int(0.1*len(idxs))]

        trainloader=  DataLoader(DatasetSplit(dataset, idxs_train),
                                batch_size= len(idxs_train), shuffle=False)
        #,
                                # batch_size=self.args.local_bs, shuffle=True)
        # = DataLoader(DatasetSplit(dataset, idxs_val),
                  #               batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                               batch_size=len(idxs_test), shuffle=False)
        return trainloader , testloader  #trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # # Set mode to train model
        # model.train()
        # epoch_loss = []

        # # Set optimizer for the local updates
        # if self.args.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
        #                                 momentum=0.5)
        # elif self.args.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)#,
        #                                  #weight_decay=1e-4)

        # for iter in range(self.args.local_ep):
        #     batch_loss = []
        #     for batch_idx, (images, labels) in enumerate(self.trainloader):
        #         images, labels = images.to(self.device), labels.to(self.device)
        #         #print(images.shape)

        #         model.zero_grad()
        #         log_probs = model(images)
        #         loss = self.criterion(log_probs, labels)
        #         loss.backward()
        #         optimizer.step()

        #         if self.args.verbose and (batch_idx % 100 == 0):
        #             print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 global_round, iter, batch_idx * len(images),
        #                 len(self.trainloader.dataset),
        #                 100. * batch_idx / len(self.trainloader), loss.item()))
        #         self.logger.add_scalar('loss', loss.item())
        #         batch_loss.append(loss.item())
        #     epoch_loss.append(sum(batch_loss)/len(batch_loss))


        for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
               
                images = torch.transpose(images, 1, 3)
                images = torch.transpose(images, 1, 2)    
                labels = to_categorical(labels, num_classes=10)

                model.fit(images, labels, nb_epochs=self.args.local_ep, batch_size=self.args.local_bs)
                average_loss = np.mean(model.model.history.history['loss'])
                average_acc = np.mean(model.model.history.history['accuracy'])

        model.model.save_weights("weights.h5")
        return model.model.get_weights(), average_loss, average_acc

    
    
    
    
    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        
        
        for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                images = torch.transpose(images, 1, 3)
                images = torch.transpose(images, 1, 2)
                labels_one_hot = to_categorical(labels, num_classes=10)
                
                
                labels_pred= model.predict(images)
                
                labels_pred_values = np.argmax(labels_pred, axis=1)
                labels = labels.numpy().astype(np.int64)
                
                loss = np.mean(K.eval(categorical_crossentropy(labels_one_hot,labels_pred)))
                # Calculate accuracy
                acc = accuracy_score(labels, labels_pred_values)
                
                # Calculate precision
                pre = precision_score(labels, labels_pred_values, average='macro')

                # Calculate recall
                rec = recall_score(labels, labels_pred_values,average='macro')

                # Calculate f1-score
                f1_score_r = f1_score(labels, labels_pred_values, average='macro')
                
                #results = model.evaluate(images, labels, verbose=0)
        
        return acc, loss, pre, rec, f1_score_r


    def test_inference(self, args, model):
        
        """ Returns the test accuracy and loss.
        """
    
        loss, total, correct = 0.0, 0.0, 0.0
    
        device = 'cuda' if args.gpu else 'cpu'
        # testloader = DataLoader(test_dataset, batch_size=len(test_dataset),
        #                         shuffle=False)
    
        for batch_idx, (images, labels) in enumerate(self.testloader):
                    images, labels = images.to(device), labels.to(device)
                    images = torch.transpose(images, 1, 3)
                    images = torch.transpose(images, 1, 2)
                    labels_loss = to_categorical(labels, num_classes=10)
                    pred = model.predict(images)
                    loss = np.mean(K.eval(categorical_crossentropy(labels_loss,pred)))
                    
                    
                    labels_pred = np.argmax(pred, axis=1)
                    labels_pred = labels_pred.astype(np.int64)
                    labels = labels.numpy().astype(np.int64)
                    
                    #print(labels_pred.dtype)
                    #print(labels.dtype)
                    # Calculate accuracy
                    acc = accuracy_score(labels, labels_pred)
                    
                    # Calculate precision
                    pre = precision_score(labels, labels_pred,average='macro')
    
                    # Calculate recall
                    rec = recall_score(labels, labels_pred, average='macro')
    
                    # Calculate f1-score
                    f1_score_r = f1_score(labels, labels_pred, average='macro')
                    
                    #loss = categorical_crossentropy(labels,labels_pred)
                    
                    #results = model.evaluate(images, labels, verbose=0)
            
        return acc,loss, pre, rec, f1_score_r


def get_final_test(dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        
        #rng = RandomState()
        #rng.shuffle(idxs)
        #idxs_train = idxs.sample(frac=0.8, random_state=rng)
        #idxs_test = idxs.loc[~idxs.isin(idxs_train)]
        #idxs_train = idxs[int(0.1*len(idxs)):]
        #idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[:int(len(idxs))]

        #trainloader=  DataLoader(DatasetSplit(dataset, idxs_train),
         #                       batch_size= len(idxs_train), shuffle=False)
        #,
                                # batch_size=self.args.local_bs, shuffle=True)
        # = DataLoader(DatasetSplit(dataset, idxs_val),
                  #               batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                               batch_size=len(idxs_test), shuffle=False)
        return  testloader  #trainloader, validloader, testloader

def test_inference(args, model,test_data,test_idxs):
         
         """ Returns the test accuracy and loss.
         """
         final_testloader = get_final_test(test_data, list(test_idxs))
         #loss, total, correct = 0.0, 0.0, 0.0
     
         device = 'cuda' if args.gpu else 'cpu'
         # testloader = DataLoader(test_dataset, batch_size=len(test_dataset),
         #                         shuffle=False)
     
         for batch_idx, (images, labels) in enumerate(final_testloader):
                        images, labels = images.to(device), labels.to(device)
                        images = torch.transpose(images, 1, 3)
                        images = torch.transpose(images, 1, 2)
                        labels_loss = to_categorical(labels, num_classes=10)
                        pred = model.predict(images)
                        loss = np.mean(K.eval(categorical_crossentropy(labels_loss,pred)))
                        
                        
                        labels_pred = np.argmax(pred, axis=1)
                        labels_pred = labels_pred.astype(np.int64)
                        labels = labels.numpy().astype(np.int64)
                        
                        #print(labels_pred.dtype)
                        #print(labels.dtype)
                        # Calculate accuracy
                        acc = accuracy_score(labels, labels_pred)
                        
                        # Calculate precision
                        pre = precision_score(labels, labels_pred,average='macro')
        
                        # Calculate recall
                        rec = recall_score(labels, labels_pred, average='macro')
        
                        # Calculate f1-score
                        f1_score_r = f1_score(labels, labels_pred, average='macro')
                        
                        #loss = categorical_crossentropy(labels,labels_pred)
                        
                        #results = model.evaluate(images, labels, verbose=0)
                
         return acc,loss, pre, rec, f1_score_r