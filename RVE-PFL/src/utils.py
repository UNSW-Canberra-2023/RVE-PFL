#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import numpy as np 
from tensorflow.keras import backend as K
import tensorflow as tf 
from torch.utils.data import ConcatDataset

def categorical_crossentropy(y_true, y_pred):
    return K.eval(K.categorical_crossentropy(y_true, y_pred))

def precision(y_true, y_pred):
            """Precision metric.
        
            Only computes a batch-wise average of precision.
        
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives =  tf.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))), tf.float32)
            predicted_positives = tf.cast( K.sum(K.round(K.clip(y_pred, 0, 1))), tf.float32)
            
            precision = true_positives / (predicted_positives + K.epsilon())
            return K.eval(precision)
        
def recall(y_true, y_pred):
            """Recall metric.
        
            Only computes a batch-wise average of recall.
        
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = tf.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))), tf.float32)
            possible_positives = tf.cast(K.sum(K.round(K.clip(y_true, 0, 1))), tf.float32)
            recall = true_positives / (possible_positives + K.epsilon())
            return K.eval(recall)

def f1(y_true, y_pred):
            def recall(y_true, y_pred):
                true_positives = tf.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))), tf.float32)
                possible_positives = tf.cast(K.sum(K.round(K.clip(y_true, 0, 1))), tf.float32)
                recall = true_positives / (possible_positives + K.epsilon())
                return recall
        
            def precision(y_true, y_pred):
                true_positives =  tf.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))), tf.float32)
                predicted_positives =  tf.cast(K.sum(K.round(K.clip(y_pred, 0, 1))), tf.float32)
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision
        
            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return K.eval(2*((precision*recall)/(precision+recall+K.epsilon())))
        
def accuracy(y_true, y_pred):
    #y_pred = np.round(y_pred)
    correct = K.cast(K.equal(y_true, y_pred), K.floatx())
    accuracy = K.mean(correct)
    return K.eval(accuracy)     
        
def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
           [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
        combined_dataset = ConcatDataset([train_dataset, test_dataset])
        print(f"Number of samples in the full dataset: {len(combined_dataset)}")
        #combined_dataset= torch.utils.data.random_split(combined_dataset, [len(train_dataset), len(test_dataset)])[0]
        #print(f"Number of samples in the full dataset: {len(combined_dataset)}")
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(combined_dataset, args.num_users)
            user_groups_test = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(combined_dataset, args.num_users)

    elif args.dataset == 'mnist' or args.dataset =='fmnist' or args.dataset  =='FashionMNIST':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        elif  args.dataset == 'FashionMNIST' :
            data_dir = '../data/FashionMNIST/'
        elif  args.dataset == 'fmnist' :
            data_dir = '../data/fmnist/'
            
            
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        
        if args.dataset == 'FashionMNIST' :
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
    
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
            combined_dataset = ConcatDataset([train_dataset, test_dataset])
        
        if args.dataset == 'mnist' or  args.dataset == 'fmnist' :
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
    
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        
            combined_dataset = ConcatDataset([train_dataset, test_dataset])


        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(combined_dataset, args.num_users)
            user_groups_test = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(combined_dataset, args.num_users)
                user_groups_test = cifar_iid(test_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(combined_dataset, args.num_users)
                user_groups_test = cifar_iid(test_dataset, args.num_users)

    return combined_dataset,train_dataset, test_dataset, user_groups,user_groups_test




def average_weights(w):
    """
    Returns the average of the weights.
    """
    # w_avg = copy.deepcopy(w[0])
    # for key in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[key] += w[i][key]
    #     w_avg[key] = torch.div(w_avg[key], len(w))
    # Get the length of the lists
    result = list()
    n = len(w)
    # Get the length of the first list (all lists should have the same length)
    m = len(w[0])
    # Create a list of zeros with the same length as the first list
    
    for i in range(m):
      l = [0]* len(w[0][i])
      for j in range(n):
        l = [ x+y for x,y in zip(l,w[j][i])]
        
      result.append(l)
            
    return result


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
