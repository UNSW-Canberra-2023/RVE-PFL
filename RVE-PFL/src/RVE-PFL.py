#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate , test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar,VAE_CNN, Classifier, VAE_CNN,CNN
from utils import get_dataset, average_weights, exp_details,precision, recall, f1

from keras.optimizers import Adam

from art.estimators.classification import KerasClassifier
import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
import sys
import warnings
warnings.filterwarnings("ignore")
import os
import csv 
from keras.losses import categorical_crossentropy
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
script_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    #Determine the device
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    combined_dataset,train_dataset, test_dataset, user_groups, user_groups_test= get_dataset(args)

    # Build global model
    latent_size = 784
    n_dim =1
    if args.dataset =="mnist" or args.dataset =="fmnist":
          latent_size = 784
          n_dim = 1
    elif args.dataset =="cifar":
          latent_size = 3072
          n_dim = 3
    VAE = VAE_CNN()
    model = VAE.build(latent_size =  latent_size,n_dim = n_dim)
    #custom_loss = lambda y_true, y_pred: dataset

    model.compile(optimizer= tf.keras.optimizers.legacy.Adam(lr=1e-3, epsilon=1e-7, decay=1e-3), loss=categorical_crossentropy, metrics=['accuracy'])  #loss=categorical_crossentropy,
    global_model = KerasClassifier(clip_values=(0, 255),  model=model, use_logits=False)
    

    #save_weight_initial_model
    global_model.model.save_weights('initial_global_model.h5')
    global_weights = global_model.model.get_weights()
    
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    
    #Intialize the accumulated weights
    init_weights = global_model.model.get_weights()   
    weights_accumulated = np.array( init_weights, dtype='object')*0   # initialize empty weights    
    weights_dict = {}
    classifier_layers = ['classifier_layer1', 'classifier_layer11','classifier_layer3', 'classifier_layer4', 'classifier_layer41','classifier_layer6', 'classifier_layer8', 'classifier_layer10']             

    #Headers for excel sheets to save model evaluations
    header_client_training = ['client#', 'Round#', 'Training Loss', 'ACC',"Pre","Recall","f1-score"] 
    header_client_testing = ['client#', 'Round#', 'Testing Loss', 'ACC',"Pre","Recall","f1-score"]
    header_epoch_training = ['Round#', 'Training Loss', 'ACC',"Pre","recall","f1-score"] 
    header_epoch_testing = ['Round#', 'Testing Loss', 'ACC','Pre','Recall','f1-score']
    general_test_client= ['client#', 'Round#', 'Testing Loss', 'ACC','Pre','Recall','f1-score']
    general_test_epoch = ['Round#', 'Testing Loss', 'ACC','Pre','Recall','f1-score']



    if os.path.exists("client_training.csv"):
            os.remove("client_training.csv")

    if os.path.exists("client_testing.csv"):
            os.remove("client_testing.csv")
 
    if os.path.exists("epoch_training.csv"):
            os.remove("epoch_training.csv")

    if os.path.exists("epoch_testing.csv"):
            os.remove("epoch_testing.csv")
            
    if os.path.exists("general_test_clients.csv"):
            os.remove("general_test_clients.csv")
            
    if os.path.exists("general_test_epochs.csv"):
                os.remove("general_test_epochs.csv")

            
    with open('client_training.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_client_training)

    with open('client_testing.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_client_testing)

    with open('epoch_training.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_epoch_training)

    with open('epoch_testing.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_epoch_testing)
 
    with open('general_test_clients.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(general_test_client)
        
 
    with open('general_test_epochs.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(general_test_epoch)        
            
    #Start Global Rounds 
    
    for epoch in tqdm(range(args.epochs)):
        
        local_losses, local_acc, local_pre, local_recall, local_f1  = [], [], [], [], []
        local_losses_test, local_acc_test, local_pre_test, local_recall_test, local_f1_test  = [], [], [], [], []
        general_test_acc=[]
        general_test_loss=[]
        general_test_pre=[]
        general_test_recall=[]
        general_test_f1=[]
        print(f'\n | Global Training Round : {epoch+1} |\n')

        idxs_users =range(args.num_users) #np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            
            print("User # "+str(idx)+" starts local training in Global round # "+str(epoch))
            weights_dict_users = {}
            
            if epoch ==0:
            
                global_model.model.load_weights('initial_global_model.h5')
                r = [0,0,0,0,0,0]
                general_test_acc.append(r[0]) 
                general_test_loss.append(r[1]) 
                general_test_pre.append(r[2]) 
                general_test_recall.append(r[3]) 
                general_test_f1.append(r[4]) 
                general_test_row_client = [idx+1, 0, r[1], r[0], r[2], r[3], r[4]]  
                
                with open('general_test_clients.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(general_test_row_client)
                
            else:
                
                global_model.model.load_weights("personalized_model_user_"+ str(idx)+".h5")
                
                for layer in global_model.model.layers:
                    
                        if len(layer.get_weights()) > 0 and layer.name in classifier_layers:
                            
                                 layer.set_weights(weights_dict[layer.name])
            
                
            if epoch !=0 :
                     
                local_model = LocalUpdate(args=args, dataset=combined_dataset,
                                      idxs=user_groups[idx], logger=logger)
                r = local_model.test_inference(args, global_model)
                general_test_acc.append(r[0]) 
                general_test_loss.append(r[1]) 
                general_test_pre.append(r[2]) 
                general_test_recall.append(r[3]) 
                general_test_f1.append(r[4]) 
                general_test_row_client = [idx+1, epoch, r[1], r[0], r[2], r[3], r[4]]  
                
                with open('general_test_clients.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(general_test_row_client)
                    
                                    
            local_model = LocalUpdate(args=args, dataset=combined_dataset,
                                      idxs=user_groups[idx], logger=logger)
            
            w, loss, acc= local_model.update_weights(
                model= global_model, global_round=epoch)
            
            
            
            #Load the local learnt Parameters
            global_model.model.load_weights("weights.h5")
            
            #Accumulate the weights
            weights_accumulated = weights_accumulated + np.array(w, dtype='object')
            
            #Save Personalized Parameters
            global_model.model.save_weights("personalized_model_user_"+ str(idx)+".h5")
            
            global_model.model.save_weights(script_dir+"/Users_epochs_models/personalized_model_user_"+ str(idx)+"_epoch_"+str(epoch)+".h5")
            
            #Print the progress
            print("User # "+str(idx)+" ends local training in Global round # "+str(epoch)+" and the loss is "+ str(loss))
            
            
            for layer in global_model.model.layers:
                if len(layer.get_weights()) > 0 and layer.name in classifier_layers:
                    weights_dict_users[layer.name] = layer.get_weights()
                    
            with open(script_dir+'/classifier_models/classifier_weights_dict_user_'+str(idx)+'_epoch_'+str(epoch)+'.pkl', 'wb') as f:
                      pickle.dump(weights_dict_users, f)  
      
            results = local_model.inference(global_model)
            
            local_acc.append(results[0])  #results[0]
            local_losses.append(results[1])
            local_pre.append(results[2])
            local_recall.append(results[3])
            local_f1.append(results[4])
            
            print(str(results[0])+"--"+str(acc)+"  "+str(results[1]))

            client_training = [idx, epoch, results[1], results[0], results[2], results[3],results[4]]   
            with open('client_training.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(client_training)
                
            results = local_model.test_inference(args, global_model)
                
            local_losses_test.append(results[1])
            local_acc_test.append(results[0])
            local_pre_test.append(results[2])
            local_recall_test.append(results[3])
            local_f1_test.append(results[4])
       
            
            client_testing = [idx+1, epoch+1, results[1], results[0], results[2], results[3], results[4]]   
            with open('client_testing.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(client_testing)
           
        # update global weights
        weights_accumulated = weights_accumulated / args.num_users
        
        global_model.model.set_weights(weights_accumulated)
        
        
        for layer in global_model.model.layers:
            if len(layer.get_weights()) > 0 and layer.name in classifier_layers:
                weights_dict[layer.name] = layer.get_weights()
  
        
        with open('classifier_weights_full_model.pkl', 'wb') as f:
                pickle.dump(weights_accumulated, f)

        with open('classifier_weights_dict.pkl', 'wb') as f:
                pickle.dump(weights_dict, f)    
                
                
                
        with open(script_dir+'/classifier_models/classifier_weights_dict_'+str(epoch)+'.pkl', 'wb') as f:
                pickle.dump(weights_dict, f)         
                
                
        epoch_training = [epoch+1, np.mean(local_losses), np.mean(local_acc), np.mean(local_pre), np.mean(local_recall),np.mean(local_f1)]   
        with open('epoch_training.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(epoch_training)
                
                
        epoch_testing = [epoch+1, np.mean(local_losses_test), np.mean(local_acc_test), np.mean(local_pre_test), np.mean(local_recall_test),np.mean(local_f1_test)]   
        with open('epoch_testing.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(epoch_testing)   
                
        general_test_row_epoch = [epoch+1, np.mean(general_test_loss),np.mean(general_test_acc), np.mean(general_test_pre), np.mean(general_test_recall), np.mean(general_test_f1)]  
                
        with open('general_test_epochs.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(general_test_row_epoch)
                
    