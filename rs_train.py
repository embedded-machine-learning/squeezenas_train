# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import segmentation_models_pytorch.segmentation_models_pytorch as smp
from pre_process import main_dataset
from countmacs import MAC_Counter
from nets import SQUEEZENAS_NETWORKS

def main():
    #start_time = time.time()
    parser = argparse.ArgumentParser(description='Training SqueezeNAS models with RailSem dataset')

    parser.add_argument('-n', '--net', default='squeezenas_lat_small',
                        help='SqueezeNAS model for training', required=False)
    
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='Epoch count', required=False)
    
    parser.add_argument('-i', '--image_count',type=int, default=4000,
                        help='The count of images from the dataset for training and validation'
                        , required=False)
    
    parser.add_argument('-c', '--use_cpu', action='store_true',
                        help='If this option supplied, the network will be evaluated using the cpu.'
                             ' Otherwise the gpu will be used to run evaluation.',required=False)
    
    parser.add_argument('-d', '--data_dir', default='./railsem',
                        help='Location of the dataset.'
                        , required=False)
    
    parser.add_argument('-v', '--val_split',type=float, default=0.3,
                        help='Validation Split for training and validation set.'
                        , required=False)
    
    parser.add_argument('-s', '--save_dir', default='./railsem_trained_weights_new/',
                        help='New weights will be saved here after training.'
                        , required=False)
    
    
    args, unknown = parser.parse_known_args()
    print(args)

    net_name =  args.net
    net_constructor = SQUEEZENAS_NETWORKS[net_name]
    model = net_constructor()

    if (args.use_cpu):
        DEVICE = 'cpu'
    DEVICE = 'cuda'


    images_dir = os.path.join(args.data_dir,'jpgs/rs19_val')
    masks_dir = os.path.join(args.data_dir,'uint8/rs19_val')
    jsons_dir = os.path.join(args.data_dir,'jsons/rs19_val')
    org_dataset = main_dataset(
        images_dir, 
        masks_dir, 
        args.image_count,
    )


    validation_split = args.val_split
    train_dataset, val_dataset = random_split(org_dataset, [int(args.image_count*(1-args.val_split)),
                                                          int(args.image_count*args.val_split)] )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)#, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)#, num_workers=4)




    loss = smp.utils.losses.CrossEntropyLoss(ignore_index=255)
    metrics = [
        #smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Accuracy(threshold=0.5),
    ]


    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    val_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )


    max_score = 2

    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_name = args.save_dir+net_name+'.pth'
    save_name_best = args.save_dir+net_name+'_bestloss.pth'

    print('Training on {}'.format(args.net))
    for i in range(0, args.epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = val_epoch.run(val_loader)
        
        # saving model ...
        if max_score > valid_logs['cross_entropy_loss']:
            max_score = valid_logs['cross_entropy_loss']

            torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_logs['cross_entropy_loss'],
                        'acc' : valid_logs['accuracy']
                        }, save_name_best)
            print('Model saved!')
        torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_logs['cross_entropy_loss'],
                'acc' : valid_logs['accuracy']
                }, save_name)
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

  
if __name__ == "__main__":

    main()
