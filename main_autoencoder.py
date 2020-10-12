import os
import time

import shutil
import pickle

import argparse

import ipdb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from medpy.metric import binary

import dataloader as loader
from utils import *
import torch.optim as optim
from pred_autoencoder import main_test
from adamp import AdamP

parser = argparse.ArgumentParser()

# arguments for dataset

parser.add_argument('--server', default='server_B')
parser.add_argument('--exp',default='test4',type=str)
parser.add_argument('--train-mode',default=True,type=str2bool)

parser.add_argument('--source-dataset',default='site2',
                    help='site2,site3,site4')

parser.add_argument('--input-size',default=128,type=int)
parser.add_argument('--train-size',default=0.7,type=float)
parser.add_argument('--val-size',default=0,type=float)

parser.add_argument('--batch-size',default=32,type=int)
parser.add_argument('--aug-mode',default=True,type=str2bool)
parser.add_argument('--aug-range',default='aug2', type=str)


# arguments for model
parser.add_argument('--arch', default='unet', type=str)
parser.add_argument('--start-channel', default=64, type=int)
parser.add_argument('--denoising',default=False,type=str2bool)
parser.add_argument('--salt-prob', default=0.1, type=float)

# arguments for optim & loss
parser.add_argument('--optim',default='adam',
                    choices=['adam','adamp','sgd'],type=str)

parser.add_argument('--eps',default=1e-08,type=float)
parser.add_argument('--weight-decay',default=1e-4,type=float)

parser.add_argument('--loss-function',default='bce',type=str)

parser.add_argument('--scheduler',default=True,type=str)
parser.add_argument('--lr',default=0.0005,type=float,help='initial-lr')
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)


# arguments for test mode
parser.add_argument('--file-name', default='result_all', type=str)
parser.add_argument('--test-mode',default=True,type=str2bool)




args = parser.parse_args()


def main():
    # save input stats for later use
    if args.server == 'server_A':
        work_dir = os.path.join('/data1/JM/spinal_cord_segmentation', args.exp)
        print(work_dir)
    elif args.server == 'server_B':
        work_dir = os.path.join('/data1/workspace/JM_gen/'
                                'spinal_cord_segmentation', args.exp)
        print(work_dir)
    elif args.server == 'server_D':
        work_dir = os.path.join('/daintlab/home/woans0104/workspace/'
                                'spinal-cord-segmentation',args.exp)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    #import ipdb;ipdb.set_trace()


    source_dataset, target_dataset1, target_dataset2, target_dataset3 \
        = loader.dataset_condition(args.source_dataset)


    # 1.load_dataset
    train_loader_source,test_loader_source \
        = loader.get_loader(server=args.server,
                            dataset=source_dataset,
                            train_size=args.train_size,
                            input_size=args.input_size,
                            aug_mode=args.aug_mode,
                            aug_range=args.aug_range,
                            batch_size=args.batch_size,
                            work_dir=work_dir)


    train_loader_target1, _ = loader.get_loader(server=args.server,
                                                dataset=target_dataset1,
                                                train_size=1,
                                                input_size=args.input_size,
                                                aug_mode=False,
                                                aug_range=args.aug_range,
                                                batch_size=1,
                                                work_dir=work_dir)


    train_loader_target2, _ = loader.get_loader(server=args.server,
                                                dataset=target_dataset2,
                                                train_size=1,
                                                input_size=args.input_size,
                                                aug_mode=False,
                                                aug_range=args.aug_range,
                                                batch_size=1,
                                                work_dir=work_dir)

    train_loader_target3, _ = loader.get_loader(server=args.server,
                                                dataset=target_dataset3,
                                                train_size=1,
                                                input_size=args.input_size,
                                                aug_mode=False,
                                                aug_range=args.aug_range,
                                                batch_size=1,
                                                work_dir=work_dir)

    test_data_li = [test_loader_source,
                    train_loader_target1,
                    train_loader_target2,
                    train_loader_target3]




    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))


    # 3.model_select
    model_seg = ae_spinal(
        in_shape=(1, args.input_size, args.input_size))

    # 4.gpu select
    model_seg = nn.DataParallel(model_seg).cuda()
    cudnn.benchmark = True

    # 5.optim

    if args.optim == 'adam':
        optimizer_seg = torch.optim.Adam(model_seg.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         eps=args.eps)

    elif args.optim == 'adamp':
        optimizer_seg = AdamP(model_seg.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay,
                              eps=args.eps)
    elif args.optim == 'sgd':
        optimizer_seg = torch.optim.SGD(model_seg.parameters(), lr=args.lr,
                                        momentum=0.9,
                                        weight_decay=args.weight_decay)




    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_seg,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=0.1)

    # 6.loss
    if args.loss_function == 'bce':
        criterion = nn.BCELoss()
    elif args.loss_function == 'bce_logit':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_function == 'dice':
        criterion = DiceLoss()
    elif args.loss_function == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_function == 'Cldice':
        bce = nn.BCEWithLogitsLoss().cuda()
        dice = DiceLoss().cuda()
        criterion = ClDice(bce,dice,alpha=1,beta=1)

    criterion = criterion.cuda()


###############################################################################

    # train


    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):

                train(model=model_seg, train_loader=train_loader_source,
                      epoch=epoch, criterion=criterion,
                      optimizer=optimizer_seg,
                      logger=trn_logger, sublogger=trn_raw_logger)

                iou = validate(model=model_seg, val_loader=test_loader_source,
                               epoch=epoch, criterion=criterion,
                               logger=val_logger,work_dir=work_dir)

                print('validation_result ************************************')

                lr_scheduler.step()

                if args.val_size ==0:
                    is_best = 1
                else:
                    is_best = iou > best_iou
                best_iou = max(iou, best_iou)
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model_seg.state_dict(),
                                 'optimizer': optimizer_seg.state_dict()},
                                is_best,
                                work_dir,
                                filename='checkpoint.pth')


        print("train end")
    except RuntimeError as e:
        print('error message : {}'.format(e))


        import ipdb
        ipdb.set_trace()

    draw_curve(work_dir, trn_logger, val_logger)
    # here is load model for last pth
    check_best_pth(work_dir)

    # validation
    if args.test_mode:
        print('Test mode ...')
        main_test(model=model_seg,test_loader=test_data_li, args=args)





def train(model, train_loader, epoch, criterion, optimizer, logger, sublogger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()


    model.train()
    end = time.time()

    for i, (input, target,_,_) in enumerate(train_loader):


        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()


        if args.denoising == True:
            noisy_batch_input = make_noise_input(target)

            output, _ = model(noisy_batch_input)########

        else:

            output, _ = model(target)########

        loss = criterion(output, target)


        iou, dice = performance(output, target, dist_con=False)
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'IoU {iou.val:.4f} ({iou.avg:.4f})\t'
              'Dice {dice.val:.4f} ({dice.avg:.4f})\t'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses,
            iou=ious, dice=dices))

        if i % 10 == 0:
            sublogger.write([epoch, i, loss.item(), iou, dice])

    logger.write([epoch, losses.avg, ious.avg, dices.avg])





def validate(model, val_loader, epoch, criterion, logger, work_dir):

    batch_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, ori_img,_ ) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output, _ = model(target)########

            loss = criterion(output, target)

            iou, dice = performance(output, target, dist_con=False)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))

            ###################################
            save = 1
            if save:
                if i % 10 == 0:
                    save_fig(str(epoch), ori_img, target, output, iou,
                             work_dir,'epoch_predict', str(i))

            #############################

            batch_time.update(time.time() - end)
            end = time.time()

    print(' * IoU {ious.avg:.3f}({ious.std:.3f}) '
          'Dice {dices.avg:.3f}({dices.std:.3f})'.format(
        ious=ious, dices=dices))

    logger.write([epoch, losses.avg, ious.avg, dices.avg])

    return ious.avg



if __name__ == '__main__':
    main()