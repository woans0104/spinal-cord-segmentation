

import os
import time
import shutil
import pickle
from copy import deepcopy
import argparse
import ipdb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms

from medpy.metric import binary

import dataloader as loader  ############################
from utils import *
from predict import main_test
from model import *
from losses import DiceLoss,ClDice
from adamp import AdamP



parser = argparse.ArgumentParser()
# arguments for dataset
parser.add_argument('--server', default='server_B')
parser.add_argument('--exp',default="test4", type=str)
parser.add_argument('--train-mode',default=True,type=str2bool)

parser.add_argument('--source-dataset',default='site2',
                    help='site2,site3,site4')

parser.add_argument('--input-size',default=128,type=int)
parser.add_argument('--train-size',default=0.7,type=float)
parser.add_argument('--val-size',default=0,type=float)

parser.add_argument('--batch-size',default=32,type=int)
parser.add_argument('--aug-mode',default=False,type=str2bool)
parser.add_argument('--aug-range',default='aug6', type=str)

# arguments for model
parser.add_argument('--arch-seg', default='unet', type=str)
parser.add_argument('--arch-ae', default='ae_v2', type=str)
parser.add_argument('--arch-ae-detach', default=True, type=str2bool)

parser.add_argument('--embedding-alpha', default=1, type=float)
parser.add_argument('--denoising',default=True,type=str2bool)
parser.add_argument('--salt-prob', default=0.1, type=float)

# arguments for optim & loss
parser.add_argument('--optim',default='sgd',
                    choices=['adam','adamp','sgd'],type=str)
parser.add_argument('--weight-decay',default=5e-4,type=float)
parser.add_argument('--eps',default=1e-8,type=float, help='adam eps')


parser.add_argument('--seg-loss-function',default='bce_logit',type=str)
parser.add_argument('--ae-loss-function',default='bce_logit',type=str)
parser.add_argument('--embedding-loss-function',default='mse',type=str)

parser.add_argument('--lr', default=0.1, type=float, help='initial-lr')
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)

# arguments for test mode
parser.add_argument('--file-name', default='result_all', type=str)
parser.add_argument('--test-mode',default=True,type=str2bool)

args = parser.parse_args()



def main():

    ###########################################################################
    if args.server == 'server_A':
        work_dir = os.path.join('/data1/JM/spinal_cord_segmentation', args.exp)
        print(work_dir)
    elif args.server == 'server_B':
        work_dir = os.path.join('/data1/workspace/JM_gen/'
                                'spinal_cord_segmentation', args.exp)
        print(work_dir)

    elif args.server == 'server_D':
        work_dir = os.path.join('/daintlab/home/woans0104/workspace/'
                                'spinal-cord-segmentation', args.exp)
        print(work_dir)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    ###########################################################################


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

    trn_logger_ae = Logger(os.path.join(work_dir, 'ae_train.log'))
    val_logger_ae = Logger(os.path.join(work_dir, 'ae_validation.log'))


    # 2.model_select
    model_seg  = Unet2D_spinal(in_shape=(1, args.input_size, args.input_size))
    model_seg=model_seg.cuda()
    model_ae = ae_spinal(in_shape=(1, args.input_size, args.input_size))
    model_ae = model_ae.cuda()

    cudnn.benchmark = True

    # 3.gpu select
    model_seg = nn.DataParallel(model_seg)
    model_ae = nn.DataParallel(model_ae)



    # 4.optim
    if args.optim == 'adam':
        optimizer_seg = torch.optim.Adam(model_seg.parameters(),
                                         eps=args.eps,
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)

        optimizer_ae = torch.optim.Adam(model_ae.parameters(),
                                        eps=args.eps,
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)
    elif args.optim == 'adamp':
        optimizer_seg = AdamP(model_seg.parameters(),
                              eps=args.eps,
                              lr=args.lr,
                              weight_decay=args.weight_decay)

        optimizer_ae = AdamP(model_ae.parameters(),
                             eps=args.eps,
                             lr=args.lr,
                             weight_decay=args.weight_decay)

    elif args.optim == 'sgd':
        optimizer_seg = torch.optim.SGD(model_seg.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)

        optimizer_ae = torch.optim.SGD(model_ae.parameters(),
                                       lr=args.lr,
                                       weight_decay=args.weight_decay)



    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler_seg = optim.lr_scheduler.MultiStepLR(optimizer_seg,
                                                      milestones=lr_schedule[:-1],
                                                      gamma=0.1)

    lr_scheduler_ae = optim.lr_scheduler.MultiStepLR(optimizer_ae,
                                                     milestones=lr_schedule[:-1],
                                                     gamma=0.1)


    # 5.loss

    criterion_seg = select_loss(args.seg_loss_function)
    criterion_ae = select_loss(args.ae_loss_function)
    criterion_embedding = select_loss(args.embedding_loss_function)

    ############################################################################
    # train

    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):

                train(model_seg=model_seg,
                      model_ae=model_ae,
                      train_loader=train_loader_source,
                      epoch=epoch,
                      criterion_seg=criterion_seg,
                      criterion_ae=criterion_ae,
                      criterion_embedding=criterion_embedding,
                      optimizer_seg=optimizer_seg,
                      optimizer_ae=optimizer_ae,
                      logger=trn_logger,
                      sublogger=trn_raw_logger,
                      logger_ae=trn_logger_ae)

                iou = validate(model_seg=model_seg,
                               model_ae=model_ae,
                               val_loader=test_loader_source,
                               epoch=epoch,
                               criterion=criterion_seg,
                               logger=val_logger,
                               logger_ae=val_logger_ae)

                print('validation result ************************************')

                lr_scheduler_seg.step()
                lr_scheduler_ae.step()

                if args.val_size == 0:
                    is_best = 1
                else:
                    is_best = iou > best_iou
                best_iou = max(iou, best_iou)
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model_seg.state_dict(),
                                 'optimizer': criterion_seg.state_dict()},
                                is_best, work_dir, filename='checkpoint.pth')

        print("train end")
    except RuntimeError as e:
        print('#jm_private',
              '-----------------------------------  error train : '
              'send to message JM  '
              '& Please send a kakao talk -------------------------- '
              '\n error message : {}'.format(e))

        import ipdb
        ipdb.set_trace()

    draw_curve(work_dir, trn_logger, val_logger)
    draw_curve(work_dir, trn_logger_ae, val_logger_ae, labelname='ae')

    # here is load model for last pth
    check_best_pth(work_dir)

    # validation
    if args.test_mode:
        print('Test mode ...')
        main_test(model=model_seg, test_loader=test_data_li, args=args)


def train(model_seg, model_ae, train_loader, epoch,
          criterion_seg, criterion_ae,
          criterion_embedding, optimizer_seg, optimizer_ae,
          logger, sublogger, logger_ae):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    embedding_losses = AverageMeter()
    recon_losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    losses_ae = AverageMeter()
    ious_ae = AverageMeter()
    dices_ae = AverageMeter()

    model_seg.train()
    model_ae.train()
    end = time.time()

    for i, (input, target, _, _) in enumerate(train_loader):

        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()

        # segmentation
        output_seg, bottom_seg = model_seg(input)

        # autoencoder
        if args.denoising == True:
            noisy_batch_input = make_noise_input(target, args.salt_prob)
            output_ae, bottom_ae = model_ae(noisy_batch_input)

        else:
            output_ae, bottom_ae = model_ae(target)

        # loss-function

        loss_seg = criterion_seg(output_seg, target)
        loss_ae = criterion_ae(output_ae, target)

        # embedding loss
        loss_embedding = embedding_loss(args.embedding_loss_function,
                                        criterion_embedding,
                                        bottom_seg,
                                        bottom_ae, args.arch_ae_detach)

        loss_embedding = float(args.embedding_alpha) * loss_embedding

        if args.arch_seg == 'unet_recon':
            recon_loss = nn.L1Loss()
            loss_recon = recon_loss(output_seg, output_ae.detach())

            recon_losses.update(loss_recon.item(), input.size(0))
            total_loss = (loss_seg) + (loss_embedding) + (loss_recon)

        else:
            total_loss = (loss_seg) + (loss_embedding)

        print('loss_seg : ', loss_seg)
        print('loss_ae : ', loss_ae)
        print('alpha * loss_embedding : ', loss_embedding)
        if args.arch_seg == 'unet_recon':
            print('loss_recon : ', loss_recon)

        print('Total_loss : ', total_loss)

        # seg-log
        iou, dice = performance(output_seg, target, dist_con=False)
        losses.update(total_loss.item(), input.size(0))
        embedding_losses.update(loss_embedding.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))

        # ae-log

        iou_ae, dice_ae = performance(output_ae, target, dist_con=False)
        losses_ae.update(loss_ae.item(), input.size(0))
        ious_ae.update(iou_ae, input.size(0))
        dices_ae.update(dice_ae, input.size(0))

        optimizer_seg.zero_grad()
        optimizer_ae.zero_grad()

        # first ae backward

        if args.arch_ae_detach:
            loss_ae.backward()
        else:
            loss_ae.backward(retain_graph=True)

        optimizer_ae.step()

        # second se backward

        total_loss.backward()
        optimizer_seg.step()

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
            try:
                sublogger.write([epoch, i, total_loss.item(), iou, dice])
            except:
                # ipdb.set_trace()
                print('acd,asd : None')

    if args.arch_seg == 'unet_recon':
        logger.write(
            [epoch, losses.avg, embedding_losses.avg, recon_losses.avg,
             ious.avg, dices.avg])
        logger_ae.write([epoch, losses_ae.avg, ious_ae.avg, dices_ae.avg])
    else:
        logger.write([epoch, losses.avg, embedding_losses.avg,
                      ious.avg, dices.avg])
        logger_ae.write([epoch, losses_ae.avg, ious_ae.avg, dices_ae.avg])


def validate(model_seg, model_ae, val_loader, epoch, criterion,
             logger, logger_ae):
    batch_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    losses_ae = AverageMeter()
    ious_ae = AverageMeter()
    dices_ae = AverageMeter()

    model_seg.eval()
    model_ae.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, ori_img, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output, _ = model_seg(input)
            loss = criterion(output, target)

            # seg-log
            iou, dice = performance(output, target, dist_con=False)
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))

            # ae-log

            output_ae, _ = model_ae(target)
            loss_ae = criterion(output_ae, target)

            iou_ae, dice_ae = performance(output_ae, target, dist_con=False)
            losses_ae.update(loss_ae.item(), input.size(0))
            ious_ae.update(iou_ae, input.size(0))
            dices_ae.update(dice_ae, input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

    print(' * IoU {ious.avg:.3f}({ious.std:.3f}) '
          'Dice {dices.avg:.3f}({dices.std:.3f})'
          .format(ious=ious, dices=dices))

    logger.write([epoch, losses.avg, ious.avg, dices.avg])
    logger_ae.write([epoch, losses_ae.avg, ious_ae.avg, dices_ae.avg])

    return ious.avg




def select_loss(loss_function):
    if loss_function == 'bce':
        criterion = nn.BCELoss()
    elif loss_function == 'bce_logit':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function == 'dice':
        criterion = DiceLoss()
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
    elif loss_function == 'l1':
        criterion = nn.L1Loss()
    elif loss_function == 'kl' or loss_function == 'jsd':
        criterion = nn.KLDivLoss()
    elif loss_function == 'Cldice':
        bce = nn.BCEWithLogitsLoss().cuda()
        dice = DiceLoss().cuda()
        criterion = ClDice(bce,dice,alpha=1,beta=1)
    else:
        raise ValueError('Not supported loss.')
    return criterion.cuda()


def embedding_loss(embedding_loss, criterion_embedding,
                   bottom_seg, bottom_ae, detach):
    if detach == True:
        bottom_ae = bottom_ae.detach()

    if embedding_loss == 'kl':
        loss_embedding = criterion_embedding(F.log_softmax(bottom_seg),
                                             F.softmax(bottom_ae))
    elif embedding_loss == 'jsd':
        loss_embedding = (0.5 * criterion_embedding(F.log_softmax(bottom_seg),
                                                    F.softmax(bottom_ae))) \
                         + (0.5 * criterion_embedding(
            F.log_softmax(bottom_seg),
            F.softmax(bottom_ae)))

    elif embedding_loss == 'bce':
        bottom_seg = F.sigmoid(bottom_seg).cuda()
        bottom_ae = F.sigmoid(bottom_ae).cuda()
        loss_embedding = criterion_embedding(bottom_seg, bottom_ae)  # bce
    else:
        # MSE
        loss_embedding = criterion_embedding(bottom_seg, bottom_ae)

    return loss_embedding


if __name__ == '__main__':
    main()