
import argparse
import random
import time

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataloading.dataset_specifics import *
from dataloading.datasets import TestDataset
from dataloading.datasets import TrainDataset as TrainDataset
from models.fewshot_anom import FewShotSeg
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_arguments():
    parser = argparse.ArgumentParser()

    # CHAOS
    # parser.add_argument('--data_root', default="/home/cmz/experiments/ADNet-main/data/CHAOST2", type=str)
    # parser.add_argument('--dataset', default="CHAOST2", type=str)
    # parser.add_argument('--n_sv', default=5000, type=int)

    # CMR
    parser.add_argument('--n_sv', default=1000, type=int)    # chaos => 5000    CMR => 1000
    parser.add_argument('--dataset', default="CMR", type=str)
    parser.add_argument('--data_root', default="/home/cmz/experiments/ADNet-main/data/CMR", type=str)

    # Training specs.
    parser.add_argument('--workers', default=8, type=int)    # change number of workers !
    parser.add_argument('--steps', default=100000, type=int)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--n_way', default=1, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--max_iterations', default=1000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)          # origin: 1e-3
    parser.add_argument('--lr_gamma', default=0.95, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=0.0005, type=float)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--bg_wt', default=0.1, type=float)
    parser.add_argument('--t_loss_scaler', default=1.0, type=float)

    # test specs
    parser.add_argument('--test_nshot', default=3, type=int)        # EP1 => 3  EP2 => 1
    parser.add_argument('--EP1', default=True, type=bool)          # 确认是否使用 EP1
    parser.add_argument('--all_slices', default=False, type=bool)

    # TODO
    parser.add_argument('--fold', default=1, type=int)                                                                  # change fold !
    parser.add_argument('--save_root', default="/home/cmz/experiments/ADNet-main/results_abd/train/fold1_supp1", type=str)    # change fold !
    parser.add_argument('--supp_idx', default=1, type=int)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Deterministic setting for reproducability.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Set up logging.
    logger = set_logger(args.save_root, 'train.log')
    logger.info(args)

    # Setup the path to save.
    args.save_model_path = os.path.join(args.save_root, 'model_final.pth')

    # Init model.
    model = FewShotSeg(False)  # 这里 pretrain=False 意思是只是不加载deeplab的权重，加载了resnet101的默认权重
    model = model.cuda()

    # Init optimizer.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,     # 0.001
                                momentum=args.momentum,          # 0.9
                                weight_decay=args.weight_decay)  # 0.0005
    milestones = [(ii + 1) * 2000 for ii in range(args.steps // 2000 - 1)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)

    # Define loss function.
    my_weight = torch.FloatTensor([args.bg_wt, 1.0]).cuda()    # loss function 系数
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True

    # Define data set and loader.
    train_dataset = TrainDataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    logger.info('  Training on images not in test fold: ' +
                str([elem[len(args.data_root):] for elem in train_dataset.image_dirs]))   # 打印一下 哪些用于训练了

    # Define data for test
    test_dataset = TestDataset(args)
    test_loader = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)

    # Start training.
    sub_epochs = args.steps // args.max_iterations   # 50000 // 1000
    logger.info('  Start training ...')

    best_mDice = 0.0
    for epoch in range(sub_epochs):  # 50

        # Train.
        batch_time, data_time, losses, q_loss, align_loss, t_loss = train(train_loader, model, criterion, optimizer,
                                                                          scheduler, args)

        # Log
        logger.info('============== Epoch [{}] =============='.format(epoch))
        logger.info('  Batch time: {:6.3f}'.format(batch_time))
        logger.info('  Loading time: {:6.3f}'.format(data_time))
        logger.info('  Total Loss  : {:.5f}'.format(losses))
        logger.info('  Query Loss  : {:.5f}'.format(q_loss))
        logger.info('  Align Loss  : {:.5f}'.format(align_loss))
        logger.info('  Threshold Loss  : {:.5f}'.format(t_loss))

        # Test.
        class_dice = test(logger, test_dataset, test_loader, model, args)

        # Save
        dice_list = class_dice.values()
        mean_dice = sum(dice_list) / len(dice_list)

        if best_mDice < mean_dice:
            best_mDice = mean_dice
            path = args.save_root + '/' + 'model_{:.3f}.pth'.format(best_mDice)
            torch.save(model.state_dict(), path)

        logger.info('current best mean dice: ' + str(best_mDice))    
            
    # Save trained model.
    torch.save(model.state_dict(), args.save_model_path)


def train(train_loader, model, criterion, optimizer, scheduler, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    q_loss = AverageMeter('Query loss', ':.4f')
    a_loss = AverageMeter('Align loss', ':.4f')
    t_loss = AverageMeter('Threshold loss', ':.4f')

    # Train mode.
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):

        # Extract episode data.
        support_images = [[shot.float().cuda() for shot in way]
                          for way in sample['support_images']]
        support_fg_mask = [[shot.float().cuda() for shot in way]
                           for way in sample['support_fg_labels']]

        if support_fg_mask[0][0].sum() <= 0.0:
            continue

        query_images = [query_image.float().cuda() for query_image in sample['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

        # Log loading time.
        data_time.update(time.time() - end)

        # Compute outputs and losses.
        query_pred, align_loss, thresh_loss = model(support_images, support_fg_mask, query_images,
                                                    train=True, t_loss_scaler=args.t_loss_scaler)  # => [1,2,256,256]/[1,2,256,256]/[1]/[1]

        query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                     1 - torch.finfo(torch.float32).eps)), query_labels)
        
        loss = query_loss + align_loss + thresh_loss

        # compute gradient and do SGD step
        for param in model.parameters():
            param.grad = None

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss.
        losses.update(loss.item(), query_pred.size(0))
        q_loss.update(query_loss.item(), query_pred.size(0))
        a_loss.update(align_loss.item(), query_pred.size(0))
        t_loss.update(thresh_loss.item(), query_pred.size(0))

        # Log elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

    return batch_time.avg, data_time.avg, losses.avg, q_loss.avg, a_loss.avg, t_loss.avg

def test(logger, test_dataset, test_loader, model, args):
    # Inference.
    logger.info('  Start inference ... Note: EP1 is ' + str(args.EP1))
    logger.info('  Support: ' + str(test_dataset.support_dir[len(args.data_root):]))
    logger.info('  Query: ' +
                str([elem[len(args.data_root):] for elem in test_dataset.image_dirs]))

    # Get unique labels (classes).
    labels = get_label_names(args.dataset)

    # Loop over classes.
    class_dice = {}
    class_iou = {}
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name is 'BG':
            continue

        logger.info('  *------------------Class: {}--------------------*'.format(label_name))
        logger.info('  *--------------------------------------------------*')

        # Get support sample + mask for current class.
        support_sample = test_dataset.getSupport(label=label_val, all_slices=args.all_slices, N=args.test_nshot)
        test_dataset.label = label_val

        # Infer.
        with torch.no_grad():
            scores = infer(model, test_loader, support_sample, args, logger)

        # Log class-wise results
        class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
        class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()

        logger.info('      Mean class IoU: {}'.format(class_iou[label_name]))
        logger.info('      Mean class Dice: {}'.format(class_dice[label_name]))
        logger.info('  *--------------------------------------------------*')

    # Log final results.
    logger.info('  *-----------------Final results--------------------*')
    logger.info('  *--------------------------------------------------*')
    logger.info('  Mean IoU: {}'.format(class_iou))
    logger.info('  Mean Dice: {}'.format(class_dice))
    logger.info('  *--------------------------------------------------*')

    return class_dice

def infer(model, query_loader, support_sample, args, logger):

    # Test mode.
    model.eval()

    # Unpack support data.
    support_image = [support_sample['image'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]  # n_shot x 3 x H x W
    support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]  # n_shot x H x W

    # Loop through query volumes.
    scores = Scores()
    for i, sample in enumerate(query_loader):

        # Unpack query data.
        query_image = [sample['image'][i].float().cuda() for i in range(sample['image'].shape[0])]  # [C x 3 x H x W]
        query_label = sample['label'].long()  # C x H x W
        query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]

        # Compute output.
        if args.EP1 is True:
            # Match support slice and query sub-chunck.
            query_pred = torch.zeros(query_label.shape[-3:])
            C_q = sample['image'].shape[1]
            idx_ = np.linspace(0, C_q, args.test_nshot+1).astype('int')
            for sub_chunck in range(args.test_nshot):
                support_image_s = [support_image[sub_chunck]]  # 1 x 3 x H x W
                support_fg_mask_s = [support_fg_mask[sub_chunck]]  # 1 x H x W
                query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck+1]]  # C' x 3 x H x W
                query_pred_s, _, _ = model([support_image_s], [support_fg_mask_s], [query_image_s], train=False)  # C x 2 x H x W
                query_pred_s = query_pred_s.argmax(dim=1).cpu()  # C x H x W
                query_pred[idx_[sub_chunck]:idx_[sub_chunck+1]] = query_pred_s

        else:  # EP 2
            query_pred, _, _ = model([support_image], [support_fg_mask], query_image, train=False)  # C x 2 x H x W
            query_pred = query_pred.argmax(dim=1).cpu()  # C x H x W

        # Record scores.
        scores.record(query_pred, query_label)

        # Log.
        logger.info('    Tested query volume: ' + sample['id'][0][len(args.data_root):]
                    + '. Dice score:  ' + str(scores.patient_dice[-1].item()))

    return scores

if __name__ == '__main__':
    main()

