from glob import glob

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from utilities.metrics import dice_coef, batch_iou, mean_iou, iou_score,accuracy,sensitivity,myroc
from tqdm import tqdm
import pandas as pd
import os
import datetime
import time
from dataset.dataset import Dataset
import matplotlib.pyplot as plt
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def test(args,  model, criterion):
    timestamp  = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('models/test/{}/{}'.format(args.name,timestamp)):
        os.makedirs('models/test/{}/{}'.format(args.name,timestamp))
    loss = 0
    iou = 0
    dice1 = 0
    dice2 = 0
    acc=0
    sen = 0

    accs = AverageMeter()
    sens = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    # Data loading code
    img_paths = glob('./data/testImage/*')
    mask_paths = glob('./data/testMask/*')
    # test_img_paths= glob('./data/testImage/*')
    # test_mask_paths = glob('./data/testMask/*')

    test_img_paths, val_img_paths, test_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.9, random_state=39)
    print("test_num:%s"%str(len(test_img_paths)))

    train_dataset = Dataset(args, test_img_paths, test_mask_paths, args.aug)

    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    log = pd.DataFrame(index=[], columns=[
        'lr', 'test_loss', 'test_iou', 'test_dice_1', 'test_dice_2','acc','sen','fpr','tpr','threshold','precision','recall','thresholds'
    ])

    model.eval()
    all_labels = np.array([0])
    all_preds = np.array([0])
    y_scores=[]
    y_labels=torch.empty([])
    first_time = time.time()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                # add_labels = target.cpu().detach().numpy()
                # y_pred = output.data.max(1,keepdim=True)[1]
                # add_preds = y_pred.squeeze()
                # add_preds = add_preds.cpu().detach().numpy()
                # all_labels = np.concatenate([all_labels, add_labels])
                # all_preds = np.concatenate([all_preds, add_preds])
                all_labels,all_preds=myroc(output,target)
                acc = accuracy(output,target)
                sen = sensitivity(output,target)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice_1 = dice_coef(output, target)[0]
                dice_2 = dice_coef(output, target)[1]
            accs.update(acc,input.size(0))
            sens.update(sen,input.size(0))
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
            dices_2s.update(torch.tensor(dice_2), input.size(0))

    # 计算ROC
    all_labels = all_labels[1:]
    all_preds = all_preds[1:]
    dfshuju = pd.DataFrame({"lables":all_labels,"preds":all_preds})
    dfshuju.to_csv("./models/test/{}/{}/shuju.csv".format(args.name,timestamp),index=0)
    end_time = time.time()
    print("time:", (end_time - first_time) / 60)

    # compute ROC
    fpr, tpr, threshold = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=threshold)

    lw = 2
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    plt.savefig('./models/test/{}/{}/roc_test_testing1.png'.format(args.name,timestamp))

    # 计算PR
    precision, recall, thresholds= precision_recall_curve(all_labels, all_preds)
    prauc=auc(recall, precision)

    lw = 2
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, color='darkorange', lw=lw,
             label='ROC curve (area = %0.2f)' % prauc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 01.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision/Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig('./models/test/{}/{}/prc_test_testing1.png'.format(args.name,timestamp))


    tmp = pd.Series([
        args.lr,
        losses.avg,
        ious.avg,
        dices_1s.avg,
        dices_2s.avg,
        accs.avg,
        sens.avg,
        all_labels,
        all_preds,


    ], index=['lr', 'test_loss', 'test_iou', 'test_dice_1', 'test_dice_2','acc','sen','all_labels','all_preds'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('./models/test/{}/{}/testlog.csv'.format(args.name,timestamp), index=False)
    torch.cuda.empty_cache()
