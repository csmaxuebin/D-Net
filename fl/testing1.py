#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from pandas import np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def test_img1(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    y_scores=[]
    y_labels=torch.empty([])
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1 :
            # data, target = data.cpu(), target.cpu()
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)

        # print(log_probs)
        # y_probas = gcf.predict_proba(X_te)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # print("***********************************************************************8")
        # print(y_pred)
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cuda().sum()

        # 处理分数
        y_temp=log_probs.detach().cpu().numpy()
        y_temp = y_temp[:, 1]
        y_scores.extend(y_temp)
        # 处理标签
        if idx==0:
            y_labels=target.clone().detach().cpu().numpy()
        else:
            y_labels = torch.cat([y_labels, target], 0).cpu().numpy()
            # print(y_labels)


    # 计算ROC
    # test_y_score = net_g.decision_function(data)
    # y_scores = log_probs.detach().numpy()
    # y_scores=y_scores[:,1]#关键一步得到scores
    fpr, tpr, threshold = roc_curve(y_labels, y_scores)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=threshold)
    roc_auc = auc(fpr, tpr)

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
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('./save/none_roc_test_testing1.png')

    # 计算PR
    # y_scores = log_probs.detach().numpy()
    # y_scores = y_scores[:, 1]  # 关键一步得到scores
    precision, recall, thresholds= precision_recall_curve(y_labels, y_scores)
    prauc=auc(recall, precision)

    lw = 2
    plt.figure(figsize=(8, 5))
    plt.plot(recall,precision, color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' % prauc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 01.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision/Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig('./save/none_pr_test_testing1.png')

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))


    return accuracy, test_loss

