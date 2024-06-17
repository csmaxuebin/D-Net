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


def shuzhi(net_g, datatest, args):
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
            data, target = data.cpu(), target.cpu()
        log_probs = net_g(data)
        # print(log_probs)
        # y_probas = gcf.predict_proba(X_te)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]

        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        # 处理分数
        y_temp=log_probs.detach().numpy()
        y_temp = y_temp[:, 1]
        y_scores.extend(y_temp)
        # 处理标签
        if idx==0:
            y_labels=target.clone().detach()
        else:
            y_labels = torch.cat([y_labels, target], 0)
            # print(y_labels)

    # 计算ROC
    # test_y_score = net_g.decision_function(data)
    # y_scores = log_probs.detach().numpy()
    # y_scores=y_scores[:,1]#关键一步得到scores
    fpr, tpr, threshold = roc_curve(y_labels, y_scores)
    roc_auc = auc(fpr, tpr)



    # # 计算PR
    # y_scores = log_probs.detach().numpy()
    # y_scores = y_scores[:, 1]  # 关键一步得到scores
    precision, recall, thresholds= precision_recall_curve(y_labels, y_scores)
    prauc=auc(recall, precision)




    return roc_auc,prauc

