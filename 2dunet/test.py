from importlib import import_module
import random, sys, yaml, os, json, time
import numpy as np
import torch
import torch.nn.functional as F
from dataset1.dataset_test import CTDataset
from ops.dataset_ops import Train_Collatefn
from ops.log_ops import setup_logger
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from jfd import get_DC
from ops.acc_ops import roc
import matplotlib.pyplot as plt
from ops.stat_ops import ScalarContainer, count_consective_num
from ops.acc_ops import topks_correct, topk_errors, topk_accuracies, CalcIoU
from sklearn.metrics import average_precision_score

random.seed(0); torch.manual_seed(0); np.random.seed(0)
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


CFG_FILE = "cfgs/test.yaml"
# 这是第一种用到的函数
def dice_coef(output, target):
    smooth = 1e-5
    num = output.shape[0]
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    input_1 = output[:,0,:,:]
    input_2 = output[:,1,:,:]

    target_1 = target[:,0,:,:]
    target_2 = target[:,1,:,:]

    intersection_1 = (input_1 * target_1)
    intersection_2 = (input_2 * target_2)

    dice_1 = (2. * intersection_1.sum() + smooth) / (input_1.sum() + target_1.sum() + smooth)
    dice_2 = (2. * intersection_2.sum() + smooth) / (input_2.sum() + target_2.sum() + smooth)

    return dice_1,dice_2



############### Set up Variables ###############
with open(CFG_FILE, "r") as f: cfg = yaml.safe_load(f)

DATA_ROOT = cfg["DATASETS"]["DATA_ROOT"]
MODEL_UID = cfg["MODEL"]["MODEL_UID"]
PRETRAINED_MODEL_PATH = cfg["MODEL"]["PRETRAINED_MODEL_PATH"]
NUM_CLASSES = cfg["MODEL"]["NUM_CLASSES"]
SAMPLE_NUMBER = int(cfg["DATALOADER"]["SAMPLE_NUMBER"])
NUM_WORKERS = int(cfg["DATALOADER"]["NUM_WORKERS"])
RESULE_HOME = cfg["TEST"]["RESULE_HOME"]
LOG_FILE = cfg["TEST"]["LOG_FILE"]

model = import_module(f"model.{MODEL_UID}")
UNet = getattr(model, "UNet")

############### Set up Dataloaders ###############
Validset = CTDataset(data_home=DATA_ROOT,
                               split='test',
                               sample_number=SAMPLE_NUMBER)

model = UNet(n_channels=1, n_classes=NUM_CLASSES)
model = torch.nn.DataParallel(model).cuda()

model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, \
             map_location=f'cuda:{0}'), strict=True)

############### Logging out some info ###############
logger = setup_logger(logfile=LOG_FILE)
logger.info("Config {}...".format(CFG_FILE))
logger.info("{}".format(json.dumps(cfg, indent=1)))
logger.warning(f"Loading init model path {PRETRAINED_MODEL_PATH}")

############### Testing ###############
ValidLoader = torch.utils.data.DataLoader(Validset,
                                    batch_size=1,
                                    num_workers=NUM_WORKERS,
                                    collate_fn=Train_Collatefn,
                                    shuffle=False,)

logger.info("Do evaluation...")

os.makedirs(RESULE_HOME, exist_ok=True)
os.makedirs("visual", exist_ok=True)
DC = 0.		# Dice Coefficient
length=0
all_labels = np.array([0])
all_preds = np.array([0])
训练过程
with torch.no_grad():
    for i, (all_F, all_M, all_info) in enumerate(ValidLoader):
        logger.info (all_info)
        all_E = []
        images = all_F.cuda()
        all_M = all_M.cuda()
        #(lh, uh), (lw, uw) = all_info[0]["pad"]
        num = len(images)

        for ii in range(num):
            image = images[ii:ii+1]
            # print("image",image)
            pred = model(image)
            all_labels,all_preds=roc(pred,image)
            SR = pred.sigmoid()
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            all_E.append(pred)
            GT = image.cuda()
            DC += get_DC(SR, GT)
            length += images.size(0)
        DC = DC / length
        all_E = torch.cat(all_E, dim=0).cpu().numpy().astype('uint8')
        all_OF = np.uint8(all_F[:, 0, :, :].cpu().numpy().astype('float32') * 255)
        unique_id = all_info[0]["name"].split('/')[-1].replace('.npy', '')
        np.save("{}/{}.npy".format(RESULE_HOME, unique_id), all_OF)
        np.save("{}/{}-dlmask.npy".format(RESULE_HOME, unique_id), all_E)

        if False:
            from zqlib import imgs2vid
            imgs2vid(np.concatenate([all_OF, all_E*255], axis=2), "visual/{}.avi".format(unique_id))
all_labels = all_labels[1:]
all_preds = all_preds[1:]
fpr, tpr, threshold = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)
optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=threshold)

lw = 2
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend()
plt.savefig('./save/roc.png')

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
plt.legend()
plt.savefig('./save/prc.png')


