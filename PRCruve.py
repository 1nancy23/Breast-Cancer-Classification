import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.datasets import ImageFolder
# from utils.transform import get_transform_for_test
from Data_in import Dataset2,Dataset3,Dataset1
# from senet.se_resnet import FineTuneSEResnet50
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

data_root = r'G:\DATAS\breast_2\breast_2'  # 测试集路径
# test_weights_path = r"C:\Users\admin\Desktop\fsdownload\epoch_0278_top1_70.565_'checkpoint.pth.tar'"  # 预训练模型参数
num_class = 8  # 类别数量
gpu = "cuda:0"


# mean=[0.948078, 0.93855226, 0.9332005], var=[0.14589554, 0.17054074, 0.18254866]
def test1(model, test_path):
    # 加载测试集和预训练模型参数
    test_dir = os.path.join(data_root, 'val')
    class_list = list(os.listdir(test_dir))
    class_list.sort()
    # transform_test = get_transform_for_test(mean=[0.948078, 0.93855226, 0.9332005],
    #                                         var=[0.14589554, 0.17054074, 0.18254866])
    # test_dataset = ImageFolder(test_dir, transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)
    test_loader=Dataset1
    # checkpoint = torch.load(test_path)
    # model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        score_tmp = outputs  # (batchsize, nclass)

        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
    print("score_array:", score_array.shape)  # (batchsize, classnum) softmax
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum]) onehot

    # 调用sklearn库，计算每个类别对应的precision和recall
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    for i in range(num_class):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
        average_precision_dict[i] = average_precision_score(label_onehot[:, i], score_array[:, i])
        print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])

    # micro
    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_onehot.ravel(),
                                                                              score_array.ravel())
    average_precision_dict["micro"] = average_precision_score(label_onehot, score_array, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision_dict["micro"]))

    # 绘制所有类别平均的pr曲线
    plt.figure(dpi=300)
    plt.step(recall_dict['micro'], precision_dict['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision_dict["micro"]))
    plt.savefig("train_pr_curve.jpg")
    # plt.show()


def get2(model, test_path):
    from itertools import cycle
    from scipy import interp
    # 加载测试集和预训练模型参数
    test_dir = os.path.join(data_root, 'test')
    class_list = list(os.listdir(test_dir))
    class_list.sort()
    # transform_test = get_transform_for_test(mean=[0.948078, 0.93855226, 0.9332005],
    #                                         var=[0.14589554, 0.17054074, 0.18254866])
    # test_dataset = ImageFolder(test_dir, transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)
    # checkpoint = torch.load(test_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    test_loader=Dataset1
    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        score_tmp = outputs  # (batchsize, nclass)

        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # 绘制所有类别平均的roc曲线
    plt.figure(dpi=300)
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('train_roc.jpg')
    plt.show()

if __name__ == '__main__':
    model=torch.load('./weights/Model1_1.pt')
    test1(model,1)