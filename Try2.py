from effnetv2 import effnetv2_s
from effnetv2 import effnetv2_m
import torch
import torchmetrics
from Data_in_seg import Dataset1, Dataset2, Dataset3
from Resnet101 import resnext50_32x4d
from VGG import VGG16
from torch import nn
# model1=torch.load('./weights/Model2.pt').cuda()
# model1=effnetv2_m(8).cuda()
model1=effnetv2_s(2).cuda()
Loss1 = nn.CrossEntropyLoss().cuda()
optm1 = torch.optim.AdamW(model1.parameters(), lr=0.002,weight_decay=0.002)
# scher = torch.optim.lr_scheduler.ReduceLROnPlateau(optm1, mode='min', patience=3, eps=1e-8)
scher=torch.optim.lr_scheduler.StepLR(optm1,step_size=5,gamma=0.4)
Loss_train=[]
Acc_val=[]
Acc_Test=[]
Dataset3 = Dataset3
Dataset1 = Dataset1
Dataset2 = Dataset2
def Train(Model, Loss, Dataset, Optm):
    for i in range(200):
        torch.cuda.empty_cache()
        # model1.train()
        for i, (img, target) in enumerate(Dataset1):
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            loss = Loss1(Model(img), target)
            if i % 40 == 0:
                Loss_1=loss.item()
                print("Loss: {}".format(Loss_1),"lr: {}".format(optm1.param_groups[0]['lr']))
                Loss_train.append(Loss_1)
                f = open("loss_trainx.txt", "w")
                f.write(str(Loss_train))
                f.close()
            optm1.zero_grad()
            loss.backward()
            optm1.step()
        scher.step()
        # model1.eval()
        # test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8).cuda()
        #
        # # test_recall=torchmetrics.Recall(task="multiclass",num_classes=8).cuda()
        # for i, (img, target) in enumerate(Dataset2):
        #     img = img.cuda(non_blocking=True)
        #     # print(img)
        #     target = target.cuda(non_blocking=True)
        #     pred = model1(img)
        #     test_acc(pred.argmax(1), target)
        #     # test_recall(pred.argmax(1),target)
        # val=test_acc.compute()
        # print("val",val)
        # Acc_val.append(val.item())
        # del val
        # f = open("Acc_valx.txt", "w")
        # f.write(str(Acc_val))
        # f.close()
        # # print("test",test_recall.compute())
        # train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8).cuda()
        # # train_recall=torchmetrics.Recall(task="multiclass", num_classes=8).cuda()
        # for i, (img, target) in enumerate(Dataset3):
        #     img = img.cuda(non_blocking=True)
        #     # print(img)
        #     target = target.cuda(non_blocking=True)
        #     pred = model1(img)
        #     train_acc(pred.argmax(1), target)
        #     # train_recall(pred.argmax(1),target)
        # test=train_acc.compute()
        # print("test", test)
        # Acc_Test.append(test.item())
        # del test
        # f = open("Acc_Testx.txt", "w")
        # f.write(str(Acc_Test))
        # f.close()
        # # print(("test_val2",train_recall.compute()))
        torch.save(model1, "./weights/Model_rx_1.pt")
Train(model1, Loss1, Dataset1, optm1)


