import os
import sys
from data import *

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import warnings
import torch
from torch.autograd import Variable
import numpy as np
import cv2
# from SMENet import build_SMENet
# from matplotlib import pyplot as plt
# from data import VOCDetection, VOCAnnotationTransform
# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# import argparse
# import os
# warnings.filterwarnings("ignore", category=UserWarning)
# cfg = voc
# parser = argparse.ArgumentParser(description= 'SMENet Test')
# parser.add_argument('--trained_model', default='./weights/fuben_new_9.pth',
#                     type=str, help='Trained state_dict file path to open')
# args = parser.parse_args()
# net = build_SMENet('test', 400, cfg['num_classes'])
# net.load_state_dict(torch.load(args.trained_model), strict=False)
#
# testset = VOCDetection("/root/VOCdevkit", [('2012', 'val')], None, VOCAnnotationTransform())
def get_filelist(path):
    Filelist = []

    for home, dirs, files in os.walk(path):

        for filename in files:
            # 文件名列表，包含完整路径

            Filelist.append(os.path.join(home, filename))

            # # 文件名列表，只包含文件名

            # Filelist.append( filename)

    return Filelist

def cls(path):
    q = get_filelist(path)
    for i in q:
        os.remove(i)
# path1="/root/SMENet/SMENet/result"
# cls(path1)
import sys
import argparse

import torchvision.models as models
import torch

from ptflops import get_model_complexity_info
ost = sys.stdout
net=torch.load('./weights/model1_1.pt')
if torch.cuda.is_available():
    net=net.cuda()

flops, params = get_model_complexity_info(net, (3, 224, 224),
                        as_strings=True,
                        print_per_layer_stat=True,
                        ost=ost)
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
# for img_id in range(len(testset)):
#     image,index = testset.pull_image(img_id)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     x = cv2.resize(image, (400, 400)).astype(np.float32)
#     # x -= (86.0, 91.0, 82.0)
#     x = x.astype(np.float32)
#     x = x[:, :, ::-1].copy()
#     x = torch.from_numpy(x).permute(2, 0, 1)

#     xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
#     if torch.cuda.is_available():
#         xx = xx.cuda()
#     y= net(xx)
#     from data import VOC_CLASSES as labels
#     top_k=10
 
#     plt.figure(figsize=(10.4,10.4))
#     aaa=np.linspace(0, 1, 16)
#     # aaa[12]=0.2
#     # aaa[13]=0.6
#     # aaa[14]=0.7
#     # aaa[14]=1
#     colors = plt.cm.hsv(aaa).tolist()

#     plt.imshow(rgb_image)
#     currentAxis = plt.gca()
   
#     detections = y.data
#     scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
#     for i in range(detections.size(1)):
#         j = 0
#         while detections[0,i,j,0] >= 0.5:
#             score = detections[0,i,j,0]
#             label_name = labels[i-1]
#             display_txt = '%s: %.2f'%(label_name, score)
#             pt = (detections[0,i,j,1:]*scale).cpu().numpy()
#             coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
#             color = colors[i]
#             currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1.5))
#             # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.3,'pad':0},fontsize=7, color='white')
#             j+=1
#     plt.axis('off') 
#     if not os.path.exists("./result/"):
#         os.makedirs("./result/")
#     currentAxis.figure.savefig(f"./result/test_result{index}.jpg",dpi=100, bbox_inches='tight', pad_inches=0)
# plt.close()