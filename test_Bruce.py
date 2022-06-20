'''
author:zhujunwen
Guangdong University of Technology
'''
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from UNet import Unet,resnet34_unet
from attention_unet import AttU_Net
from channel_unet import myChannelUnet
from r2unet import R2U_Net
from segnet import SegNet
from unetpp import NestedUNet
from fcn import get_fcn8s
from dataset import *
from metrics import *
from torchvision.transforms import transforms
from plot import loss_plot
from plot import metrics_plot
from torchvision.models import vgg16
def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train")
    parse.add_argument("--epoch", type=int, default=21)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                       help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument('--dataset', default='bruce',  # dsb2018_256
                       help='dataset name:liver/esophagus/dsb2018Cell/corneal/driveEye/isbiCell/kaggleLung')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    args = parse.parse_args()
    return args


if __name__ =="__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5], [0.5])  # ->[-1,1]
    ])


    model_path = "./saved_model/UNet_4_bruce_50.pth"
    y_transforms = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Unet(1, 1).to(device)
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint)
    model = net.eval()


    x_path = r"C:\Users\shzhoujun\Desktop\eye_unet\data_train\test\p14_day01_196_right.png"
    x_path_label = r"C:\Users\shzhoujun\Desktop\eye_unet\data_train\test\p14_day01_196_rightlabel.png"
    origin_x = cv2.imread(x_path,cv2.IMREAD_GRAYSCALE)
    img_x = x_transforms(origin_x)
    x = img_x.unsqueeze(0).to(device)
    y = model(x)
    predict = y.squeeze(0).data.cpu().numpy()
    predict = predict[0]
    height = predict.shape[0]
    weight = predict.shape[1]
    # print(height*weight)
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                predict[row, col] = 0
            else:
                predict[row, col] = 255
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1
    predict = predict.astype(np.int16)

    cv2.imwrite("./31.jpg", predict)


    img = cv2.imread('./3.jpg', -1)
    ret, thresh = cv2.threshold(img, 150, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(origin_x, contours, -1, (120, 155, 255), 1)
    cv2.imshow('result', origin_x)

    img_label = cv2.imread(x_path_label, -1)
    ret_label, thresh_label = cv2.threshold(img_label, 150, 255, 0)
    contours_label, hierarchy_label = cv2.findContours(thresh_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(origin_x, contours_label, -1, (255, 255, 255), 1)
    cv2.imshow('result_label', origin_x)



    cv2.waitKey(0)