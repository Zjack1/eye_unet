import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
from torchvision.transforms import transforms
from torch.utils.data import Dataset


class VOCDatasets(Dataset):

    def __init__(self):
        self.path = r"C:\Users\shzhoujun\Desktop\Unet\data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012"
        # 语义分割需要的图片的加载进来，做标签，总共2913张图片
        self.name = os.listdir(os.path.join(self.path, "SegmentationClass"))
        self.trans = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    def __len__(self):
        return len(self.name)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真
    def __trans__(self, img, size):
        # 图片的宽高
        h, w = img.shape[0:2]
        # 需要的尺寸
        _w = _h = size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):
        # 拿到的图片
        name = self.name[index]
        # 把标签名的格式改成jpg，与原始图片一致
        name2jpg = name[:-3] + "jpg"

        # 所有的原始图片和标签
        img_path = [os.path.join(self.path, i) for i in ("JPEGImages", "SegmentationClass")]
        # 读取原始图片和标签，并转RGB
        img_o = cv2.imread(os.path.join(img_path[0], name2jpg))
        img_l = cv2.imread(os.path.join(img_path[1], name))
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

        # 转成网络需要的正方形
        img_o = self.__trans__(img_o, 256)
        img_l = self.__trans__(img_l, 256)
        img_o_ = self.trans(img_o)
        img_l_ = self.trans(img_l)
        return img_o_, img_l_


class BruceDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"C:\Users\shzhoujun\Desktop\eye_unet\data_train\img_and_label"
        self.val_root = r"C:\Users\shzhoujun\Desktop\eye_unet\data_train\test"
        self.test_root = self.val_root
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        self.name_img = os.listdir(root)
        for i in range(len(self.name_img)):
            img_name = self.name_img[i]
            if img_name[-5] == "t":
                img = os.path.join(root,img_name)
                pics.append(img)
            else:
                mask = os.path.join(root,img_name)
                masks.append(mask)
        return pics, masks

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path)
        origin_y = Image.open(y_path)
        # origin_x = cv2.imread(x_path)
        y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)



class LiverDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"C:\Users\shzhoujun\Desktop\Unet\data\liver\train"
        self.val_root = r"C:\Users\shzhoujun\Desktop\Unet\data\liver\val"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "%03d.png" % i)  # liver is %03d
            mask = os.path.join(root, "%03d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
            #imgs.append((img, mask))
        return pics, masks

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path)
        origin_y = Image.open(y_path)
        # origin_x = cv2.imread(x_path)
        # origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)

class esophagusDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"E:\datasets\data_sta_all\train_data"
        self.val_root = r"E:\datasets\data_sta_all\test_data"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root
        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "%05d.png" % i)  # liver is %03d
            mask = os.path.join(root, "%05d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
            #imgs.append((img, mask))
        return pics,masks

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)

class dsb2018CellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'E:\codes\pytorch-nested-unet-master\pytorch-nested-unet-master\input\dsb2018_256'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths = None,None
        self.train_mask_paths, self.val_mask_paths = None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + '\images\*')
        self.mask_paths = glob(self.root + '\masks\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.val_img_paths,self.val_mask_paths  #因数据集没有测试集，所以用验证集代替

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class CornealDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'E:\datasets\CORN\CORN\Corneal nerve curivilinear segmentation\Corneal nerve curivilinear segmentation'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.train_img_paths = glob(self.root + r'\training\train_images\*')
        self.train_mask_paths = glob(self.root + r'\training\train_mask\*')
        self.val_img_paths = glob(self.root + r'\val\val_images\*')
        self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        self.test_img_paths = glob(self.root + r'\test\test_images\*')
        self.test_mask_paths = glob(self.root + r'\test\test_mask\*')
        # self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
        #     train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class DriveEyeDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'E:\datasets\DRIVE\DRIVE'
        self.pics, self.masks = self.getDataPath()
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.train_img_paths = glob(self.root + r'\training\images\*')
        self.train_mask_paths = glob(self.root + r'\training\1st_manual\*')
        self.val_img_paths = glob(self.root + r'\test\images\*')
        self.val_mask_paths = glob(self.root + r'\test\1st_manual\*')
        self.test_img_paths = self.val_img_paths
        self.test_mask_paths = self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        imgx,imgy=(576,576)
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        #print(pic_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        if mask == None:
            mask = imageio.mimread(mask_path)
            mask = np.array(mask)[0]
        pic = cv2.resize(pic,(imgx,imgy))
        mask = cv2.resize(mask, (imgx, imgy))
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class IsbiCellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'E:\datasets\isbi'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\train\images\*')
        self.mask_paths = glob(self.root + r'\train\label\*')
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        # self.test_img_paths = glob(self.root + r'\test\test_images\*')
        # self.test_mask_paths = glob(self.root + r'\test\test_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class LungKaggleDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'E:\datasets\finding-lungs-in-ct-data-kaggle'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\2d_images\*')
        self.mask_paths = glob(self.root + r'\2d_masks\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)