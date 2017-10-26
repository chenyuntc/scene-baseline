#coding:utf8
import torch as t
from torchvision import transforms
from torch.utils import data
import os
import PIL
import random

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def load(path):
    return PIL.Image.open(path).convert('RGB')

class ClsDataset(data.Dataset):
    def __init__(self,opt):
        self.opt = opt
        self.datas = t.load(opt.meta_path)
        
        self.val_transforms =  transforms.Compose([
            transforms.Scale(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
            normalize,
        ])
        self.train_transforms =  transforms.Compose([
            transforms.RandomSizedCrop(opt.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.train()

    def __getitem__(self,index):
        img_path = os.path.join(self.path,self.imgs[index])
        img = load(img_path)
        img = self.transforms(img)
        return img,self.labels[index],self.imgs[index]

    def train(self):
        data = self.datas['train']
        self.imgs,self.labels = data['ids'],data['labels']
        self.path = self.opt.train_dir
        self.transforms = self.train_transforms
        return self

    def test(self):
        data= self.datas['test1']
        self.imgs,self.labels = data['ids'],data['labels']
        self.path = self.opt.test_dir
        self.transforms=self.val_transforms
        return self

    def val(self):
        data = self.datas['val']
        self.imgs,self.labels = data['ids'],data['labels']
        self.path = self.opt.val_dir
        self.transforms=self.val_transforms
        return self

    def __len__(self):
        return len(self.imgs)

if __name__=='__main__':
   test 
