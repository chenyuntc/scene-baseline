#coding:utf8
import torchvision as tv
from torch import nn
from basic_module import BasicModule
import torch as t

class ResNet(BasicModule):
    def __init__(self,model,opt=None,feature_dim=2048,name='resnet'):
        super(ResNet, self).__init__(opt)
        self.model_name=name
        
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        del model.fc
        model.fc = lambda x:x
        self.features = model
        self.classifier = nn.Linear(feature_dim,80)

    def forward(self,x):
        features = self.features(x)
        return self.classifier(features)

def resnet18(opt):
    model = tv.models.resnet18(pretrained=not opt.load_path)
    return ResNet(model,opt,feature_dim=512,name='res18')

def resnet34(opt):
    model = tv.models.resnet34(pretrained=not opt.load_path)
    return ResNet(model,opt,feature_dim=512,name='res34')

def resnet50(opt):
    model = tv.models.resnet50(pretrained=not opt.load_path)
    return ResNet(model,opt,name='res50')

def resnet101(opt):
    model = tv.models.resnet101(pretrained=not opt.load_path)
    return ResNet(model,opt,name='res101')

def resnet152(opt):
    model = tv.models.resnet152(pretrained=not opt.load_path)
    return ResNet(model,opt,name='res152')

def resnet365(opt):
    model = t.load('checkpoints/whole_resnet50_places365.pth.tar')
    # model = tv.models.resnet50()
    return ResNet(model,opt,name='res_365')
