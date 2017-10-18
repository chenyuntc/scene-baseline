#coding:utf8
import os,json,time,fire,ipdb,tqdm

import numpy as np
import torch as t
from torch import nn, optim
from torch.autograd import Variable
from torchnet import meter

from utils import Visualizer,topk_acc,opt
from dataset import ClsDataset
import models

vis = Visualizer(env=opt.env)

def submit(**kwargs):
    '''
    测试验证集，并生成可提交的json文件
    '''
    opt.parse(kwargs)
    
    # 模型
    model = getattr(models,opt.model)(opt)
    model.load(opt.load_path)
    model.eval().cuda()

    # 数据
    dataset = ClsDataset(opt)
    dataset.test()
    dataloader =  t.utils.data.DataLoader(dataset,opt.batch_size, shuffle=False, num_workers=opt.workers,pin_memory=True)
    
    # 测试
    results = []
    for ii, data in tqdm.tqdm(enumerate(dataloader)):
        input, label,image_ids = data
        val_input = Variable(input, volatile=True).cuda()
        val_label = Variable(label.type(t.LongTensor), volatile=True).cuda()
        score = model(val_input)
        predict = score.data.topk(k=3)[1].tolist()
        result = [  {"image_id": image_id,
                     "label_id": label_id }
                         for image_id,label_id in zip(image_ids,predict) ] 
        results+=result

    # 保存文件
    with open(opt.result_path,'w') as f:
        json.dump(results,f)


def val(model,dataset):
    '''
    计算模型在验证集上的准确率
    返回top1和top3的准确率 
    '''
    model.eval()
    dataset.val()
    acc_meter = meter.AverageValueMeter()
    top1_meter = meter.AverageValueMeter()
    dataloader =  t.utils.data.DataLoader(dataset,opt.batch_size, opt.shuffle, num_workers=opt.workers,pin_memory=True)
    for ii, data in tqdm.tqdm(enumerate(dataloader)):
        input, label,_ = data
        val_input = Variable(input, volatile=True).cuda()
        val_label = Variable(label.type(t.LongTensor), volatile=True).cuda()
        score = model(val_input)
        acc = topk_acc(score.data,label.cuda()  )
        top1 = topk_acc(score.data,label.cuda(),k=1)
        acc_meter.add(acc)
        top1_meter.add(top1)
    model.train()
    dataset.train()
    print(acc_meter.value()[0],top1_meter.value()[0])
    return acc_meter.value()[0], top1_meter.value()[0]


def train(**kwargs):
    '''
    训练模型
    '''
    opt.parse(kwargs)

    lr1, lr2 = opt.lr1, opt.lr2
    vis.vis.env = opt.env
    
    # 模型
    model = getattr(models,opt.model)(opt)
    if opt.load_path:
        model.load(opt.load_path)
    print(model)
    model.cuda()
    optimizer = model.get_optimizer(lr1,lr2)
    criterion = getattr(models,opt.loss)()

    # 指标：求均值
    loss_meter = meter.AverageValueMeter()
    acc_meter = meter.AverageValueMeter()
    top1_meter = meter.AverageValueMeter()
    
    step = 0
    max_acc = 0
    vis.vis.texts = ''

    # 数据
    dataset = ClsDataset(opt)
    dataloader =  t.utils.data.DataLoader(dataset, opt.batch_size, opt.shuffle, num_workers=opt.workers,pin_memory=True)
    
    # 训练
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        acc_meter.reset()
        top1_meter.reset()

        for ii, data in tqdm.tqdm(enumerate(dataloader, 0)):
            # 训练
            optimizer.zero_grad()
            input, label,_ = data
            input = Variable(input.cuda())
            label = Variable(label.cuda())
            output = model(input).squeeze()
            error = criterion(output, label)
            error.backward()
            optimizer.step()

            # 计算损失的均值和训练集的准确率均值
            loss_meter.add(error.data[0])
            acc = topk_acc(output.data,label.data)
            acc_meter.add(acc)
            top1_acc = topk_acc(output.data,label.data,k=1)
            top1_meter.add(top1_acc)

            # 可视化
            if (ii+1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                log_values = dict(loss = loss_meter.value()[0],
                                   train_acc = acc_meter.value()[0],
                                     epoch = epoch, 
                                    ii = ii,
                                    train_top1_acc= top1_meter.value()[0]
                                    )
                vis.plot_many(log_values)
        
        # 数据跑一遍之后，计算在验证集上的分数
        accuracy,top1_accuracy = val(model,dataset)
        vis.plot('val_acc', accuracy)
        vis.plot('val_top1',top1_accuracy)
        info = time.strftime('[%m%d_%H%M%S]') + 'epoch:{epoch},val_acc:{val_acc},lr:{lr},val_top1:{val_top1},train_acc:{train_acc}<br>'.format(
            epoch=epoch,
            lr=lr1,
            val_acc=accuracy,
            val_top1=top1_accuracy,
            train_acc=acc_meter.value()
        )
        vis.vis.texts += info
        vis.vis.text(vis.vis.texts, win=u'log')

        # 调整学习率
        # 如果验证集上准确率降低了，就降低学习率，并加载之前的最好模型
        # 否则保存模型，并记下模型保存路径
        if accuracy > max_acc:
            max_acc = accuracy
            best_path = model.save(accuracy)
        else:
            if lr1==0:	lr1=lr2
            model.load(best_path)
            lr1, lr2 = lr1 *opt.lr_decay, lr2 * opt.lr_decay
            optimizer = model.get_optimizer(lr1,lr2)

        vis.vis.save([opt.env])

if __name__ == '__main__':
    fire.Fire()
