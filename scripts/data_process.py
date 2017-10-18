# 数据预处理脚本



train_ann_file = '/data/image/ai_cha/scene/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
val_ann_file = '/data/image/ai_cha/scene/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
test_dir = '/data/image/ai_cha/scene/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922/'
clas_file = '/data/image/ai_cha/scene/ai_challenger_scene_test_a_20170922/scene_classes.csv'

import pandas as pd
from collections import namedtuple
# ClassName = namedtuple('ClassName',['zh','en'])
# SceneDataAll = namedtuple('SceneDataAll',['train','val','test1','label'])
# SceneData = namedtuple('SceneData',['ids','labels','id2ix'])


a=pd.read_csv(clas_file,header=None)
label_ids,label_zh,label_en = a[0],a[1],a[2]

#79: ClassName(zh=u'\u96ea\u5c4b/\u51b0\u96d5(\u5c71)', en='igloo/ice_engraving')}
id2label = {k:(v1.decode('utf8'),v2) for k,v1,v2 in zip(label_ids,label_zh,label_en)}
# id2label = {k:ClassName(v1.decode('utf8'),v2) for k,v1,v2 in zip(label_ids,label_zh,label_en)}

import json
with open(train_ann_file) as f:
    datas = json.load(f)

ids = [ii['image_id'] for ii in datas]
labels = [int(ii['label_id']) for ii in datas]
id2ix = {id:ix for ix,id in enumerate(ids)}


#train = SceneData(ids,labels,id2ix)
train = dict(ids=ids,labels = labels,id2ix=id2ix)

with open(val_ann_file) as f:
    datas = json.load(f)

ids = [ii['image_id'] for ii in datas]
labels = [int(ii['label_id']) for ii in datas]
id2ix = {id:ix for ix,id in enumerate(ids)}
val = dict(ids=ids,labels = labels,id2ix=id2ix)
# val = SceneData(ids,labels,id2ix)
import os
ids = os.listdir(test_dir)
id2ix = {id:ix for ix,id in enumerate(ids)}
# test = SceneData(ids,None,id2ix)
test = dict(ids=ids,labels = labels,id2ix=id2ix)
# all = SceneDataAll(train,val,test,id2label)

all = dict(train=train,test1=test,val=val,id2label=id2label)
import torch as t
t.save(all,'scene.pth')