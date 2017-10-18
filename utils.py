#coding:utf8
import visdom
import numpy as np

class Visualizer():
    '''
    对可视化工具visdom的封装
    '''
    def __init__(self, env, **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_many(self, d):
        for k, v in d.iteritems():
            self.plot(k, v)

    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=unicode(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

def topk_acc(score,label,k=3):
    '''
    topk accuracy,默认是top3准确率
    '''
    topk = score.topk(k)[1]
    label = label.view(-1,1).expand_as(topk)
    acc = (label == topk).float().sum()/(0.0+label.size(0))
    return acc

class Config:
    train_dir = '/data/image/ai_cha/scene/sl/train/'
    test_dir = '/data/image/ai_cha/scene/sl/testa'
    val_dir = '/data/image/ai_cha/scene/sl/val'
    meta_path = '/data/image/ai_cha/scene/sl/scene.pth'
    img_size=256

    lr1 = 0
    lr2 = 0.0005
    lr_decay = 0.5
    batch_size = 128 
    max_epoch = 100  
    debug_file = '/tmp/debugc'
    shuffle = True
    env = 'scene'  # visdom env
    plot_every = 10 # 每10步可视化一次

    workers = 4 # CPU多线程加载数据
    load_path=None#
    model = 'resnet50'# 具体名称查看 models/__init__.py 
    loss='celoss'
    result_path='result.json' #提交文件保存路径

def parse(self,kwargs,print_=True):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.iteritems():
            if not hasattr(self,k):
                raise Exception("opt has not attribute <%s>" %k)
            setattr(self,k,v) 
        if print_:
            print('user config:')
            print('#################################')
            for k in dir(self):
                if not k.startswith('_') and k!='parse' and k!='state_dict':
                    print k,getattr(self,k)
            print('#################################')
        return self


def state_dict(self):
    return  {k:getattr(self,k) for k in dir(self) if not k.startswith('_') and k!='parse' and k!='state_dict' }

Config.parse = parse
Config.state_dict = state_dict
opt = Config()
