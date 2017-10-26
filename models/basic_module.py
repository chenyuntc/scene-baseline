
#coding:utf8
import torch as t
import time
class BasicModule(t.nn.Module):
    '''
    封装了nn.Module
    '''

    def __init__(self,opt=None):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self).__name__)# 默认名字
        self.opt = opt
        
    def load(self, path,map_location=lambda storage, loc: storage):
        checkpoint = t.load(path,map_location=map_location)
        if 'opt' in checkpoint:
            self.load_state_dict(checkpoint['d'])
            print('old config：')
            print(checkpoint['opt'])
        else:
            self.load_state_dict(checkpoint)
        # for k,v in checkpoint['opt'].items():
        #     setattr(self.opt,k,v)

    def save(self, name=''):
        format = 'checkpoints/'+self.model_name+'_%m%d_%H%M_'
        file_name = time.strftime(format) + str(name)
        
        state_dict = self.state_dict()
        opt_state_dict = dict(self.opt.state_dict())
        optimizer_state_dict = self.optimizer.state_dict()

        t.save({'d':state_dict,'opt':opt_state_dict,'optimizer':optimizer_state_dict}, file_name)
        return file_name

    def get_optimizer(self,lr1,lr2):
        self.optimizer =  t.optim.Adam(
            [
             {'params': self.features.parameters(), 'lr': lr1},
             {'params': self.classifier.parameters(), 'lr':lr2}
            ] )
        return self.optimizer

    def update_optimizer(self,lr1,lr2):
        param_groups = self.optimizer.param_groups
        param_groups[0]['lr']=lr1
        param_groups[1]['lr']=lr2
        return self.optimizer
