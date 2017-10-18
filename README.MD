# PyTorch Baseline for AI challenger scene
AI Challenger [场景分类竞赛](https://challenger.ai/competition/scene/)的示例代码

## 使用
### 1.环境配置
安装：
- [PyTorch](http://pytorch.org/),根据官网说明下载指定的版本即可。
- 第三方依赖： `pip install -r requirements.txt`

### 2.数据预处理：

这一步主要是根据json文件进行简单的处理，生成二进制scene.pth，可以跳过这一步，直接下载随程序附带的scene.pth

如果你想自己生成scene.pth，修改scripts/data_process.py 中的文件路径，然后运行
```bash
python scripts/data_process.py
```

### 3.启动visdom
可视化工具[visdom](https://github.com/facebookresearch/visdom)
```bash
nohup python2 -m visdom.server&
```

### 4.训练
注意修改utils.py 中文件路径

```bash
python main.py train --model='resnet34'
```
在 Titan Xp下，大概90分钟可以在验证集上得到大约0.94的准确率

打开浏览器 输入http://ip:8097 可以看到训练过程。visdom 中要用到两个js文件`plotly.min.js`和`react-grid-layout.min.js`，这两个js文件被防火墙所拦截~ 所以你可能需要自备梯子才能用visdom。

另外一个解决方法是：
`locate locate visdom/static/index.html`,修改index.html中两行js的地址

### 5.提交
```bash
python main.py submit --model='resnet34' --load-path='res34_1014_1222_0.932421875' 

```
会在当前目录生成`result.json`文件，直接诶提交即可

## 关于CPU运行
把所有`.cuda()`代码去掉，就能使得程序在CPU上运行


## 各个文件说明
欢迎参考之前在[知乎专栏](https://zhuanlan.zhihu.com/p/29024978)写过的一篇关于PyTorch的文件组织安排的文章了解每个文件的作用

`models/`: 存放各个模型定义，所有的模型继承自`basic_module.py`中的`BasicModule`.

`models/resnet.py`: 对torchvision中的resnet18, resnet34, resnet50, resnet101 和resnet152 进行了简单的封装。

`dataset.py`: 数据加载相关

`main.py`: 主程序，包含训练和测试

`utils.py`: 可视化工具visdom的封装，计算top3准确率函数，可配置变量（可通过命令行参数修改，也可以通过修改文件配置）等。
