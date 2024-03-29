# Notebook
# 
# python  
  1. 搜索路径  

    ```sys.path```  
    返回当前搜索路径list，如果有ModuleNotFoundError可以看看路径是否在这个list中  
    可以通过  
    ```sys.path.insert(0, path)```  
    或者  
    '''sys.path.append(os.path.joint(os.path.dirname(__file__), '..'))'''  
    添加搜索路径  
    ```os.path.dirname(__file__)```  
    可以返回当前文件的路径  
    ```os.path.split(path)```  
    可以对路径拆分最后一个目录 或者 文件  
  2. 对文件名、路径操作  

    https://www.cnblogs.com/sigai/p/8074329.html  
  ```
  from pathlib import Path
  path = Path('path')
  path.exists()
  path.mkdir(parents=True, exist_ok=True)
  Path / '路径' # 任然是Path
  os.path.basename(path) # 返回路径最后一个文件（夹）
  path.split('.') # 剥离文件名与格式
  ```
  3. 格式化字符串  
  ```
  str = 'hello {place}'.format(place='wuhan')
  str = '%-10s' % （'china') # 左对齐 长度为10
  ```
  4. 时间
  ```
  import time
  time_str = time.strftime('%Y-%m-%d-%H-%M')
  ```
  5. 输出日志logging

    https://www.cnblogs.com/goodhacker/p/3355660.html
  ```
  import logging
  head = '%(asctime)-15s %(message)s' # 输出格式  时间-信息
  logging.basicConfig(filename=path, format=head) # 定义输出日志路径 和 日志格式
  logger = logging.getLogger() # 相当于实例化
  logger.setLevel(logging.INFO) # 日志信息有多个等级 只有大于等于INFO的等级才会输出
  console = logging.StreamHandler() # 创建在终端输出的handler
  logger.addHandler(console) # 加到logger上
  logger.info('massage')
  ```
  在其他模块中用
  ```
  import logging
  logger = logging.getLogger(__name__)
  ```
  6. pprint  

    全称 pretty printer，可以让各种数据结构更美观地输出  
    ```pprint.pprint(dict())```美化后打印  
    ```pprint.pformat(dict())```返回美化后的字符串  
  7. python package  

    python package是一个文件夹，该文件夹下面有__init__.py，还有其他.py文件。如果要调用python package下的.py文件，应该在__init__.py文件中加入  
  ```
  import models.pose_resnet
  ```
  models就是package文件夹，pose_resnet就是pose_resnet.py文件  
  8. 字典的key可以是hashable（不可变）的数据类型，所以可以用tuple作为key  
  9. 代码格式
  安装yapf
  在~/.config/yapf/下
  ```yapf --style-help > yapf_style.cfg
     mv yapf_style.cfg style
     vim style
     
     # 修改配置文件
     arithmetic_precedence_indication=True
     ...
  ```
  也可以直接下载[配置文件](https://github.com/zhoujuncug/Notebook/blob/master/style)  
  
# 深度学习中琐碎的知识
## 逆卷积计算公式
  1. 卷积后图像大小  

    o = (i-1)*s - k + 2p + 2  
    https://zhuanlan.zhihu.com/p/57348649
## NumPy
  1. 没有省略的显示矩阵
  ```
  import numpy as np
  np.set_printoptions(threshold=np.inf, linewidth=4000)
  ```

  2. 矩阵乘法  

    a*b 是对应相乘即broadcast。  
    而a.dot(b)才是熟悉的矩阵点乘
  3. 交换某行（列)
  ```
  import numpy as np 
  a = np.array([[1, 2], [3, 4]])
  # 若要交换第一行与第二行
  a[0], a[1] = a[1], a[0] # 这是错的
  a[[0, 1], :] = a[[1, 0], :] #这才是正确的
  ```
  4. argmax, amax  
  ```
  a = np.argmax(b, 2) # b.shape = 2, 17, n  返回shape=2, 17 在第2个维度上找最大值的索引
  a = np.amax(a, 2)# b.shape = 2, 17, n  返回shape=2, 17 在第2个维度上找最大值
  ```
  5. tile
  ```
  a = np.tile(b, (1, 1, 2)) # b.shape 2, 17, n 返回shape=2, 17,2n b在第2个维度上copy一次
  ```
  6. mask
  ```
  a = np.greater(b, 0) # 返回b大于0的mask
  a = np.less(b, 0)
  a = np.equal(b, 0) # 等于0的mask
  a = np.not_equal(b, 0) 
  a = a.astype(np.float32) # 把mask转为数
  ```
  7. 反转
  ```
  b = np.flip(a, 3) # 把a沿着第三维度反转
  b = a[:, :, ::-1]
  ```
  8.copy()
  ```
  b = a.copy()
  ```
  8. cv2图像与tensor 图像 

    np.ndarray的[h, w, c] rgb格式  Tensor的[c, h, w]  
  ```
  image_chw = np.transpose(image_hwc, (2,0,1)) 相互转换
  ```
  cv2显示图像，如果np.array是np.float32的  像素范围是（0， 1），如果是uint8则（0， 255）
  ```
  image_uint = image_float.astype(np.uint8)
  image_float = image_uint8.astype(np.float32)
  ```
  9. boardcast  
  参考：https://blog.csdn.net/qq_33228413/article/details/106204625?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242  
  10. reshape
  参考 https://zhuanlan.zhihu.com/p/72381219
  
## Visdom  
  0. download scripts  

    https://github.com/casuallyName/document-sharing/tree/master/static  
  1. launch visdom  

    ```python -m visdom.server```  
  2. Open http://localhost:8097 in your browser  
  3. Dynamic curve  
  ```
  from visdom import Visdom
import numpy as np
import time


viz = Visdom(env='loss')
x, y = 0, 0
win = viz.line(X=np.array([x]), Y=np.array([y]))

for i in np.linspace(0, 10 * np.pi, 1000):
    x = i
    y = np.sin(x)

    viz.line(X=np.array([x]), Y=np.array([y]), win=win, update='append')
    time.sleep(0.00001)
  ```
   4. More examples  
      https://blog.csdn.net/cpongo3/article/details/93624339?utm_source=app  
      https://www.pytorchtutorial.com/pytorch-visdom/  

## torch
  1. 转换通道  

    ```image.permute(0, 3, 1, 2)```
  2. mul方法
  ```
  a = torch.tensor(........)
  a.mul(b)
  ```
  如果b为一个数，则a的所有元素乘b
  如果b为一个size为（1， a.size(1))或者（a.size),则为a的每一行，对应乘b
  如果b为一个size为（a.size(0), 1),则为a的每一列，对应乘b
  对于tensor，不能像numpy那样用.shape,而是.size()  
  3. pretrained model download urls  

    https://blog.csdn.net/lxy_2011/article/details/97488494  
    https://pytorch.org/docs/stable/torchvision/models.html  
  4. pretrained model and init_weights  

    ```checkpoint = torch.load('pretrained_model.pth')```  
    可以读取预训练模型参数，checkpoint为一个OrderedDict，OrderedDict.items()返回keys与values，OrderedDict.keys()返回key。  
    对于一个model，可以用model.named_parameters()和model.parameters()返回模型的参数。  
    对于某个parameter可以通过  
    ```param.data = X```  
    对某个parameter赋值  
  5. torch.backends.cudnn
  ```
  import torch.backends.cudnn as cudnn
  cudnn.benchmark = True
  ```
  这行代码可以在网络结构和输入不变的情况下会减少训练时间
  https://blog.csdn.net/byron123456sfsfsfa/article/details/96003317
  ```
  torch.backends.cudnn.deterministic = True
  ```
  让cuda实现reproducibility,当然要是同一个pytorch版本、同一个硬件平台，而且torch、numpy都使用伪随机数的条件下  
  https://pytorch.org/docs/stable/notes/randomness.html
  ```
  torch.backends.cudnn.enabled = False
  ```
  禁用cudnn  
  6. 多GPU并行  

    ```model = torch.nn.DataParallel(model, device_ids=gpus)```
    gpus为一个list,[0, 1, 2, 3] 分别为gpu代号
  7. loss计算

    把损失计算封装到nn.Module的子类中，就像是写model，也有forward。如下：
    ```criterion = JointsMSELoss().cuda()```
    其好处，我猜测是可以用gpu加速

  8. nn.MseLoss()
  ```
  mse = nn.MSELoss()
  loss = mse(a, b) # a b 可以为相同的任意维度，如果报错a,b转为.float()
  ```
  9. 自定义层  

    自定义层的权重应该在 ```self.__init__()```下面定义而且必须是nn.Parameter() 比如  
    ```self.weight = torch.nn.Parameter(torch.Tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True))```  
  10. 优化器

    优化器的第一个输入是一个列表，该列表下面是tensor。  
    所以要对一个参数a 优化，则应该输入```[a]```  
11. hook

hook主要是用于显示变量的梯度或可视化网络的输入输出。可以在跑网络前用hook显示网络结构。

https://zhuanlan.zhihu.com/p/75054200



## torchvision

  1. 把一个batch中的多个图像，拼接为一个超大的图像，便于显示
  ```
  # output.shape为(batch_size, channel, w, h)
  cat_img = torchvision.utils.make_grid(output, nrow=8, padding=2)
  # 注意，此时cat_img的shape(3， w*8, h*batch_size/8) 如果用opencv显示，需要转一下通道(w, h, 3)
  ```
  2. transporms.ToTensor()不仅仅是把图像转为tensor，而且还会归一化

    一般经过
  ```
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  transforms.Compose([transforms.ToTensor(), normalize,]))
  ```
  图像会归一化后，然后3个通道分别减去mean，除std
  因此要把数据还原，首先要乘std，加mean，还要逆向归一化
## tensorboardX  
  https://blog.csdn.net/bigbennyguo/article/details/87956434  
  TensorboardX 这个工具使得 TensorFlow 外的其他神经网络框架也可以使用到 Tensorboard 的便捷功能.  
  首先要安装tensorboardX 和 tensorflow，在网页可视化需要tensorflow.  
  ```
  from tensorboardX import SummaryWriter
  writer = SummaryWriter('logdir') #首先创建一个writer
  writer.add_scalar('data_name', data, global_step=epoch) # 这是添加数字的方法，分别是数字的名字，数字(float)，epoch
  writer.add_graph(model, (dump_input, ), verbose=False) # 这是添加模型的方法，dump_input，是一个torch.tensor(batch_size,channel,width,height)
  # 还有add_image,记录图像等等
  writer.close() # 这个要添加一下，否者可视化的时候可能有问题
  ```
 ## Pretrained Model  
 https://blog.csdn.net/weixin_36047799/article/details/103564627  

## COCO2017
  https://blog.csdn.net/gzj2013/article/details/82385164
  安装COCO API
  1. 安装git
     ```
     sudo add-apt-repository ppa:git-core/ppa       //添加源
     sudo apt-get update                                          //更新
     sudo apt-get install git                //自动安装git
     git --version                                                       //确认git版本
     ```
   2. 安装coco api

     # COCOAPI=/path/to/clone/cocoapi
     git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
     cd $COCOAPI/PythonAPI
     # Install into global site-packages
     pip install Cython
     make install
     # Alternatively, if you do not have permissions or prefer
     # not to install the COCO API into global site-packages
     python3 setup.py install --user
   3. coco api

     from pycocotools.coco import COCO
     coco = COCO('coco_annotation_file')
     coco.getCatIds() # 返回annotation文件中类型的索引，比如：人，猫，滑板
     coco.loadCats([1]) # 返回索引1中类型包含的信息
     coco.getImgIds(catIds=*) # 返回*类型图像的索引
     coco.loadImgs(391895) # 返回索引为391895的图像的信息
     coco.getAnnIds(391895) # 返回索引为391895的图像的标注索引 可能有多个标注索引
     coco.loadAnns((coco.getAnnIds(391895))[0]) # 返回索引为391895的图像的标注索引[0]的标注信息

## opencv python
1. 仿射变换
   ```
   import cv2
   # src dst为3x2的np.array 为变换前后的 3个对应坐标
   trans = cv2.getAffineTransform(np.float32(src),np.float32(dst)) # trans为2x3的矩阵
   # 对于任一坐标可以通过下面代码得到变换后的位置
   coor = np.array([coor[0], coor[1], 1]).T
   new_coor = np.dot(trans, coor)
   # 对于图像
   input = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])), # 变换后的图像shape
            flags=cv2.INTER_LINEAR
        )
   ```
 ## imgaug  
https://imgaug.readthedocs.io/en/latest/index.html  
https://blog.csdn.net/limiyudianzi/article/details/86497345  
mgaug is a library for image augmentation in machine learning experiments.  
bbox  
https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B02%20-%20Augment%20Bounding%20Boxes.ipynb  
keypoints  
https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B01%20-%20Augment%20Keypoints.ipynb  

 ## Tensorflow-gpu 
 Version of tensorflow must be corresponded to version of cuda  
 https://tensorflow.google.cn/install/source#linux  
 ```
 # if cuda == 11.0
 pip install tensorflow==2.4
 ```
 bug: Could not load dynamic library 'libcudnn.so.8'  
 https://blog.csdn.net/qq_46244851/article/details/110731411  
 

# 重装系统记录
# 一定！一定！一定！记得要备份，如果实在想不到要备份什么，起码要把文档备份了！
## 系统
1. 安装双系统时，应该先安装win，然后ubuntu
2. 安装win后，如果想把ubuntu安装到E盘（比如）那么删除E盘，其他盘要激活
3. 安装ubuntu时，选择 ubuntu alongside with win (这时，ubuntu会默认安装到第一个未分配的盘)
4. 如果是ubuntu系统 可以在开机时选择进入bios  win下，按esc
5. **安装完Ubuntu，记得换源  
   ```
   sudo cp /etc/apt/sources.list /etc/apt/sources_init.list
   sudo gedit /etc/apt/sources.list 
   ```
   把里面的都删除，copy下面的，保存
   ```
   deb http://mirrors.aliyun.com/ubuntu/ xenial main
   deb-src http://mirrors.aliyun.com/ubuntu/ xenial main

   deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main
   deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main

   deb http://mirrors.aliyun.com/ubuntu/ xenial universe
   deb-src http://mirrors.aliyun.com/ubuntu/ xenial universe
   deb http://mirrors.aliyun.com/ubuntu/ xenial-updates universe
   deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates universe

   deb http://mirrors.aliyun.com/ubuntu/ xenial-security main
   deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main
   deb http://mirrors.aliyun.com/ubuntu/ xenial-security universe
   deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security universe
   ```
   然后
   ```
   sudo apt-get update
   sudo apt-get -f install
   sudo apt-get upgrade
   ```


## CUDA
1. 显卡驱动  
   ## new method
   1.1 open Software & Update
   1.2 Additional Drivers
   1.3 choose one driver for Nvidia gpu
   
   ## The following method is probably out-of-date
   1.1 ```sudo add-apt-repository ppa:graphics-drivers/ppa```  
   1.2 ```sudo apt-get update```  
   1.3 ```ubuntu-drivers devices```  
       这里记录一下recommend 的driver 型号，我的是nvidia-driver-430  
   1.4 ```sudo apt install nvidia-driver-440  
       default option if need  
   1.5 重启  
   1.6 ```nvidia-smi  
       有信息显示就OK  
   ### troubleshoot  
       No additional driver available.  
       ```  
          sudo add-apt-repository ppa:graphics-drivers/ppa  
          sudo apt-get update  
       ```
   
2. cuda  
   2.1 https://developer.nvidia.com/cuda-toolkit-archive  
       Operating: System Linux  
       Architecture: x86_64  
       Distribution: Ubuntu  
       Version: 18.04  
       Installer Type: deb(local)  
       10-1, you must select this version  
       https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal  
       some pkg has been downloaded in my passport 
       然后官网会提示下载的命令  
       安装完成后 在安装目录/usr/local/cuda-10.1/samples/1_Utilities/deviceQuery 中打开 terminal  
       ```sudo make  
       ```sudo ./deviceQuery  
       如果有 Result = PASS 就OK 不成功就重启一下试试  
   
3. cuDNN  
   3.1 https://developer.nvidia.com/cudnn  
       需要账号，我自己是用QQ登录就行了  
       Download cuDNN  
       I Agree To the Terms of the cuDNN Software License Agreement  
       Download cuDNN v7.6.1 (June 24, 2019), for CUDA 10.1   
       cuDNN Library for Linux  
   3.2 解压  
   3.3 执行下面的代码  
    ```
       sudo cp cuda/include/cudnn.h /usr/local/cuda/include  
       sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64  
       sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    ```
   
4. pip3及换源  
   ```sudo apt-get install python3-pip```
   换源  
   ```
   mkdir ~/.pip
   gedit ~/.pip/pip.conf
   ```
   copy如下内容保存
   ```
   [global]
   index-url = https://pypi.mirrors.ustc.edu.cn/simple/
   ```
   
   Timeout
   
   ```
   pip --default-timeout=100 install numpy
   ```
   
   
## 火狐浏览器
   1. 链接不安全
      搜索框输入 ```about:config```
      选 我了解此风险
      输入 ```security.enterprise_roots.enabled```
      切换为 TRUE
## TIM  
   https://gitee.com/wszqkzqk/deepin-wine-for-ubuntu

## 向日葵  
  1. 连接后，显示已断开  
     需要安装并使用lightdm  
     ``` 
     sudo apt-get install lightdm  
     duso dpkg-reconfigure lightdm
     ```
     
## Git
1. [本地上传到远端, 远端不存在时](https://blog.csdn.net/qq_38408785/article/details/89322266)  
2. [本地上传到远端，远端已经存在时](http://www.fwxstar.com/code/16_77)  
3. 有时候，code和output文件夹不在同一处，可以[创建软连接](https://www.jianshu.com/p/cf35ffff3afa)把output放在code下  

## Docker proxy
``` export http_proxy=http://yourproxyaddress:proxyport ```  
``` export https_proxy=http://yourproxyaddress:proxyport ```  

## VS Code + docker container
  [教程](https://blog.csdn.net/weixin_42548893/article/details/112452997?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242)  
  1. [pylance ReportMissingImport](https://blog.csdn.net/qq_44486439/article/details/107177385)  
  2. 莫名其妙的下划波浪线，extensions setting --> Diagnostic Severity Overrides --> Add Item --> Value --> None  
  
         
## Ubuntu Wireless network card
https://github.com/aircrack-ng/rtl8812au/tree/v5.3.4
