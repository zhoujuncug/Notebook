# Notebook
# 
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
     make install
     # Alternatively, if you do not have permissions or prefer
     # not to install the COCO API into global site-packages
     python3 setup.py install --user
   3. coco api
     
     from pycocotools.coco import COCO
     coco = COCO('coco_annotation_file')
     coco.getCatIds() # 返回annotation文件中类型的索引，比如：人，猫，滑板
     coco.loadCats([1]) # 返回索引1中类型包含的信息
     coco.getImgIds() # 返回图像的索引
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
   1.1 ```sudo add-apt-repository ppa:graphics-drivers/ppa```  
   1.2 ```sudo apt-get update```  
   1.3 ```ubuntu-drivers devices```  
       这里记录一下recommend 的driver 型号，我的是nvidia-driver-430  
   1.4 ```sudo apt install nvidia-driver-440   
       default option if need.
   1.5 重启  
   1.6 ```nvidia-smi  
       有信息显示就OK  
2. cuda  
   2.1 https://developer.nvidia.com/cuda-toolkit-archive  
       Operating: System Linux  
       Architecture: x86_64  
       Distribution: Ubuntu  
       Version: 18.04  
       Installer Type: deb(local)  
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
   
## 火狐浏览器
   1. 链接不安全
      搜索框输入 ```about:config```
      选 我了解此风险
      输入 ```security.enterprise_roots.enabled```
      切换为 TRUE
