# Notebook
# 
# 重装系统记录
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

