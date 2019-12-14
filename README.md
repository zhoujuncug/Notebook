# Notebook
# 
# 重装系统记录
## 系统
1. 安装双系统时，应该先安装win，然后ubuntu
2. 安装win后，如果想把ubuntu安装到E盘（比如）那么删除E盘，其他盘要激活
3. 安装ubuntu时，选择 ubuntu alongside with win (这时，ubuntu会默认安装到第一个未分配的盘)

## CUDA
1. 显卡驱动
   1.1 sudo add-apt-repository ppa:graphics-drivers/ppa
   1.2 sudo apt-get update
   1.3 ubuntu-drivers devices
       这里记录一下recommend 的driver 型号，我的是nvidia-driver-430
   1.4 sudo apt install nvidia-driver-440
   1.5 重启
   1.6 $ nvidia-smi
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
       sudo make
       sudo ./deviceQuery
       如果有 Result = PASS 就OK 不成功就重启一下试试
