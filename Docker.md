# Docker with PyCharm
## Install Docker
Refer to https://yeasy.gitbook.io/docker_practice/install/ubuntu
## Install Nvidia-Docker
Refer to https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-docker-ce
## Pull a PyTorch Docker Image
Refer to https://github.com/anibali/docker-pytorch
## Create a Container
Refer to https://zhuanlan.zhihu.com/p/52827335  
Refer to https://github.com/anibali/docker-pytorch  

Refer to https://www.cnblogs.com/defineconst/p/10035529.html

Note that: -it terminal, --gpus=0(all), -p port mapping, -v space mapping, --ipc=host multiprocessing.   

*run --rm* will shutdown the container after exit it.

docker exec -it {container ID} /bin/bash is better 

Recommend at least 3 ports, 8097 for visdom, 22 for ssh.

## SSH
Refer to https://blog.csdn.net/qq_27068845/article/details/77015432  
sudo is needed

1. Install ssh and vim  
```
apt-get update
apt-get install -y openssh-server
apt-get install vim
```
2. Change root password  
``` passwd root```  
3. setup  
```
vim /etc/ssh/sshd_config
```
Annotate
```
PermitRootLogin prohibit-password.
```
Add
```
PermitRootLogin yes
```
4. ```/etc/init.d/ssh restart```
5. Test
  In ssh clinet,  
```
ssh root@127.0.0.1 - 8022
```
## PyCharm connects the remote server by ssh
In PyCharm, Tools --> Development --> configuration --> + --> SFTP --> setup Connection and Mappings;  
SSH Interperter: File --> Setting --> Project:Project --> Python Interperter --> Python Interperter --> add --> SSH Interperter.

## "Python" referring
Refer to https://blog.csdn.net/chen1234520nnn/article/details/102658300  
Python path  
```
which python
```
"Python" referring  
```
sudo rm /usr/bin/python
sudo ln -s {python interperter path} /usr/bin/python
PATH=/usr/bin:$PATH
```
## SSH under win10
Refer to https://www.jianshu.com/p/5af4ed62a752  
Right click "windows bottom" --> Windows PowerShell --> root@172.0.0.1 -p 8022

## CONDA for root

Refer to http://www.siyuanblog.com/?p=31126

```
CUDA_VISIBLE_DEVICES=0 python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml 
```