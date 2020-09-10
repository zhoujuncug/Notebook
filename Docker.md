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
Note that: -it terminal, --gpus=0(all), -p port mapping, -v space mapping, --ipc=host multiprocessing. 
Recommend at least 3 ports, 8097 for visdom, 22 for ssh.
## SSH
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
4. Test
In ssh clinet,  
```
ssh root@127.0.0.1 - 8022
```
## PyCharm connects the remote server by ssh
In PyCharm, Tools --> Development --> configuration --> + --> SFTP --> setup Connection and Mappings;
SSH Interperter: File --> Setting --> Project:Project --> Python Interperter --> Python Interperter --> add --> SSH Interperter.

## "Python" referring
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
Right click "windows bottom" --> Windows PowerShell --> root@172.0.0.1 -p 8022
