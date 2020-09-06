# Synergy  
Synergy make 2 PCs share one keyboard and one mouse.  
## 1. install and setup
For windows, I store the package in my mobile HDD.And Digital key is required.  

For Ubuntu 18, it is key-free.  
```
sudo apt-get install synergy
```

For detail, please refer to https://v.qq.com/x/page/r03967n7ar4.html.  

## 2. Bug.
https://blog.csdn.net/pblearning/article/details/101278721  

## 3. Auto-startup
1. ``` gnome-session-properties```
2. add /usr/bin/synergyc -f --name <client-name> <server ip>  

## 4. Automatic login
Settings --> Details --> Users --> Unlock (on the up-right) --> Automatic Login (On).  
