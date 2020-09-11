# How to install Anaconda
1. Browse https://www.anaconda.com/products/individual, and copy url of the Anaconda installer for linux.  
2. ``` wget <url_above> ```  
3. ```source ~/.bashrc ```
4. ``` conda list ```

### Change source

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/

# 设置搜索时显示通道地址
conda config --set show_channel_urls yes
```

### CommandNotFoundError

```
source activate
source deactivate
conda activate your_virtual_name
```

