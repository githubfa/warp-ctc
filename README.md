# PyTorch bindings for Warp-ctc

[![Build Status](https://travis-ci.org/SeanNaren/warp-ctc.svg?branch=pytorch_bindings)](https://travis-ci.org/SeanNaren/warp-ctc)

This is an extension onto the original repo found [here](https://github.com/baidu-research/warp-ctc).

## 安装方法
git clone https://github.com/SeanNaren/warp-ctc.git 或点击链接直接下载Zip
cd warp-ctc
git reset ac045b6
mkdir build; cd build
cmake ..
make

cd ../pytorch_binding
python setup.py install

以上成功之后 直接将 
warp-ctc/warp-ctc-cuda10.1/pytorch_binding/build/lib.linux-x86_64-3.7 下的 
warpctc_pytorch 文件夹
复制到~/anaconda3/lib/python3.7/site-packages 里面

（该文件夹里面有__init__.py 和 _warp_ctc.cpython-37m-x86_64-linux-gnu.so 两个文件）
（以上路径和文件名可能根据Python和系统环境有所差别。）
复制完成之后可以测试import

###以上成功之后添加以下2行到：~.bashrc  不用该操作
###export LD_LIBRARY_PATH=~/anaconda3/lib/python3.7/site-packages/warp-ctc/pytorch_binding:$LD_LIBRARY_PATH
###export WARP_CTC_PATH="~/anaconda3/lib/python3.7/site-packages/warp-ctc/build"

注意问题：
    一、如果当前的cuda是只是使用conda安装的cudatoolkit版本，则需再安装完整的cuda版本才能完成cmake编译gpu版本 否则只会编译cpu版本。
安装方法可以使用：
    1、到 https://developer.nvidia.com/cuda-toolkit-archive 自行下载 run安装包安装
    
    二、当前环境
        Python 3.7
        pytorch 1.3.1
        CUDA 10.1

从 git clone https://github.com/SeanNaren/warp-ctc.git 下来的编译代码比较旧 CUDA编译不过 需要按照下面内容进行修改：

https://github.com/SeanNaren/deepspeech.pytorch/issues/397
zhenglilei commented on 15 Oct
src/reduce.cu
Line 44 to : shuff = __shfl_down_sync(0xFFFFFFFF, x, offset);

include/contrib/moderngpu/include/device/intrinsics.cuh
Line 115 to : var = __shfl_up_sync(0xFFFFFFFF, var, delta, width);
Line 125 to : p.x = __shfl_up_sync(0xFFFFFFFF, p.x, delta, width);
Line 126 to : p.y = __shfl_up_sync(0xFFFFFFFF, p.y, delta, width);
Line 143 to : "shfl.up.sync.b32 r0|p, %1, %2, %3, %4;"
Line 158 to : "shfl.up.sync.b32 r0|p, %1, %2, %3, %4;"

works fine with CUDA 10.1

This is the correct solution by Oct. 2019.



## Installation

Install [PyTorch](https://github.com/pytorch/pytorch#installation) v0.4.

`WARP_CTC_PATH` should be set to the location of a built WarpCTC
(i.e. `libwarpctc.so`).  This defaults to `../build`, so from within a
new warp-ctc clone you could build WarpCTC like this:

```bash
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
```

Now install the bindings:
```bash
cd pytorch_binding
python setup.py install
```

If you try the above and get a dlopen error on OSX with anaconda3 (as recommended by pytorch):
```bash
cd ../pytorch_binding
python setup.py install
cd ../build
cp libwarpctc.dylib /Users/$WHOAMI/anaconda3/lib
```
This will resolve the library not loaded error. This can be easily modified to work with other python installs if needed.

Example to use the bindings below.

```python
import torch
from warpctc_pytorch import CTCLoss
ctc_loss = CTCLoss()
# expected shape of seqLength x batchSize x alphabet_size
probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
labels = torch.IntTensor([1, 2])
label_sizes = torch.IntTensor([2])
probs_sizes = torch.IntTensor([2])
probs.requires_grad_(True)  # tells autograd to compute gradients for probs
cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
cost.backward()
```

## Documentation

```
CTCLoss(size_average=False, length_average=False)
    # size_average (bool): normalize the loss by the batch size (default: False)
    # length_average (bool): normalize the loss by the total number of frames in the batch. If True, supersedes size_average (default: False)

forward(acts, labels, act_lens, label_lens)
    # acts: Tensor of (seqLength x batch x outputDim) containing output activations from network (before softmax)
    # labels: 1 dimensional Tensor containing all the targets of the batch in one large sequence
    # act_lens: Tensor of size (batch) containing size of each output sequence from the network
    # label_lens: Tensor of (batch) containing label length of each example
```
