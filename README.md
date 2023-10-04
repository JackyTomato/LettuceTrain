# LettuceTrain

This is a repository used to develop and train deep learning models for a lettuce image dataset with PyTorch.

## Conda environment
In order to train models on GPUs, PyTorch should be installed with CUDA support. This requires you to install PyTorch in a specific way.
Here, we use Anaconda (or rather Mamba) to install PyTorch and other related packages such as torchvision.
To install PyTorch with CUDA support make sure you install PyTorch with a build that contains `cuda`.
You can specify the build you want when using `conda install` with `pytorch=<version>=<build>`.
Use `conda search pytorch -c pytorch` to obtain a list of PyTorch versions and builds in Anaconda.
Similarly, get a `torchvision` build that contains `cu`. Additionally, you should install `pytorch-cuda` alongside `pytorch` and `torchvision`.
For all the packages, make sure the CUDA versions match and that the CUDA version is supported by the machine.
Run `nvidia-smi` in the terminal to see the latest CUDA version that is supported by your machine.

As an example, here is my install command, for package builds made for Python 3.11 and CUDA 12.1:
```
mamba create -n pytorch_cuda12.1 pytorch=2.1.0=py3.11_cuda12.1_cudnn8.9.2_0 torchvision=0.16.0=py311_cu121 pytorch-cuda=12.1 -c pytorch -c nvidia
```
