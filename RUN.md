Grid.ai Session and Run examples of Nvidia [DALI PyTorch Lightning MNIST](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-lightning.html)


- create datastore that has MNIST + DALI MNIST data in a single place
```bash
# start a grid session
grid session create --instance_type g4dn.xlarge --name g4dn-xlarge-1
grid ssh g4dn-xlarge-1

# download DALI_extra that uses git-lfs
sudo apt-get install git-lfs
git lfs install --skip-repo
git clone https://github.com/NVIDIA/DALI_extra.git

# download the code
git clone https://github.com/robert-s-lee/grid-dali
cd grid-dali
pip install -r requirements.txt 

# run to save the MNIST data inside DALI_extra
python pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=~/DALI_extra --dali_data_dir=~/DALI_extra

# save the datastore that has DALI_extra + MNIST
grid datastore create --source ~/DALI_extra --name dali-mnist

# pause this session
grid session pause g4dn-xlarge-1
```

- test on a new session by mounting the datastore 
```bash
# start a session with datastore from the previous step
grid session create --instance_type g4dn.xlarge --datastore_name dali-mnist --name g4dn-xlarge-2
grid ssh g4dn-xlarge-2

# verify datastore is mounted
ls /home/jovyan/dali-mnist

# download the code
git clone https://github.com/robert-s-lee/grid-dali
cd grid-dali
pip install -r requirements.txt 

# use absolute path as ~/dali-mnist does not work.  
python pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=/home/jovyan/dali-mnist --dali_data_dir=/home/jovyan/dali-mnist
python pytorch-lightning-dali-mnist.py --gpus=1 --dali_data_dir=/home/jovyan/dali-mnist
```

- run experiments
```bash
grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=grid:dali-mnist:1 --dali_data_dir=grid:dali-mnist:1

# fails with datastore error (v2 datastore) not present on session 
grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --dali_data_dir=grid:dali-mnist:1

# (v1 datastore)
grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=grid:hello-mnist:1 

```

# TODO Fix 

## in run - fatal

`grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --dali_data_dir=grid:dali-mnist:1`
```logs
[experiment] [2021-09-23T16:53:54.173479+00:00] OSError: [Errno 30] Read-only file system: '/datastores/dali-mnist/MNIST/processed'
```

- running GPU with DALI
```log
[experiment] [2021-09-23T16:10:14.670620+00:00] RuntimeError: Tensor for 'out' is on CPU, Tensor for argument #1 'self' is on CPU, but expected them to be on GPU (while checking arguments for addmm)
```

`grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=grid:dali-mnist:1 --dali_data_dir=grid:dali-mnist:1`
there is no log

## in Session - warnings
- on CPU run
```log
/home/jovyan/conda/lib/python3.8/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
```

- on GPU run
```log
Epoch 0:   0%|                                                                                                                    | 0/938 [00:00<?, ?it/s][W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
```