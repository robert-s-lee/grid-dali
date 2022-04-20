Grid.ai Session and Run examples of Nvidia [DALI PyTorch Lightning MNIST](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-lightning.html)


- create datastore that has MNIST + DALI MNIST data in a single place
```bash
# start a grid session
grid session create --instance_type g4dn.xlarge --name g4dn-xlarge-1
grid ssh-keys add lit_key ~/.ssh/id_ed25519.pub; grid session ssh g4dn-xlarge-1

# download DALI_extra that uses git-lfs
sudo apt-get install git-lfs
git lfs install --skip-repo
git clone https://github.com/NVIDIA/DALI_extra.git

# setup conda
conda create --yes --name dali python=3.8
conda activate dali

# download the code
git clone https://github.com/robert-s-lee/grid-dali
cd grid-dali
pip install -r requirements.txt 

# run to save the MNIST data inside DALI_extra
# ~ does not work with DALI library
python pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=$HOME/DALI_extra --dali_data_dir=$HOME/DALI_extra

python pytorch-lightning-dali-mnist.py --gpus=1 


# hack TODO: figure out why this is required in Run but not in session 
# SESSION and RUN library mismatch ?? clues
# https://github.com/PyTorchLightning/pytorch-lightning/pull/986/files 
mkdir ~/DALI_extra/MNIST/processed  

# save the datastore that has DALI_extra + MNIST
grid datastore create --source ~/DALI_extra --name dali-mnist
watch grid datastore status --name dali-mnist

# pause this session
grid session pause g4dn-xlarge-1
```

- test on a new session by mounting the datastore 
```bash
# start a session with datastore from the previous step
grid session create --instance_type g4dn.xlarge --datastore_name dali-mnist --datastore_version 2 --name g4dn-xlarge-2
grid ssh g4dn-xlarge-2

# verify datastore is mounted
ls /home/jovyan/dali-mnist

# download the code
git clone https://github.com/robert-s-lee/grid-dali
cd grid-dali
pip install -r requirements.txt 

# use absolute path as ~/dali-mnist does not work.  
python pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=$HOME/dali-mnist --dali_data_dir=$HOME/dali-mnist
python pytorch-lightning-dali-mnist.py --gpus=1 --dali_data_dir=$HOME/dali-mnist
```

- run experiments
```bash
export name=dali-$(date '+%y%m%d-%H%M%S'); grid run --name=$name --gpus=1 --instance_type=g4dn.xlarge --dependency_file requirements.txt pytorch-lightning-dali-mnist.py --gpus=1 

# fails with datastore error (v2 datastore) not present on session 
export name=dali-$(date '+%y%m%d-%H%M%S'); grid run --name=$name --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --dali_data_dir=grid:dali-mnist:2

export name=dali-$(date '+%y%m%d-%H%M%S'); grid run --name ${name} --gpus 1 --instance_type g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus 1 --dali_data_dir grid:dali-mnist:2


# (v1 datastore)
grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=grid:dali-mnist:2 
```
|          | datatore v1 (hello-mnist)|  datastore v2 (dali-mnist) | 
| session  |       works              |        works
| run      |       works              |  [Errno 30] Read-only file system: '/datastores/dali-mnist/MNIST/processed'

hello-mnist creates processed.  

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

grid run --instance_type g4dn.xlarge --datastore_name dali-mnist --datastore_version 2 

`grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=grid:dali-mnist:1 --dali_data_dir=grid:dali-mnist:1`
there is no log

## in Session - warnings
- on CPU run
```log
/home/jovyan/conda/lib/python3.8/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
```

- MNIST/process is required in run is not required in session (for read only)


- on GPU run
```log
Epoch 0:   0%|                                                                                                                    | 0/938 [00:00<?, ?it/s][W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
```