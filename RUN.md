Grid.ai Session and Run examples of Nvidia (DALI Pytorch Lightning MNIST)[https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/pytorch-lightning.html]

- create datastore that has MNIST + DALI MNIST in a single place for HPO
```bash
sudo apt-get install git-lfs
git lfs install --skip-repo
git clone https://github.com/NVIDIA/DALI_extra.git
python pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=~/DALI_extra --dali_data_dir=~/DALI_extra
grid datastore create --source ~/DALI_extra --name dali-mnist
```

- test on a new session by mounting the datastore 
```bash
grid session create --instance_type g4dn.xlarge --datastore_name dali-mnist 
grid ssh xxx
# verify datastore is mounted
ls /home/jovyan/dali-mnist
git clone https://github.com/robert-s-lee/grid-dali
cd grid-dali
git checkout mnist-run
pip install -r requirements.txt 
# ~/dali-mnist does not work.  use absolute path
python pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=/home/jovyan/dali-mnist --dali_data_dir=/home/jovyan/dali-mnist
python pytorch-lightning-dali-mnist.py --gpus=1 --dali_data_dir=/home/jovyan/dali-mnist
```

- run experiments
```bash
grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=grid:dali-mnist:1 --dali_data_dir=grid:dali-mnist:1
grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --dali_data_dir=grid:dali-mnist:1

```

# TODO Fix in Session

- on CPU run
```log
/home/jovyan/conda/lib/python3.8/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
```

- on GPU run
```log
Epoch 0:   0%|                                                                                                                    | 0/938 [00:00<?, ?it/s][W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
```