

- create datastore that has MNIST + DALI MNIST in a single place for HPO
```bash
python pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=~/DALI_extra --dali_data_dir=~/DALI_extra
grid datastore create --source ~/DALI_extra --name dali-mnist
```

- create a new session and mount the datastore to verify
```bash
grid session create --instance_type g4dn.xlarge --datastore_name dali-mnist 
git clone https://github.com/robert-s-lee/grid-dali
cd grid-dali
git checkout mnist-run
pip install -r requirements.txt 
# ~/dali-mnist does not work.  use absolute path
python pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=/home/jovyan/dali-mnist --dali_data_dir=/home/jovyan/dali-mnist
```

- run experiments
THis results in MNIST download error 
```bash
grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --dali_data_dir=grid:dali-mnist:1
```

# results in MNIST write error
grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=grid:dali-mnist:1 --dali_data_dir=grid:dali-mnist:1




```
(dali) gridai@ixsession → python pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=/home/jovyan/dali-mnist --dali_data_dir=/home/jovyan/dali-mnist
GPU available: True, used: False
TPU available: None, using: 0 TPU cores
/home/jovyan/conda/envs/dali/lib/python3.8/site-packages/pytorch_lightning/utilities/distributed.py:50: UserWarning: GPU available but not used. Set the --gpus flag when calling the script.
  warnings.warn(*args, **kwargs)
/home/jovyan/conda/envs/dali/lib/python3.8/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:180.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

  | Name    | Type   | Params
-----------------------------------
0 | layer_1 | Linear | 100 K
1 | layer_2 | Linear | 33.0 K
2 | layer_3 | Linear | 2.6 K
-----------------------------------
136 K     Trainable params
0         Non-trainable params
136 K     Total params
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:10<00:00, 91.09it/s, loss=0.113, v_num=1]
GPU available: True, used: True
TPU available: None, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name    | Type   | Params
-----------------------------------
0 | layer_1 | Linear | 100 K
1 | layer_2 | Linear | 33.0 K
2 | layer_3 | Linear | 2.6 K
-----------------------------------
136 K     Trainable params
0         Non-trainable params
136 K     Total params
Epoch 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:05<00:00, 158.26it/s, loss=0.111, v_num=1]
GPU available: True, used: True
TPU available: None, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name    | Type   | Params
-----------------------------------
0 | layer_1 | Linear | 100 K
1 | layer_2 | Linear | 33.0 K
2 | layer_3 | Linear | 2.6 K
-----------------------------------
136 K     Trainable params
0         Non-trainable params
136 K     Total params
Epoch 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:03<00:00, 239.07it/s, loss=0.123, v_num=1]
GPU available: True, used: True
TPU available: None, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name    | Type   | Params
-----------------------------------
0 | layer_1 | Linear | 100 K
1 | layer_2 | Linear | 33.0 K
2 | layer_3 | Linear | 2.6 K
-----------------------------------
136 K     Trainable params
0         Non-trainable params
136 K     Total params
Epoch 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:03<00:00, 241.05it/s, loss=0.142, v_num=1]
(dali) gridai@ixsession → exit
```