
pip install nvidia-pyindex
pip install nvidia-dali-cuda110

download LFS from git with CURL
git-lfs-fetch qq
https://gist.github.com/fkraeutli/66fa741d9a8c2a6a238a01d17ed0edc5
https://github.com/liberapay/git-lfs-fetch.py

git lfs fetch https://github.com/NVIDIA/DALI_extra/tree/main/db/MNIST

curl -O https://raw.githubusercontent.com/NVIDIA/DALI/main/docs/examples/frameworks/pytorch/pytorch-lightning.ipynb

ghclone https://github.com/NVIDIA/DALI_extra/tree/main/db/MNIST
mv MNIST dali-MNIST
git lfs track "*.mdb"

pip install jupyterlab
jupyter njupyter nbconvert --to script pytorch-lightning-dali-mnist.ipynb



on g4dn.xlarge

| 938/938 [00:09<00:00, 94.34it/s, loss=0.117, v_num=2]
 938/938 [00:06<00:00, 147.76it/s, loss=0.132, v_num=2]
 | 938/938 [00:04<00:00, 227.39it/s, loss=0.132, v_num=0]
 | 938/938 [00:04<00:00, 228.44it/s, loss=0.125, v_num=0]


# works
python pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=/home/jovyan/dali-mnist --dali_data_dir=/home/jovyan/dali-mnist

# testing
grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --dali_data_dir=grid:dali-mnist:1

grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=grid:dali-mnist:1 --dali_data_dir=grid:dali-mnist:1

grid run --gpus=1 --instance_type=g4dn.xlarge pytorch-lightning-dali-mnist.py --gpus=1 --data_dir=grid:dali-mnist:1 --dali_data_dir=grid:dali-mnist:1
