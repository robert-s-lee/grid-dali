
pip install nvidia-pyindex
pip install nvidia-dali-cuda110

curl -O https://raw.githubusercontent.com/NVIDIA/DALI/main/docs/examples/frameworks/pytorch/pytorch-lightning.ipynb

ghclone https://github.com/NVIDIA/DALI_extra/tree/main/db/MNIST
mv MNIST dali-MNIST

pip install jupyterlab
jupyter nbconvert --to script pytorch-lightning.ipynb
