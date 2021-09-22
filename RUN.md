
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




