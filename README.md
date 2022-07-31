
Grid.ai Session example running NVIDIA [DALI](https://github.com/NVIDIA/DALI).  Other Grid.ai NVIDIA examples:
- Grid.ai NVIDIA [DALI](https://github.com/robert-s-lee/grid-nvidia-dali))
- Grid.ai NVIDIA [fsi-samples](https://github.com/robert-s-lee/grid-nvidia-fsi-samples)

This guide talks thru NVIDIA examples [getting_started.html](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting_started.html) and [video_reader_simple_example.ipynb](https://github.com/NVIDIA/DALI/blob/main/docs/examples/sequence_processing/video/video_reader_simple_example.ipynb).  

# Prerequisite
## Setup Grid.ai 
- [Create an Grid.ai account](https://docs.grid.ai/getting-started/getting-started-with-grid#login-steps) with Grid.ai
- [Create a Grid.ai Session using Product Tour](https://docs.grid.ai/features/sessions#product-tour) 
- [Delete the test Session](https://docs.grid.ai/features/sessions#delete-a-session)

## Create Grid.ai Session with GPU
- Create a Grid.ai Session with g4dn.xlarge GPU
- [Start Jupyter Notebook](https://docs.grid.ai/features/sessions/jupyterlab-with-sessions). 

## Get Bash Access 
To run `bash` commands in this example, one of the following methods can be used: 
- From Grid.ai Jupyter Notebook: `File` -> `New` -> `Terminal`
- From Terminal: `grid session ssh`
- From VS Code: `Command P` -> `Remote SSH: Connect to Host`

## ssh into Grid.is session
  
```bash
grid ssh-keys add lit_key ~/.ssh/id_ed25519.pub
grid session ssh g4dn.xlarge 
tmux # 
```

- setup Conda environment
```bash
conda create --yes --name dali python=3.8
conda activate dali # note you may get prompt to run `conda init bash && exit`
pip install ipykernel # allow usage with Jupyter notebook
python -m ipykernel install --user --name=dali # show conda env in Jupyter notebook
ipython profile create
```
- From Grid.ai Jupyter Notebook: `Kernel` -> `Restart Kernel...`

# Check for CUDA library 
Run the following `bash` command(s)
```bash
ldconfig -p | grep libnvcuvid.so
# make sure ldconfig returns correctly with this
#        libnvcuvid.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1
exit
```

# Download Tools 
Run the following `bash` command(s)

- install [`git-lfs`](https://git-lfs.github.com). Note: Git LFS is used there to download large videos from the used in `DALI_extra`. If git-lfs isn’t installed, then git cloned repository will be missing those files.

```bash
# Instructions from 
#   https://github.com/git-lfs/git-lfs/wiki/Installation#ubuntu and 
#   https://github.com/git-lfs/git-lfs/issues/3964#issuecomment-570586798
sudo apt-get install git-lfs
git lfs install --skip-repo
```

- install python tools and examples
```bash
python -m install pip --upgrade && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
# DALI examples
python -m pip install numpy
git clone https://github.com/NVIDIA/DALI
git clone https://github.com/NVIDIA/DALI_extra.git
```

- set DALI_EXTRA_PATH for python and Jupyter notebook scripts that need them

```bash
# for python scripts
cat ~/.bashrc
cat >> ~/.bashrc <<EOF
export DALI_EXTRA_PATH="/home/jovyan/DALI_extra"
EOF
chmod a+x ~/.bashrc

# for notebooks
ipython profile create
cat > ~/.ipython/profile_default/startup/00-dali.py <<EOF
import os
os.environ['DALI_EXTRA_PATH'] = "/home/jovyan/DALI_extra"
EOF
# to see the list scripts that need DALI_EXTRA_PATH
cd DALI
grep -R --include *.ipynb --include *.py DALI_EXTRA ~/DALI
```

# Jupyter Notebook: Run getting_started.ipynb
Run the following from Jupyter Notebook UI.

- Click on `File` -> `Open From path` -> `/DALI/docs/examples/getting_started.ipynb`
- Click on `Run` -> `Run All Cells`

![output](images/getting_started_image.png) 

# Jupyter Notebook: Run getting_started.ipynb
Run the following from Jupyter Notebook UI.

- Click on `File` -> `Open From path` -> `/DALI/docs/examples/sequence_processing/video/video_reader_simple_example.ipynb`

- Click on `Run` -> `Run All Cells`

![output](images/Screen%20Shot%202021-08-19%20at%2010.14.17%20AM.png)

# Instructions on Starting Grid.ai Session and Jupyter Notebook


## Start a Grid.ai `g4dn.xlarge — 1 x T4` session

- Go to [https://grid.ai](https://grid.ai)
- Click on ![Sign In](images/signin.png)
- Click on ![new](images/new.png) 
- Click on ![session](images/session.png) 
- Click on ![g4dn.xlarge](images/new_session.png)

## Start a Jupyter notebook

- Click ![Sessions](images/sessions.png)
- Click on ![Jupyter icon](images/Screen%20Shot%202021-08-16%20at%2011.16.34%20AM.png).

time python pytorch-lightning-dali-mnist.py --mode gpu_dali_better --gpus 1 --batch_size 128 --num_workers 1
real    0m42.354s
time python pytorch-lightning-dali-mnist.py --mode gpu_dali --batch_size 128 --num_workers 1
real    0m56.414s
time python pytorch-lightning-dali-mnist.py --mode gpu_dali --gpus 1 --batch_size 128 --num_workers 1
real    0m43.863s
time python pytorch-lightning-dali-mnist.py --mode cpu --batch_size 128 --num_workers 1
real    2m28.856s
time python pytorch-lightning-dali-mnist.py --mode gpu --gpus 1 --batch_size 128 --num_workers 1
real    2m21.803s


time python pytorch-lightning-dali-mnist.py --mode gpu_dali_better --gpus 1 --batch_size 128 --num_workers 2
real    0m48.014s

time python pytorch-lightning-dali-mnist.py --mode gpu_dali_better --gpus 1 --batch_size 128 --num_workers 3
real    0m46.613s

time python pytorch-lightning-dali-mnist.py --mode gpu_dali_better --gpus 1 --batch_size 128 --num_workers 4
real    0m47.048s