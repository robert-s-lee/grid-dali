
Grid.ai Session example of running [getting_started.html](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting_started.html) 

- [Jupyter Notebook: Initialize conda](#jupyter-notebook-initialize-conda)
- [Jupyter Notebook: Download Tools](#jupyter-notebook-download-tools)
- [Jupyter Notebook: Run getting_started.ipynb](#jupyter-notebook-run-getting_startedipynb)
- [Jupyter Notebook: Run getting_started.ipynb](#jupyter-notebook-run-getting_startedipynb-1)
- [Start a Grid.ai `g4dn.xlarge — 1 x T4` session](#start-a-gridai-g4dnxlarge--1-x-t4-session)
- [Start a Jupyter notebook](#start-a-jupyter-notebook)
  
Login into [Grid.ai](#start-a-gridai-g4dnxlarge--1-x-t4-session), start a Session with GPU instance, and [Start Jupyter Notebook](#start-a-jupyter-notebook).

# Jupyter Notebook: Initialize conda 
- `File` -> `New` -> `Terminal`
```bash
conda init bash
ldconfig -p | grep libnvcuvid.so
# make sure ldconfig returns correctly with this
#        libnvcuvid.so.1 (libc6,x86-64) => /lib64/libnvcuvid.so.1
exit
```

# Jupyter Notebook: Download Tools
- `File` -> `New` -> `Terminal`
```bash
# Instructions from 
#   https://github.com/git-lfs/git-lfs/wiki/Installation#ubuntu and 
#   https://github.com/git-lfs/git-lfs/issues/3964#issuecomment-570586798
sudo apt-get install git-lfs
git lfs install --skip-repo
# python tools
pip3 install pip --upgrade && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
# DALI
git clone https://github.com/NVIDIA/DALI
git clone https://github.com/NVIDIA/DALI_extra.git
```

# Jupyter Notebook: Run getting_started.ipynb

- `File` -> `Open From path` -> `/DALI/docs/examples/getting_started.ipynb`
- `Run` -> `Run All Cells`

![output](images/getting_started_image.png) 

# Jupyter Notebook: Run getting_started.ipynb

- `File` -> `Open From path` -> `/DALI/docs/examples/sequence_processing/video/video_reader_simple_example.ipynb`

- DALI_EXTRA_PATH should be set

```bash
import os
import numpy as np
os.environ['DALI_EXTRA_PATH'] = "/home/jovyan/DALI_extra"
```

- `Run` -> `Run All Cells`

![output](images/Screen%20Shot%202021-08-19%20at%2010.14.17%20AM.png)

# Start a Grid.ai `g4dn.xlarge — 1 x T4` session

- Go to [https://grid.ai](https://grid.ai)
- Click on ![Sign In](images/signin.png)
- Click on ![new](images/new.png) 
- Click on ![session](images/session.png) 
- Click on ![g4dn.xlarge](images/new_session.png)

# Start a Jupyter notebook

- Click ![Sessions](images/sessions.png)
- Click on ![Jupyter icon](images/Screen%20Shot%202021-08-16%20at%2011.16.34%20AM.png).
