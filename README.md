
Grid.ai Session example of running [getting_started.html](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting_started.html) 

- [Start a Grid.ai `g4dn.xlarge — 1 x T4` session](#start-a-gridai-g4dnxlarge--1-x-t4-session)
- [Start a Jupyter notebook](#start-a-jupyter-notebook)
- [Jupyter Notebook: Initialize conda](#jupyter-notebook-initialize-conda)
- [Jupyter Notebook: Download Tools](#jupyter-notebook-download-tools)
- [Jupyter Notebook: Run getting_started.ipynb](#jupyter-notebook-run-getting_startedipynb)

# Start a Grid.ai `g4dn.xlarge — 1 x T4` session

- Go to [https://grid.ai](https://grid.ai)
- Click on ![Sign In](images/signin.png)
- Click on ![new](images/new.png) 
- Click on ![session](images/session.png) 
- Click on ![g4dn.xlarge](images/new_session.png)

# Start a Jupyter notebook

- Click ![Sessions](images/sessions.png)
- Click on ![Jupyter icon](images/Screen%20Shot%202021-08-16%20at%2011.16.34%20AM.png).

# Jupyter Notebook: Initialize conda 
`File` -> `New` -> `Terminal`
```bash
conda init bash
exit
```

# Jupyter Notebook: Download Tools
`File` -> `New` -> `Terminal`
```bash
pip3 install pip --upgrade && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
git clone https://github.com/NVIDIA/DALI
```

# Jupyter Notebook: Run getting_started.ipynb
`File` -> `Open From path` -> `/home/jovyan/DALI/docs/examples/getting_started.ipynb`
`Run` -> `Run All Cells`

![output](images/getting_started_image.png) 