#!/usr/bin/env python
# coding: utf-8

# # Using DALI in PyTorch Lightning
# 
# ### Overview
# 
# This example shows how to use DALI in PyTorch Lightning.
# 
# Let us grab [a toy example](https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html) showcasing a classification network and see how DALI can accelerate it.
# 
# The DALI_EXTRA_PATH environment variable should point to a [DALI extra](https://github.com/NVIDIA/DALI_extra) copy. Please make sure that the proper release tag, the one associated with your DALI version, is checked out.

# In[ ]:


import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os

# workaround for https://github.com/pytorch/vision/issues/1938 - error 403 when downloading mnist dataset
import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


# We will start by implement a training class that uses the native data loader

# In[ ]:


global args  
if __name__ == '__main__':
    from configargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=0, env_var="MNIST_GPU")
    parser.add_argument('--lr', type=float, default=1e-3, env_var="MNIST_LR")
    parser.add_argument('--batch_size', type=int, default=64, env_var="MNIST_BATCH_SIZE")
    parser.add_argument('--max_epochs', type=int, default=1, env_var="MNIST_MAX_EPOCHS")
    parser.add_argument('--data_dir', type=str, default=os.getcwd(), env_var="MNIST_DATA_DIR")
    parser.add_argument('--dali_data_dir', type=str, default=os.getcwd(), env_var="DALI_EXTRA_PATH")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--distributed_backend', type=str, default=None, help="use to enable ddp")  # ddp

    if '__file__' in globals():
        print(__file__)
        args = parser.parse_args()    # process the arg 
    else:
        args = parser.parse_args("")  # take defaults in Jupyter    

print(args) 


# In[23]:


class LitMNIST(LightningModule):

  def __init__(self):
    super().__init__()

    # mnist images are (1, 28, 28) (channels, width, height)
    self.layer_1 = torch.nn.Linear(28 * 28, 128)
    self.layer_2 = torch.nn.Linear(128, 256)
    self.layer_3 = torch.nn.Linear(256, 10)

  def forward(self, x):
    batch_size, channels, width, height = x.size()

    # (b, 1, 28, 28) -> (b, 1*28*28)
    x = x.view(batch_size, -1)
    x = self.layer_1(x)
    x = F.relu(x)
    x = self.layer_2(x)
    x = F.relu(x)
    x = self.layer_3(x)

    x = F.log_softmax(x, dim=1)
    return x

  def process_batch(self, batch):
      return batch

  def training_step(self, batch, batch_idx):
      x, y = self.process_batch(batch)
      logits = self(x)
      loss = F.nll_loss(logits, y)
      return loss

  def cross_entropy_loss(self, logits, labels):
      return F.nll_loss(logits, labels)

  def configure_optimizers(self):
      return Adam(self.parameters(), lr=1e-3)

  def prepare_data(self):
      # download data only
      self.mnist_train = MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    
  def setup(self, stage=None):
      # transforms for images
      transform=transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])
      self.mnist_train = MNIST(args.data_dir, train=True, download=False, transform=transform)

  def train_dataloader(self):
       return DataLoader(self.mnist_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)


# And see how it works with CPU

# In[31]:


from pytorch_lightning.loggers import TensorBoardLogger
model = LitMNIST()
logger = TensorBoardLogger("lightning_logs", name="pl_cpu_mnist")
trainer = Trainer(max_epochs=args.max_epochs, gpus=0, logger=logger)
# ddp work only in no-interactive mode, to test it unncoment and run as a script
# trainer = Trainer(gpus=8, distributed_backend=args.distributed_backend, max_epochs=args.max_epochs)
## MNIST data set is not always available to download due to network issues
## to run this part of example either uncomment below line
trainer.fit(model)


# And see how it works with GPU

# In[25]:


from pytorch_lightning.loggers import TensorBoardLogger
model = LitMNIST()
logger = TensorBoardLogger("lightning_logs", name="pl_gpu_mnist")
trainer = Trainer(max_epochs=args.max_epochs, gpus=1, logger=logger)
# ddp work only in no-interactive mode, to test it unncoment and run as a script
# trainer = Trainer(gpus=8, distributed_backend=args.distributed_backend, max_epochs=args.max_epochs, logger=logger)
## MNIST data set is not always available to download due to network issues
## to run this part of example either uncomment below line
trainer.fit(model)


# The next step is to define a DALI pipeline that will be used for loading and pre-processing data.

# In[26]:


import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

# Path to MNIST dataset
data_path = os.path.join(args.dali_data_dir, 'db/MNIST/training/')

@pipeline_def
def GetMnistPipeline(device, shard_id=0, num_shards=1):
    jpegs, labels = fn.readers.caffe2(path=data_path, shard_id=shard_id, num_shards=num_shards, random_shuffle=True, name="Reader")
    images = fn.decoders.image(jpegs,
                               device='mixed' if device == 'gpu' else 'cpu',
                               output_type=types.GRAY)
    images = fn.crop_mirror_normalize(images,
                                      dtype=types.FLOAT,
                                      std=[0.3081 * 255],
                                      mean=[0.1307 * 255],
                                      output_layout="CHW")
    if device == "gpu":
        labels = labels.gpu()
    # PyTorch expects labels as INT64
    labels = fn.cast(labels, dtype=types.INT64)
    return images, labels


# Now we are ready to modify the training class to use the DALI pipeline we have just defined. Because we want to integrate with PyTorch, we wrap our pipeline with a PyTorch DALI iterator, that can replace the native data loader with some minor changes in the code. The DALI iterator returns a list of dictionaries, where each element in the list corresponds to a pipeline instance, and the entries in the dictionary map to the outputs of the pipeline. For more information, check the documentation of DALIGenericIterator.

# In[27]:


class DALILitMNIST(LitMNIST):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
      # no preparation is needed in DALI
      pass

    def setup(self, stage=None):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size
        mnist_pipeline = GetMnistPipeline(batch_size=args.batch_size, device='gpu', device_id=device_id, shard_id=shard_id,
                                       num_shards=num_shards, num_threads=args.num_workers)
        self.train_loader = DALIClassificationIterator(mnist_pipeline, reader_name="Reader",
                                                       last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)                          
    def train_dataloader(self):
        return self.train_loader
    
    def process_batch(self, batch):
        x = batch[0]["data"]
        y = batch[0]["label"].squeeze(-1)
        return (x, y)


# We can now run the training

# In[28]:


# Even if previous Trainer finished his work it still keeps the GPU booked, force it to release the device.
if 'PL_TRAINER_GPUS' in os.environ:
    os.environ.pop('PL_TRAINER_GPUS')
model = DALILitMNIST()
logger = TensorBoardLogger("lightning_logs", name="pl_dali_mnist")
trainer = Trainer(max_epochs=args.max_epochs, gpus=1, logger=logger)
# ddp work only in no-interactive mode, to test it unncoment and run as a script
# trainer = Trainer(gpus=8, distributed_backend=args.distributed_backend, max_epochs=args.max_epochs, logger=logger)
trainer.fit(model)


# For even better integration, we can provide a custom DALI iterator wrapper so that no extra processing is required inside `LitMNIST.process_batch`. Also, PyTorch can learn the size of the dataset this way.

# In[29]:


class BetterDALILitMNIST(LitMNIST):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
      # no preparation is needed in DALI
      pass

    def setup(self, stage=None):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size
        mnist_pipeline = GetMnistPipeline(batch_size=args.batch_size, device='gpu', device_id=device_id, shard_id=shard_id, num_shards=num_shards, num_threads=args.num_workers)

        class LightningWrapper(DALIClassificationIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)
    
            def __next__(self):
                out = super().__next__()
                # DDP is used so only one pipeline per process
                # also we need to transform dict returned by DALIClassificationIterator to iterable
                # and squeeze the lables
                out = out[0]
                return [out[k] if k != "label" else torch.squeeze(out[k]) for k in self.output_map]

        self.train_loader = LightningWrapper(mnist_pipeline, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)

    def train_dataloader(self):
        return self.train_loader


# Let us run the training one more time

# In[30]:


# Even if previous Trainer finished his work it still keeps the GPU booked, force it to release the device.
if 'PL_TRAINER_GPUS' in os.environ:
    os.environ.pop('PL_TRAINER_GPUS')
model = BetterDALILitMNIST()
logger = TensorBoardLogger("lightning_logs", name="pl_dali_iterator_mnist")
trainer = Trainer(max_epochs=args.max_epochs, gpus=1, logger=logger)
# ddp work only in no-interactive mode, to test it unncoment and run as a script
# trainer = Trainer(gpus=8, distributed_backend=args.distributed_backend, max_epochs=args.max_epochs, logger=logger)
trainer.fit(model)


# In[ ]:




