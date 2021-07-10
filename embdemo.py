import os

import json
import os
import random
import argparse
from argparse import Namespace
import numpy as np
import glob
import gzip

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch.distributed as dist

from longformer.longformer import Longformer, LongformerConfig,LongformerEmbedding

# model=LongformerEmbedding('schen/longformer-chinese-base-4096',attention_mode='n2')

model=LongformerEmbedding('schen/longformer-chinese-base-4096',layers=[1,7],cls=False)
# model
# pass


# inputs = model.tokenizer("Hello, my dog is cute", return_tensors="pt",padding="max_length",truncation=True,max_length=model.tokenizer.model_max_length)
inputs = model.tokenizer("Hello, my dog is cute", return_tensors="pt",padding="max_length",truncation=True,max_length=400)
# ["token_type_ids"]
outputs = model(**inputs)
print("outputs",outputs)

print("outputs",outputs.keys())

print("outputs",outputs['logits'].size())
# model.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

# print(dir(model))
print(model)
torch.save(model.state_dict(),"model.bin")