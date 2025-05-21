"""
This Python script provides a CLI version of xCAPT5 training without
XGBoost.
"""

import os
import math
import argparse

import torch
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from xgboost import XGBClassifier
from tqdm import tqdm
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

from xCAPT5_utils import (get_T5_model, read_fasta, get_embeddings,
    save_embeddings, pad, leaky_relu, multi_cnn)


try:
   os.mkdir("protT5")
   os.mkdir("protT5/protT5_checkpoint")
   os.mkdir("protT5/sec_struct_checkpoint")
   os.mkdir("protT5/output")
except FileExistsError:
   print("\nDirectories have already been created.\n")


parser = argparse.ArgumentParser()

parser.add_argument("pairs_file", type=str)
parser.add_argument("fasta_file", type=str)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument(
   "--neptune_name",
   type=str,
   default="xCAPT5_training_without_XGBoost"
)
parser.add_argument("--MCAPST5_ckpt", type=str, default=None)
parser.add_argument("--learning_rate", default=1e-3)

args = parser.parse_args()


run = neptune.init_run(
    project="mlid/xCAPT5-fine-tuning",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOGNlNjZjOS1iNWJiLTQyNGEtOGQ5My1iZDhkM2JhOWZkZjMifQ==",
    name=args.neptune_name
)


# For some strange reason, Tensorflow is looking in the wrong
# directories for the CuDNN library (i.e. the libcudnn.so.8 file)
# Therefore, in order to enable GPU usage, the CuDNN library has to be
# manually loaded
tf.load_library(
  "/home/anter87/PPI_prediction/xCAPT5/xCAPT5_venv/lib/python3.8/site-"
  "packages/nvidia/cudnn/lib/libcudnn.so.8"
)


# Conveniently enough, xCAPT5 requires the same input as SENSE-PPI, i.e.
# a FASTA file encompassing all proteins as well as a TSV file listing
# all interaction pairs to investigate
# The required input format is also identical with that of SENSE-PPI,
# i.e. the FASTA file headers exclusively contain the protein ID and the
# TSV files comprises combinations of those protein IDs
pair_path = args.pairs_file
seq_path = args.fasta_file


# Define the embedding type
# Possible options include embeddings per residue (yields a Lx1024
# matrix per protein with L being the protein's length) as well as
# embeddings protein (yields a 1024-dimensional vector per protein,
# irrespective of its length)
per_residue = True 
per_residue_path = "./protT5/output/per_residue_embeddings.h5"

per_protein = False
per_protein_path = "./protT5/output/per_protein_embeddings.h5"

# For some strange reason to still be fathomed out, secondary structures
# are somehow involved in xCAPT5
sec_struct = False
sec_struct_path = "./protT5/output/ss3_preds.fasta"

if per_residue:
   embedding_path = per_residue_path
elif per_protein:
   embedding_path = per_protein_path

assert (
    per_protein is True or per_residue is True or sec_struct is True
), print(
     "Minimally, you need to active per_residue, per_protein or "
     "sec_struct. (or any combination)"
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.")


if os.path.isfile(per_residue_path) == False:
    # Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
    model, tokenizer = get_T5_model(device=device)

    # Load example fasta.
    seqs = read_fasta( seq_path )

    for id, seq in seqs.items():
        if len(seq) > 1200:
            seqs[id] = seq[:1200]


    # Compute embeddings and/or secondary structure predictions
    results = get_embeddings(
        model, tokenizer, seqs, per_residue, per_protein, sec_struct,
        device
    )

    # Store per-residue embeddings
    if per_residue:
      save_embeddings(results["residue_embs"], per_residue_path)
    if per_protein:
      save_embeddings(results["protein_embs"], per_protein_path)
else:
    print("Already have the embedding file")


### Setting RAM GPU for training growth 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Disables caching (when set to 1) or enables caching (when set to 0) for just-in-time-compilation. When disabled,
# no binary code is added to or retrieved from the cache.
os.environ['CUDA_CACHE_DISABLE'] = '0' # orig is 0

# When set to 1, forces the device driver to ignore any binary code embedded in an application 
# (see Application Compatibility) and to just-in-time compile embedded PTX code instead.
# If a kernel does not have embedded PTX code, it will fail to load. This environment variable can be used to
# validate that PTX code is embedded in an application and that its just-in-time compilation works as expected to guarantee application 
# forward compatibility with future architectures.
os.environ['CUDA_FORCE_PTX_JIT'] = '1'# no orig


os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT']='1'

os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
os.environ['TF_DISABLE_NVTX_RANGES'] = '1'
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"



# =================================================
mixed_precision.set_global_policy('mixed_float16')


## Set constant hyperparameters
BATCH_SIZE = 64
seq_size = 1200
dim = 1024


# Load embeddings
print("Load the embedding file")
embedding_matrix= h5py.File(per_residue_path, 'r')
protein_keys = list(embedding_matrix.keys())
embedding_dict = dict()

for key in tqdm(protein_keys):
  embedding_dict[key] = np.array(embedding_matrix[key])


# Read interaction data set to DataFrame
print("Load the pair dataset file")
pair_dataframe = pd.read_csv(pair_path, sep='\t', header=None)
pair_array  = pair_dataframe.to_numpy()
np.random.seed(42)
np.random.shuffle(pair_array)
pair_dataframe = pd.DataFrame(pair_array)
pair_dataframe = pd.DataFrame(pair_array, columns = ['p1', 'p2', 'label'])
pair_dataframe['label'] = pair_dataframe['label'].astype('float16') 
pair_dataframe['p1'] = pair_dataframe['p1'].str.replace(".","_")
pair_dataframe['p2'] = pair_dataframe['p2'].str.replace(".","_")

# Split the data set into a training and a validation set; 20% of the
# entire data set are used as validation set
n_val_samples = math.ceil(len(pair_dataframe) * 0.2)
# Following slicing, the index has to be reset
train_dataframe = pair_dataframe.iloc[:-n_val_samples]
train_dataframe.reset_index(drop=True, inplace=True)
val_dataframe = pair_dataframe.iloc[-n_val_samples:]
val_dataframe.reset_index(drop=True, inplace=True)

# Read the embedding matrix
embedding_matrix= h5py.File(embedding_path, 'r')
protein_keys = list(embedding_matrix.keys())
embedding_dict = dict()

for key in protein_keys:
  embedding_dict[key] = np.array(embedding_matrix[key])


def func_train(i):
    i = i.numpy() # Decoding from the EagerTensor object
    x1= pad(embedding_dict[train_dataframe['p1'][i]])
    x2= pad(embedding_dict[train_dataframe['p2'][i]])
    y = train_dataframe['label'][i]
    return x1, x2, y

def func_val(i):
   i = i.numpy() # Decoding from the EagerTensor object
   x1= pad(embedding_dict[val_dataframe['p1'][i]])
   x2= pad(embedding_dict[val_dataframe['p2'][i]])
   y = val_dataframe['label'][i]
   return x1, x2, y

def _fixup_shape(x1, x2, y):
    x1.set_shape((seq_size, dim))
    x2.set_shape((seq_size, dim)) 
    y.set_shape(()) 

    return (x1, x2), y


# Create the test data set object as well as the validation set object
train_dataset = tf.data.Dataset.from_generator(
   lambda: range(len(train_dataframe)), tf.uint64
).map(
   lambda i: tf.py_function(
       func=func_train, inp=[i], Tout=[tf.float16, tf.float16, tf.float16]
   ),
   num_parallel_calls=tf.data.AUTOTUNE
).map(_fixup_shape).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
   lambda: range(len(val_dataframe)), tf.uint64
).map(
   lambda i: tf.py_function(
      func=func_val, inp=[i], Tout=[tf.float16, tf.float16, tf.float16]
   ),
   num_parallel_calls=tf.data.AUTOTUNE
).map(_fixup_shape).batch(BATCH_SIZE)

# Architecture of MCAPS
get_custom_objects().update({'leaky_relu': leaky_relu})
get_custom_objects().update({'mish': tfa.activations.mish})
get_custom_objects().update({'lisht': tfa.activations.lisht})
get_custom_objects().update({'rrelu': tfa.activations.rrelu})

model = multi_cnn(learning_rate=args.learning_rate)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

# Train on the training dataset
checkpoint = args.MCAPST5_ckpt

if checkpoint is not None:
   model = tf.keras.models.load_model(checkpoint)

# Define the ModelCheckpoint callback; due to memory reasons, only every
# fifth checkpoint is saved
# Previously, the `period` argument could conveniently be used to
# achieve this
# However, the `period` argument has unfortunately been removed in newer
# `Tensorflow` versions, thereby necessitating a custom callback
# Early stopping is not used in order to be able to observe convergence
# of the training
class EveryNthEpochCheckpoint(tf.keras.callbacks.Callback):
   def __init__(self, n, filepath):
      super().__init__()
      self.n = n
      self.filepath = filepath

   def on_epoch_end(self, epoch, logs=None):
      if (epoch + 1) % self.n == 0:
         val_loss = logs.get("val_loss")
         file_path = self.filepath.format(
            epoch=epoch, val_loss=val_loss
         )
         self.model.save(file_path)
         print(f"\nCheckpoint saved at epoch {epoch}.")

model_ckpt_callback = EveryNthEpochCheckpoint(
   n=5,
   filepath="ckpts/xCAPT5_without_XGBoost_epoch_{epoch}_val_loss_"\
    "{val_loss:.2f}.model.keras"
)

neptune_cbk = NeptuneCallback(run=run)

callbacks = [model_ckpt_callback, neptune_cbk]

# Create the `ckpts` directory in case of not already existing
if not os.path.exists("ckpts"):
   os.makedirs("ckpts")


