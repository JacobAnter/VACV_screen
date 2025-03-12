"""
This Python script running xCAPT5 has to be run in the corresponding
virtual environment.

Also note that this implementation accepts input from the command line.
"""

import os
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
from xgboost import XGBClassifier
from tqdm import tqdm

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

parser.add_argument("--pairs_file")
parser.add_argument("--fasta_file")
parser.add_argument("--output")
parser.add_argument("--XGBoost_checkpoint")

args = parser.parse_args()

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

# The implementation provided by the authors expects the TSV file to
# have a third column harbouring interaction labels (i.e. 1 for True and
# 0 for False), irrespective of whether they are known or not
# Therefore, in case the TSV used does not already possess a third
# column, one is introduced having zero as its values
# Also note that XGBoost should not be fitted when using arbitrary
# labels as this does not make any sense!
pair_df = pd.read_csv(pair_path, sep="\t", header=None)
if pair_df.shape[1] < 3:
   pair_df["Label"] = 0
   pair_df.to_csv(
      pair_path,
      sep="\t",
      header=False,
      index=False
   )

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


# Read the embedding matrix
embedding_matrix= h5py.File(embedding_path, 'r')
protein_keys = list(embedding_matrix.keys())
embedding_dict = dict()

for key in protein_keys:
  embedding_dict[key] = np.array(embedding_matrix[key])


def func(i):
    i = i.numpy() # Decoding from the EagerTensor object
    x1= pad(embedding_dict[pair_dataframe['p1'][i]])
    x2= pad(embedding_dict[pair_dataframe['p2'][i]])
    y = pair_dataframe['label'][i]
    return x1, x2, y

def _fixup_shape(x1, x2, y):
    x1.set_shape((seq_size, dim))
    x2.set_shape((seq_size, dim)) 
    y.set_shape(()) 

    return (x1, x2), y


# Create the test data set object
test_dataset = tf.data.Dataset.from_generator(
   lambda: range(len(pair_dataframe)), tf.uint64
).map(
   lambda i: tf.py_function(
       func=func, inp=[i], Tout=[tf.float16, tf.float16, tf.float16]
   ),
   num_parallel_calls=tf.data.AUTOTUNE
).map(_fixup_shape).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Architecture of MCAPS
get_custom_objects().update({'leaky_relu': leaky_relu})
get_custom_objects().update({'mish': tfa.activations.mish})
get_custom_objects().update({'lisht': tfa.activations.lisht})
get_custom_objects().update({'rrelu': tfa.activations.rrelu})

model = multi_cnn()
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

# Load checkpoint
# Prior to that, it is verified whether the checkpoint files exist in
# the current directory
# If not, they are downloaded
checkpoint_mcapst5 = "mcapst5_pan_epoch_20.hdf5"
if args.XGBoost_checkpoint == None:
    checkpoint_xgboost = "xgboost_pan_epoch_20.bin"
else:
   checkpoint_xgboost = args.XGBoost_checkpoint

if not os.path.isfile(checkpoint_mcapst5):
   os.system(
      'wget https://github.com/anhvt00/MCAPS/raw/master/checkpoint/Pan/mcapst5_pan_epoch_20.hdf5'
   )
if (
   (checkpoint_xgboost == "xgboost_pan_epoch_20.bin")
   and
   (not os.path.isfile(checkpoint_xgboost))
):
   os.system(
      'wget https://github.com/anhvt00/MCAPS/raw/master/checkpoint/Pan/xgboost_pan_epoch_20.bin'
   )

model = tf.keras.models.load_model(checkpoint_mcapst5)
model_ = XGBClassifier()
model_.load_model(checkpoint_xgboost)


# Evaluate on the test data set with MCAPST5-X
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(model.layers[-2].name).output)

# Use intermediate layer to transform pairs matrix
pred = intermediate_layer_model.predict(test_dataset)
p_merge=pd.DataFrame(pred)    
Trainlabels = pair_dataframe['label']
# create dataframe use transformed pairs matrix outputs and labels
X_train_feat=pd.concat((p_merge,pd.DataFrame(pd.DataFrame(Trainlabels))),axis=1,ignore_index=True)

# write to file dataframe of transformed pairs matrix and labels
X_train_feat.to_csv('X_train.csv',header=False, index=False)

# read dataframe of transformed pairs matrix and labels
Train=pd.read_csv("X_train.csv",header=None)

shape_x = model.layers[-2].get_output_at(0).get_shape()[1]
X=Train.iloc[:,0:shape_x].values

y_pred = model_.predict_proba(X)

# Save the prediction results to a TSV file
# In the TSV file, the first and second column contains the first and
# second interaction partner, respectively, whereas the third column
# harbours the predicted probability
# The `predict_proba` method of the xgboost library returns one 
# probability for each class; in our case, there are two classes, i.e.
# class 0 for no interaction and class 1 for interaction
# However, we are exclusively interested in the probabilities for
# belonging to class 1, which is why the returned array is sliced
# accordingly
results_df = pd.DataFrame(
   data={
      "protein_1": pair_dataframe["p1"],
      "protein_2": pair_dataframe["p2"],
      "interaction_probability": y_pred[:, 1]
   }
)

results_df.to_csv(
   f"xCAPT5_interaction_probs_{args.output}_with_XGBoost_no_fitting.tsv",
   sep="\t",
   index=False
)