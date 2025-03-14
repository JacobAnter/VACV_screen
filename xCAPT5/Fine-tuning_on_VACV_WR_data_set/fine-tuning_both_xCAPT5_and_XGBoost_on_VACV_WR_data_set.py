"""
The purpose of this Python script is to fine-tune the xCAPT5
architecture to the combined VACV WR data set encompassing both
confirmed positive PPIs and reliable negative PPIs. To be more precise,
both the actual xCAPT5 architecture and XGBoost are fine-tuned.

Fine-tuning is performed as the model as is obtains a rather mediocre
performance. The rationale behind fine-tuning xCAPT5 without XGBoost is
that XGBoost outputs probability values either slightly above or
slightly below 0.5, which could potentially compromise the refinement of
the normalised intensity values. I assume that probability values
spanning a larger range might benefit the rescaling of the normalised
intensities.
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
parser.add_argument("--xCAPT5_ckpt")

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
train_dataset = tf.data.Dataset.from_generator(
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

# Train on the training dataset
checkpoint = args.xCAPT5_ckpt

model = tf.keras.models.load_model(checkpoint)

# Define callbacks (EarlyStopping as well as ModelCheckpoint)
early_stopping_callback = EarlyStopping()
model_ckpt_callback = ModelCheckpoint(
   filepath="xCAPT5_without_XGBoost_epoch_{epoch}_val_loss_"\
    "{val_loss:.2f}.model.keras"
)
callbacks = [early_stopping_callback, model_ckpt_callback]

model.fit(train_dataset, epochs=40, callbacks=callbacks)

# Fit XGBoost for learned representations from MCAPST5
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(model.layers[-2].name).output)

# Use intermediate layer to transform pairs matrix
pred = intermediate_layer_model.predict(train_dataset)
p_merge=pd.DataFrame(pred)    
Trainlabels = pair_dataframe['label']
# create dataframe use transformed pairs matrix outputs and labels
X_train_feat=pd.concat((p_merge,pd.DataFrame(pd.DataFrame(Trainlabels))),axis=1,ignore_index=True)

# write to file dataframe of transformed pairs matrix and labels
X_train_feat.to_csv('X_train.csv',header=False, index=False)

# read dataframe of transformed pairs matrix and labels
Train=pd.read_csv("X_train.csv",header=None)
# Train=Train.sample(frac=1)

shape_x = model.layers[-2].get_output_at(0).get_shape()[1]
X=Train.iloc[:,0:shape_x].values
y=Train.iloc[:,shape_x:].values

extracted_df=X_train_feat


y = y.reshape(-1, )
model_= XGBClassifier(booster='gbtree', reg_lambda=1, alpha=1e-7, subsample=0.8, colsample_bytree=0.2, n_estimators=100, max_depth=5, min_child_weight=2, gamma=1e-7, eta=1e-6)
model_.fit(X, y, verbose=False)


# save xgboost parameters 
model_.save_model("xCAPT5_fitted_XGBoost_for_fine-tuned_main_model.bin")