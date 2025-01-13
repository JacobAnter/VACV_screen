"""
This Python script performing a sample run with xCAPT5 has to be run in
the corresponding virtual environment.

The objective of this Python script is to reproduce the results reported
in the supplementary material for the interspecies HIV virus data set.
"""
import os
import time

import torch
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Dense, Input, Dropout, BatchNormalization, Conv1D, GlobalMaxPooling1D, \
    AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D)
from transformers import T5EncoderModel, T5Tokenizer
from xgboost import XGBClassifier
import sklearn.metrics as metrics
from sklearn.metrics import (matthews_corrcoef,accuracy_score,
  precision_score,recall_score)
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# For some strange reason, Tensorflow is looking the wrong directories
# for the CuDNN library (i.e. the libcudnn.so.8 file)
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
# TSV files comprises combinations of those protein IDS
# Therefore, a sample run is conducted with xCAPT5 involving the same
# test files as with SENSE-PPI
pair_path = "pairs_sub.tsv"
seq_path = "dict_sub.fasta"

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

def read_fasta( fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    seqs = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                seqs[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq= ''.join( line.split() ).upper().replace("-","")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq 
    example_id=next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id,seqs[example_id]))

    return seqs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.")

def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer

# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings( model, tokenizer, seqs, per_residue, per_protein, sec_struct, 
                   max_residues=4000, max_seq_len=1000, max_batch=100 ):

    if sec_struct:
      sec_struct_model = load_sec_struct_model()

    results = {"residue_embs" : dict(), 
               "protein_embs" : dict(),
               "sec_structs" : dict() 
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict,1)):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct: # in case you want to predict secondary structure from embeddings
              d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)


            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim  
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if sec_struct: # get classification results
                    results["sec_structs"][identifier] = torch.max( d3_Yhat[batch_idx,:s_len], dim=1 )[1].detach().cpu().numpy().squeeze()
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()


    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results

def save_embeddings(emb_dict,out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None


if os.path.isfile(per_residue_path) == False:
    # Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
    model, tokenizer = get_T5_model()

    # Load example fasta.
    seqs = read_fasta( seq_path )

    for id, seq in seqs.items():
        if len(seq) > 1200:
            seqs[id] = seq[:1200]


    # Compute embeddings and/or secondary structure predictions
    results = get_embeddings(
        model, tokenizer, seqs, per_residue, per_protein, sec_struct
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

def pad(rst, length=1200, dim=1024):
    if len(rst) > length:
        return rst[:length]
    elif len(rst) < length:
        return np.concatenate((rst, np.zeros((length - len(rst), dim))))
    return rst


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

def pad(rst, length=1200, dim=1024):
    if len(rst) > length:
        return rst[:length]
    elif len(rst) < length:
        return np.concatenate((rst, np.zeros((length - len(rst), dim))))
    return rst


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
def leaky_relu(x, alpha = .2):
   return tf.keras.backend.maximum(alpha*x, x)
get_custom_objects().update({'leaky_relu': leaky_relu})
get_custom_objects().update({'mish': tfa.activations.mish})
get_custom_objects().update({'lisht': tfa.activations.lisht})
get_custom_objects().update({'rrelu': tfa.activations.rrelu})

seq_size = 1200
dim = 1024
def multi_cnn():
    DEPTH = 5
    WIDTH = 3
    POOLING_SIZE = 4
    FILTERS = 50
    KERNEL_SIZE = 2
    DEPTH_DENSE1 = 3
    DEPTH_DENSE2 = 2
    DROPOUT = DROPOUT1 = DROPOUT2 = 0.05
    DROPOUT_SPATIAL= 0.15
    ACTIVATION = 'swish'
    ACTIVATION_CNN = 'swish'
    INITIALIZER = 'glorot_normal'
    
    def BlockCNN_single(KERNEL_SIZE, POOLING_SIZE, FILTERS, LAYER_IN1, LAYER_IN2):
        c1 = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION_CNN, padding='same')
        x1 = c1(LAYER_IN1)
        x2 = c1(LAYER_IN2)

        g1 = Dropout(DROPOUT)(concatenate([GlobalMaxPooling1D()(x1),GlobalAveragePooling1D()(x1)]))
        a1 = GlobalAveragePooling1D()(x1)
        g2 = Dropout(DROPOUT)(concatenate([GlobalMaxPooling1D()(x2),GlobalAveragePooling1D()(x2)]))
        a2 = GlobalAveragePooling1D()(x1)

        x1 = SpatialDropout1D(DROPOUT_SPATIAL)(concatenate([MaxPooling1D(POOLING_SIZE)(x1), AveragePooling1D(POOLING_SIZE)(x1)]))
        x2 = SpatialDropout1D(DROPOUT_SPATIAL)(concatenate([MaxPooling1D(POOLING_SIZE)(x2), AveragePooling1D(POOLING_SIZE)(x2)]))

        return x1, x2, g1, g2, a1, a2

    def BlockCNN_multi(POOLING_SIZE, FILTERS, LAYER_IN1, LAYER_IN2, WIDTH):
      X1 = []
      X2 = []
      G1 = []
      G2 = []
      A1 = []
      A2 = []
      for i in range(2, 2+WIDTH):
        x1, x2, g1, g2, a1, a2 = BlockCNN_single(i, POOLING_SIZE, FILTERS, LAYER_IN1, LAYER_IN2)
        X1.append(x1)
        X2.append(x2)
        G1.append(g1)
        G2.append(g2)
        A1.append(a1)
        A2.append(a2)
      x1 = concatenate(X1)
      x2 = concatenate(X2)
      g1 = GlobalMaxPooling1D()(x1)
      g2 = GlobalMaxPooling1D()(x2)
      return x1, x2, g1, g2

    def BlockCNN_single_deep(KERNEL_SIZE, POOLING_SIZE, DEPTH, FILTERS, LAYER_IN1, LAYER_IN2):
      X1 = []
      X2 = []
      G1 = []
      G2 = []
      A1 = []
      A2 = []
      x1 = LAYER_IN1
      x2 = LAYER_IN2
      for i in range(DEPTH):
        x1, x2, g1, g2, a1, a2 = BlockCNN_single(KERNEL_SIZE, POOLING_SIZE, FILTERS, x1, x2)
        X1.append(x1)
        X2.append(x2)
        G1.append(g1)
        G2.append(g2)
        A1.append(a1)
        A2.append(a2)

      return X1, X2, G1, G2, A1, A2

    input1 = Input(shape=(seq_size, dim), name="seq1")
    input2 = Input(shape=(seq_size, dim), name="seq2")
    


    X1 = dict()
    X2 = dict()
    G1 = dict()
    G2 = dict()
    A1 = dict()
    A2 = dict()

    for i in range(KERNEL_SIZE, KERNEL_SIZE+WIDTH):
      X1[f'{i}'], X2[f'{i}'], G1[f'{i}'], G2[f'{i}'], A1[f'{i}'], A2[f'{i}'] = BlockCNN_single_deep(i, POOLING_SIZE, DEPTH, FILTERS, input1, input2)

    s1 = []
    s2 = []
    for i in range(KERNEL_SIZE, KERNEL_SIZE+WIDTH):
      s1.extend(G1[f'{i}'])
      s2.extend(G2[f'{i}'])

    s1 = concatenate(s1)
    s2 = concatenate(s2)
    
    s1 = BatchNormalization(momentum=.9)(s1)
    s2 = BatchNormalization(momentum=.9)(s2)

    s1 = Dropout(DROPOUT1)(s1)
    s2 = Dropout(DROPOUT1)(s2)
    
    s1_shape = s1.shape[-1]
    DENSE1 = 744 
    d1 = []
    for i in range(DEPTH_DENSE1):
        d1.append(Dense(int(DENSE1*(1/2)**i), kernel_initializer=INITIALIZER, activation=ACTIVATION))

    for i in range(DEPTH_DENSE1):
        s1 = d1[i](s1)
        s2 = d1[i](s2)
        s1 = Dropout(DROPOUT1)(s1)
        s2 = Dropout(DROPOUT1)(s2)
        
    s = concatenate([s1, s2])

    
    s_shape = s.shape[-1]
    DENSE2 = 328
        
    d2 = []
    for i in range(DEPTH_DENSE2):
        d2.append(Dense(int(DENSE2*(1/2)**i), kernel_initializer=INITIALIZER, activation=ACTIVATION))

    for i in range(DEPTH_DENSE2):
        s = d2[i](s)
        s = Dropout(DROPOUT2)(s)

    output = Dense(1, activation='sigmoid')(s)
    model = Model(inputs=[input1, input2], outputs=[output])
    
    adabelief = tfa.optimizers.AdaBelief(
    rectify=False,
    epsilon=1e-8)
    adam = Adam(learning_rate=1e-3, amsgrad=True, epsilon=1e-6)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = multi_cnn()
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

# Load checkpoint
checkpoint_mcapst5 = "mcapst5_pan_epoch_20.hdf5"
checkpoint_xgboost = "xgboost_pan_epoch_20.bin"

model = tf.keras.models.load_model(checkpoint_mcapst5)
model_ = XGBClassifier()
model_.load_model(checkpoint_xgboost)


y_pred = model.predict(test_dataset)
y_true = pair_dataframe['label'].values
cm1=confusion_matrix(y_true, np.round(y_pred))
acc = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
spec= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
sens = (cm1[1,1])/(cm1[1,0]+cm1[1,1])
prec=cm1[1,1]/(cm1[1,1]+cm1[0,1])
rec=cm1[1,1]/(cm1[1,1]+cm1[1,0])
f1 = 2 * (prec * rec) / (prec + rec)
mcc = matthews_corrcoef(y_true, np.round(y_pred))

prc = metrics.average_precision_score(y_true, y_pred)

print("============= INFERENCE BY NEURAL NETWORK ===============")
try:
  auc = metrics.roc_auc_score(y_true, y_pred)
  print(f'accuracy: {acc}, precision: {prec}, recall: {rec}, specificity: {spec}, f1-score: {f1}, mcc: {mcc}, auroc: {auc}, auprc: {prc} ')
  print(str(acc) + "\t" + str(prec) + "\t" + str(rec) + "\t" + str(spec) + "\t" + str(f1) + "\t" + str(mcc)+"\t" + str(auc)  + "\t" + str(prc) + "\n")
except ValueError:
  print(f'accuracy: {acc}, precision: {prec}, recall: {rec}, specificity: {spec}, f1-score: {f1}, mcc: {mcc}, auroc: nan, auprc: {prc} ')
  print(str(acc) + "\t" + str(prec) + "\t" + str(rec) + "\t" + str(spec) + "\t" + str(f1) + "\t" + str(mcc)+"\t nan"  + "\t" + str(prc) + "\n")


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
# Train=Train.sample(frac=1)

shape_x = model.layers[-2].get_output_at(0).get_shape()[1]
X=Train.iloc[:,0:shape_x].values
y=Train.iloc[:,shape_x:].values

extracted_df=X_train_feat


y = y.reshape(-1, )
model_= XGBClassifier(booster='gbtree', reg_lambda=1, alpha=1e-7, subsample=0.8, colsample_bytree=0.2, n_estimators=100, max_depth=5, min_child_weight=2, gamma=1e-7, eta=1e-6)
model_.fit(X, y, verbose=False)


y_pred = model_.predict(X)
y_true = y
cm1=confusion_matrix(y_true, np.round(y_pred))
acc = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
spec= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
sens = (cm1[1,1])/(cm1[1,0]+cm1[1,1])
prec=cm1[1,1]/(cm1[1,1]+cm1[0,1])
rec=cm1[1,1]/(cm1[1,1]+cm1[1,0])
f1 = 2 * (prec * rec) / (prec + rec)
mcc = matthews_corrcoef(y_true, np.round(y_pred))

prc = metrics.average_precision_score(y_true, y_pred)

print("============= INFERENCE BY HYBRID MODEL ===============")
try:
  auc = metrics.roc_auc_score(y_true, y_pred)
  print(f'accuracy: {acc}, precision: {prec}, recall: {rec}, specificity: {spec}, f1-score: {f1}, mcc: {mcc}, auroc: {auc}, auprc: {prc} ')
  print(str(acc) + "\t" + str(prec) + "\t" + str(rec) + "\t" + str(spec) + "\t" + str(f1) + "\t" + str(mcc)+"\t" + str(auc)  + "\t" + str(prc) + "\n")
except ValueError:
  print(f'accuracy: {acc}, precision: {prec}, recall: {rec}, specificity: {spec}, f1-score: {f1}, mcc: {mcc}, auroc: nan, auprc: {prc} ')
  print(str(acc) + "\t" + str(prec) + "\t" + str(rec) + "\t" + str(spec) + "\t" + str(f1) + "\t" + str(mcc)+"\t nan"  + "\t" + str(prc) + "\n")