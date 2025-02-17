"""
The purpose of this Python script is to train MaTPIP from scratch on the
human PPI data set employed in the D-SCRIPT publication ("D-SCRIPT
translates genome to phenome with sequence-based, structure-aware,
genome-scale predictions of protein-protein interactions"). This is
necessary as the MaTPIP authors do not provide any pre-trained weights.
The human PPI data set from the D-SCRIPT publication is used for
training since the MaTPIP authors use this very data set for training in
the context of cross-species PPIs.
"""

import sys

path_to_MaTPIP = "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/codebase"
sys.path.insert(0, path_to_MaTPIP)

import numpy as np

from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from utils.PPIPUtils import parseTSV
from proc.mat_p2ip.mat_p2ip_origMan_auxTlOtherMan.mat_MatP2ipNetwork_origMan_auxTlOtherMan_held20 import MatP2ipNetworkModule

# Specify the path of the human PPI training data set utilised by
# D-SCRIPT as well as that of the corresponding FASTA file
path_to_human_PPI_data_set_tsv = (
    "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/dataset/human_PPI"
    "_data_set_D-SCRIPT/human_train.tsv"
)
# The current implementation of MaTPIP expects the FASTA file to rerieve
# protein sequences from to bear the name `allSeqs.fasta`, at least in
# the case of one-dimensional manually curated features as well as of
# one-dimensional pre-trained PLM-based features
# Therefore, its path is not explicitly specified

# Perform feature extraction
# Mind the trailing slash in the path definition below, it must be
# present!
path_to_data_set = "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/"\
    "dataset/human_PPI_data_set_D-SCRIPT/"
feature_path = "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/"\
    "dataset/preproc_data/derived_feat/PPI_Datasets/Human2021/"

extract_prot_seq_2D_manual_feat(
    feature_path,
    set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7']),
    spec_type="human"
)

# Now load the TSV file harbouring the PPI pairs to train the model on
training_set = np.asarray(parseTSV(path_to_human_PPI_data_set_tsv))

# Finally, train the model and save it to a file
# However, prior to that, the hyperparameters must be set and the
# features loaded
hyp = {'fullGPU':True,'deviceType':'cuda'}
hyp['hiddenSize'] = 70
hyp['numLayers'] = 4
hyp['n_heads'] = 1
hyp['layer_1_size'] = 1024

model = MatP2ipNetworkModule(hyp)
model.loadFeatureData(feature_path)

model.train(training_set)

model.saveModelToFile(
    path_to_data_set
    +
    "MaTPIP_model_trained_on_D-SCRIPT_human_PPI_data_set.out"
)