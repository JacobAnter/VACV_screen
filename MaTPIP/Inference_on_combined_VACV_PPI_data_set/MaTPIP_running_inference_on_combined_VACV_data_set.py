"""
The purpose of this Python script is to run inference of MaTPIP trained
on the human D-SCRIPT data set on the combined VACV data set
(consisting of confirmed positive interactions and reliable negative
interactions).
"""

import sys

import pandas as pd
import numpy as np

path_to_MaTPIP = "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/codebase"
sys.path.insert(0, path_to_MaTPIP)

from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from utils.PPIPUtils import parseTSV
from proc.mat_p2ip.mat_p2ip_origMan_auxTlOtherMan.mat_MatP2ipNetwork_origMan_auxTlOtherMan_held20 import MatP2ipNetworkModule

# Define model hyperparameters
hyp = {'fullGPU':True,'deviceType':'cuda'} 

hyp['hiddenSize'] = 70
hyp['numLayers'] = 4
hyp['n_heads'] = 1
hyp['layer_1_size'] = 1024

path_to_trained_model = (
    "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/dataset/"
    "human_PPI_data_set_D-SCRIPT/MaTPIP_model_trained_on_D-SCRIPT_"
    "human_PPI_data_set.out"
)

# Mind the trailing slash in the path definition below, it must be
# present!
feature_path = (
    "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/dataset/"
    "preproc_data/derived_feat/PPI_Datasets/Human2021/"
)

path_to_test_set = (
    "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/dataset/human_"
    "PPI_data_set_D-SCRIP/VACV_WR_pos_and_nucleolus_prots_neg_PPI_"
    "instances_without_header.tsv"
)

extract_prot_seq_2D_manual_feat(
    feature_path,
    set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7']),
    spec_type="human_nucleolus_and_VACV_WR_prot_seqs"
)

# Load the test set
test_set = np.asarray(parseTSV(path_to_test_set))

model = MatP2ipNetworkModule(hyp)
model.loadFeatureData(feature_path)

model.genModel()
model.model.load(path_to_trained_model)

print(
    "\nCool, loading the trained model was successful!\n"
)

predictions, _ = model.predictPairs(test_set)

# Create the output TSV file
# In the output TSV file, the PPI pairs predictions were made for are
# supposed to be listed along with the corresponding predicted
# probabilities
# The two interaction partners of a PPI occupy one column each, whereas
# the corresponding predicted probability populates a third column
# Hence, the TSV file encompasses three columns the first of which
# harbours the first interaction partner, the second of which harbours
# the second interaction partner and the third of which lists the
# corresponding predicted probabilities
# As to the predictions returned by the `predictPairs` method above, it
# must be noted that they are provided in the form of a NumPy array of
# shape (n,2), where n is the amount of PPIs fed into the method
# The two columns, on the other hand, represent the available target
# labels in ascending order, i.e. in this case 0 and 1, with 0
# representing a non-existing PPI (negative label) and 1 representing an
# existing PPI (positive label)
# As only the probabilities of belonging to label 1 are of interest, the
# NumPy array is sliced accordingly
probs_pos_label = predictions[:, 1]

# The first two columns of the TSV file with PPI pairs are expected to
# contain the interaction partners
ppi_pairs_df = pd.read_csv(path_to_test_set, sep="\t", header=None)
first_int_partner = ppi_pairs_df.iloc[:, 0].to_list()
second_int_partner = ppi_pairs_df.iloc[:, 1].to_list()

# Finally, assemble the output TSV file and save it to disk
data = {
    "protein_1": first_int_partner,
    "protein_2": second_int_partner,
    "prob": probs_pos_label
}
output_df = pd.DataFrame(data)

output_df.to_csv(
    "predicted_probs_pos_label_combined_VACV_PPIs_data_set.tsv",
    sep="\t",
    header=False,
    index=False
)

print(
    "\nCool, running inference was successful!\n"
)