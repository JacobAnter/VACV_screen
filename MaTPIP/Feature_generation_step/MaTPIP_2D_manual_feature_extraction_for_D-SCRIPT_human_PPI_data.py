"""
The purpose of this Python script is to extract two-dimensional manual
features for the human PPI data set employed in the D-SCRIPT publication
("D-SCRIPT translates genome to phenome with sequence-based,
structure-aware, genome-scale predictions of protein-protein
interactions"). The reason this is done as a separate step and not in
conjunction with the model training is that this step does not leverage
GPU acceleration and is therefore very time-consuming (this at least
applies to the two features "PSSM" and "Blosum62"). In the light of this
step's reliance on CPU rather than GPU, it the corresponding shell
script should be run on a CPU partition with maximum run time.
"""

import sys

path_to_MaTPIP = "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/codebase"
sys.path.insert(0, path_to_MaTPIP)

from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat

# Perform feature extraction
# Mind the trailing slash in the path definition below, it must be
# present!
feature_path = "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/"\
    "dataset/preproc_data/derived_feat/PPI_Datasets/Human2021/"

# Feature extraction for the features "SkipGramAA7" as well as
# "LabelEncoding" was successfully completed in the batch job trying to
# perform 2D manual feature extraction in conjunction with model
# training
# Yet, feature extraction still has to be performed for the two features
# "PSSM" and "Blosum62"
extract_prot_seq_2D_manual_feat(
    feature_path,
    set(['PSSM', 'Blosum62']),
    spec_type="human"
)