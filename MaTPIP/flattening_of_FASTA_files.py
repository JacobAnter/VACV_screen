"""
The purpose of this Python script is to perform the flattening of the
FASTA files MaTPIP operates on. The reason this is done in a separate
Python script rather than in the main script is that the generation of
one-dimensional manual features as well as the generation of PLM-based
features rely on the TSV files too. Accordingly, this script has to be
executed prior to the generation of the features.
"""

from FASTA_file_utils import flatten_seqs_in_fasta

# MaTPIP does not allow the sequences in FASTA files to span multiple
# lines
# Therefore, the sequences have to be "flattened", i.e. altered such
# that they occupy only one line
FASTA_file_path_1 = (
    "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/dataset/orig_"
    "data/Human2021/allSeqs.fasta"
)
FASTA_file_path_2 = (
    "/bigdata/casus/MLID/Jacob/MaTPIP/mat_p2ip_prj/dataset/preproc_"
    "data/derived_feat/PPI_Datasets/Human2021/human_proteins_in_"
    "combined_data_set_and_VACV_WR_proteome.fasta"
)

FASTA_file_paths = [FASTA_file_path_1, FASTA_file_path_2]

for FASTA_file_path in FASTA_file_paths:
    flatten_seqs_in_fasta(FASTA_file_path)