"""
The purpose of this Python script is to generate AMPLIFY embeddings for
the proteins comprised in the FASTA file VACV_WR_prots_pos_human_prots_
human_peroxisome_prots_and_human_nucleolus_prots_max_length_1700_AAs
.fasta". To be more precise, the 350M parameters model of AMPLIFY with
an extended context length of 2,048 is supposed to be used.
"""

import os
import logging

import torch
import amplify
from biotite.sequence.io import fasta


# By default, the logging module logs messages with a severity level of
# WARNING or above
# Thus, the logging module has to be configured to log events of all
# levels
logging.basicConfig(level=logging.INFO)


amplify_emb_dir = "amplify_embs_350m"
if not os.path.exists(amplify_emb_dir):
    os.makedirs(amplify_emb_dir)


# Load the FASTA file
path_to_fasta = (
    "VACV_WR_prots_pos_human_prots_human_peroxisome_prots_and_human_"
    "nucleolus_prots_max_length_1700_AAs.fasta"
)

VACV_and_human_prots_fasta = fasta.FastaFile.read(path_to_fasta)


# Load the AMPLIFY model
config_path = (
    "/bigdata/casus/MLID/Jacob/AMPLIFY_dir/AMPLIFY/checkpoints/AMPLIFY"
    "_350M/config.yaml"
)
checkpoint_file = (
    "/bigdata/casus/MLID/Jacob/AMPLIFY_dir/AMPLIFY/checkpoints/AMPLIFY"
    "_350M/pytorch_model.pt"
)

model, tokenizer = amplify.AMPLIFY.load(checkpoint_file, config_path)

# Link the model to the inference API
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Device used: {DEVICE}")
predictor = amplify.inference.Predictor(model, tokenizer, DEVICE)


# Iterate over the individual protein sequences and obtain their
# embeddings
for header, seq in VACV_and_human_prots_fasta.items():
    # The 350M version of AMPLIFY has an inherent maximum sequence
    # length of 2,000 amino acids
    # Thus, for protein sequences exceeding this maximum length, the
    # sequence is split into overlapping chunks and the chunks are
    # encoded separately
    # The overlap encompasses 100 amino acids
    # As mean pooling is supposed to be performed, splitting the
    # sequence into chunks does not pose any problem
    if len(seq) > 2000:
        # Generate the chunks via a sliding window
        WINDOW_SIZE = 2000
        OVERLAP = 100
        STEP = WINDOW_SIZE - OVERLAP

        seq_chunks = []

        for i in range(0, len(seq), STEP):
            seq_chunk = seq[i:i + WINDOW_SIZE]
            seq_chunks.append(seq_chunk)
    else:
        seq_chunks = [seq]
    
    embs_list = []
    # Generate an embedding for each chunk
    for seq_chunk in seq_chunks:
        chunk_emb = predictor.embed(seq_chunk)
        embs_list.append(chunk_emb)
    
    # Finally, save the embedding(s) to a file
    result = {"label": header}
    result["representations"] = {
        -1: embs_list
    }

    torch.save(
        result,
        os.path.join(amplify_emb_dir, f"{header}.pt")
    )