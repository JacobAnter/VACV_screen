"""
The purpose of this Python script is to generate ESM-3 embeddings for
the proteins comprised in the FASTA file "VACV_WR_prots_pos_human_prots_
human_peroxisome_prots_and_human_nucleolus_prots_max_length_1700_AAs
.fasta". To be more precise, the 600M parameter model of ESM C (ESM
Cambrian) is supposed to be used. ESM C focuses on the generation of
protein representations/embeddings and is thus most suitable for this
endeavor. It is intended to use the embeddings for the generation of
soft/probabilistic negative labels.
"""

import os
import logging

import torch
from biotite.sequence.io import fasta
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


# By default, the logging module logs messages with a severity level of
# WARNING or above
# Thus, the logging module has to be configured to log events of all
# levels
logging.basicConfig(level=logging.INFO)


esm3_emb_dir = "esm_c_embs_600m"
if not os.path.exists(esm3_emb_dir):
    os.makedirs(esm3_emb_dir)

# As a preliminary step, extract the pairs of UniProt IDs and sequences
# from the FASTA file
path_to_fasta = (
    "../VACV_WR_prots_pos_human_prots_human_peroxisome_prots_and_human_"
    "nucleolus_prots_max_length_1700_AAs.fasta"
)
VACV_and_human_prots_fasta = fasta.FastaFile.read(path_to_fasta)
id_seq_pairs = list(VACV_and_human_prots_fasta.items())


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Device used: {DEVICE}")

model = ESMC.from_pretrained("esmc_600m").to(DEVICE).eval()

for id_seq_pair in id_seq_pairs:
    uniprot_ac = id_seq_pair[0]
    seq = id_seq_pair[1]

    with torch.no_grad():
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)
        output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=False, return_embeddings=True)
        )

        # Bear in mind that during tokenisation, a beginning-of-sequence
        # token as well as an end-of-sequence token are added to the
        # beginning and the end of the sequence, respectively
        # Thus, the embedding tensor has to be sliced such that they are
        # removed
        embedding = output.embeddings[0, 1:-1]

        # As a last step, the embedding is saved to a file
        result = {"label": uniprot_ac}
        # Call clone on tensors to ensure tensors are not views into a
        # larger representation
        # See https://github.com/pytorch/pytorch/issues/1995
        result["representations"] = {
            -1: embedding.clone()
        }

        torch.save(
            result,
            os.path.join(esm3_emb_dir, f"{uniprot_ac}.pt")
        )