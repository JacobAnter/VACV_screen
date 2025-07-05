"""
The purpose of this Jupyter notebook is to generate ESM-3 embeddings for
the proteins comprised in the FASTA file "VACV_WR_prots_pos_human_prots_
human_peroxisome_prots_and_human_nucleolus_prots_max_length_1700_AAs
.fasta". To be more precise, the 6B parameter model of ESM C (ESM
Cambrian) is supposed to be used as it is the largest. ESM C focuses on
the generation of protein representations/embeddings and is thus most
suitable for this endeavor. It is intended to use the embeddings for the
generation of soft/probabilistic negative labels.
"""

import os

import torch
from biotite.sequence.io import fasta
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.sdk import batch_executor


esm3_embed_dir = "esm_c_embs_6B"
if not os.path.exists(esm3_embed_dir):
    os.makedirs(esm3_embed_dir)


# Load the FASTA file and extract the headers as well as the sequences
path_to_fasta = (
    "../VACV_WR_prots_pos_human_prots_human_peroxisome_prots_and_human_"
    "nucleolus_prots_max_length_1700_AAs.fasta"
)

VACV_and_human_prots_fasta = fasta.FastaFile.read(path_to_fasta)

headers, seqs = zip(*VACV_and_human_prots_fasta.items())
headers = list(headers)
seqs = list(seqs)


client = ESM3ForgeInferenceClient(
    model="esmc-6b-2024-12",
    url="https://forge.evolutionaryscale.ai",
    token="7fgiOd4qw8rkrZEePTYhsF"
)

def embed_sequence(model, sequence):
    with torch.no_grad():
        protein = ESMProtein(sequence=sequence)
        protein_tensor = model.encode(protein)
        output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=False, return_embeddings=True)
        )
    
    return output

with batch_executor() as executor:
    outputs = executor.execute_batch(
        user_func=embed_sequence,
        model=client,
        sequence=seqs
    )


# Now, iterate over the embeddings of the individual protein sequences
# and save them to a file
for i, header in enumerate(headers):
    # Bear in mind that during tokenization, a beginning-of-sequence
    # token as well as an end-of-sequence token are added to the
    # beginning and the end of the sequence, respectively
    # Thus, the embedding tensor has to be sliced such that they are
    # removed
    current_embedding = outputs[i].embeddings[0, 1:-1]

    result = {"label": header}
    # Call clone on tensors to ensure tensors are not views into a
    # larger representation
    # See https://github.com/pytorch/pytorch/issues/1995
    result["representations"] = {
        -1: current_embedding.clone()
    }

    torch.save(
        result,
        os.path.join(esm3_embed_dir, f"{header}.pt")
    )