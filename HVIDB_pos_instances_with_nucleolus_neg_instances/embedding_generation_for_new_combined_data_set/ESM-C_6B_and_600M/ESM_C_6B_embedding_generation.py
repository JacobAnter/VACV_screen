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
import logging
from datetime import datetime

import torch
from biotite.sequence.io import fasta
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig


# By default, the logging module logs messages with a severity level of
# WARNING or above
# Thus, the logging module has to be configured to log events of all
# levels
logging.basicConfig(level=logging.INFO)


path_to_fasta = (
    "../VACV_WR_prots_pos_human_prots_human_peroxisome_prots_and_human_"
    "nucleolus_prots_max_length_1700_AAs.fasta"
)


client = ESM3ForgeInferenceClient(
    model="esmc-6b-2024-12",
    url="https://forge.evolutionaryscale.ai",
    token="7fgiOd4qw8rkrZEePTYhsF"
)

def _embed_sequence(model, sequence):
    with torch.no_grad():
        protein = ESMProtein(sequence=sequence)
        protein_tensor = model.encode(protein)
        output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=False, return_embeddings=True)
        )
    
    return output


def compute_embeds(model, fasta_file_path):
    fasta_file = fasta.FastaFile.read(fasta_file_path)
    # Iterate over the individual protein sequences and generate
    # embeddings for them
    for header, seq in fasta_file.items():
        current_output = _embed_sequence(model=model, sequence=seq)
        # Bear in mind that during tokenization, a beginning-of-sequence
        # token as well as an end-of-sequence token are added to the
        # beginning and the end of the sequence, respectively
        # Thus, the embedding tensor has to be sliced such that they are
        # removed
        current_embedding = current_output.embeddings[0, 1:-1]

        # Finally, save the embedding to a file
        result = {"label": header}
        # Call clone on tensors to ensure tensors are not views into a
        # larger representation
        # See https://github.com/pytorch/pytorch/issues/1995
        result["representation"] = {
            -1: current_embedding.clone()
        }

        torch.save(
            result,
            os.path.join(esm3_embed_dir, f"{header}.pt")
        )


esm3_embed_dir = "esm_c_embs_6B"

if not os.path.exists(esm3_embed_dir):
    compute_embeds(model=client, fasta_file_path=path_to_fasta)
else:
    fasta_file = fasta.FastaFile.read(path_to_fasta)
    seq_dict = dict(fasta_file.items())
    seq_ids = list(fasta_file.keys())

    for seq_id in seq_ids:
        if os.path.exists(os.path.join(esm3_embed_dir, seq_id + ".pt")):
            seq_dict.pop(seq_id)
    
    if len(seq_dict) > 0:
        try:
            current_time = str(datetime.now()).replace(" ", "_")
            temp_fasta_file_path = path_to_fasta.replace(
                "fasta", current_time + ".fasta.tmp"
            )

            with open(temp_fasta_file_path, "w") as f:
                for seq_id in seq_dict.keys():
                    f.write(">" + seq_id + "\n")
                    f.write(seq_dict[seq_id] + "\n")
            
            compute_embeds(
                model=client, fasta_file_path=temp_fasta_file_path
            )
        except Exception as e:
            raise e
        finally:
            if os.path.exists(temp_fasta_file_path):
                os.remove(temp_fasta_file_path)
    else:
        logging.info("All ESM C embeddings have already been computed.")