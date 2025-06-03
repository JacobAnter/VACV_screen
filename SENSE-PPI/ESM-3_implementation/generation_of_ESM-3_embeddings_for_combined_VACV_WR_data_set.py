"""
The purpose of this Python script is to generate ESM-3 embeddings for
the proteins involved in the combined VACV WR data set. As ESM Cambrian
(ESMC) does not inherently support batch processing of multiple
sequences, multiprocessing is utilized for parallelization.
"""

import os

import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from biotite.sequence.io import fasta
import multiprocessing as mp

esm3_emb_dir = "esm3_embs_300m"
if not os.path.exists(esm3_emb_dir):
    os.makedirs(esm3_emb_dir)

MAX_LEN = 800

# As a preliminary step, extract the pairs of UniProt IDs and sequences
# from the FASTA file
path_to_fasta = "human_nucleolus_and_VACV_WR_prot_seqs.fasta"
VACV_WR_fasta = fasta.FastaFile.read(path_to_fasta)
id_seq_pairs = list(VACV_WR_fasta.items())

# Now, define a worker function that will be executed in parallel
def embed_sequence(id_seq_pair):
    uniprot_ac = id_seq_pair[0]
    seq = id_seq_pair[1]

    model = ESMC.from_pretrained("esmc_300m").to("cpu").eval()

    # Convert the string to an ESMProtein
    protein = ESMProtein(sequence=seq)

    # Finally, encode the protein and its embeddings
    with torch.no_grad():
        token_tensor = model.encode(protein)

        output = model.logits(
            token_tensor,
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
            36: embedding[:MAX_LEN].clone()
        }
        torch.save(
            result,
            os.path.join(esm3_emb_dir, f"{uniprot_ac}.pt")
        )


if __name__ == "__main__":
    n_workers = mp.cpu_count()

    with mp.Pool(processes=n_workers) as pool:
        pool.map(embed_sequence, id_seq_pairs)