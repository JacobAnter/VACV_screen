"""
The purpose of this Python script is to generate ESM-2 embeddings for
the proteins comprised in the FASTA file "VACV_WR_prots_pos_human_
prots_human_peroxisome_prots_and_human_nucleolus_prots_max_length_
1700_AAs.fasta". To be more precise, the 15B parameter model of ESM-2
is supposed to be used as it is the largest. It is intended to use the
embeddings for the generation of soft/probabilistic negative labels.
"""

import os
from copy import copy
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
from biotite.sequence.io import fasta
from esm import FastaBatchedDataset, pretrained


def add_general_args(parser):
    parser.add_argument("--device", type=str, default='auto', choices=['cpu', 'gpu', 'mps', 'auto'],
                        help="Device to use for computations. Options include: cpu, gpu, mps (for MacOS), and auto."
                             "If not selected the device is set by torch automatically. WARNING: mps is temporarily "
                             "disabled, if it is chosen, cpu will be used instead.")

    return parser


# By default, the logging module logs messages with a severity level of
# WARNING or above
# Thus, the logging module has to be configured to log events of all
# levels
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()

parser = add_general_args(parser)

parser.add_argument("fasta_file", type=str)
parser.add_argument("--esm_model", type=str, default="esm2_t48_15B_UR50D")
parser.add_argument("--embed_dir", type=str, default="esm2_embs_15B")
parser.add_argument(
    "--toks_per_batch_esm", type=int, default=4096,
    help="Maximum batch size in units of tokens"
)

args = parser.parse_args()


def compute_embeds(params):
    model, alphabet = pretrained.load_model_and_alphabet(params.esm_model)
    # Setting the model to evaluation mode disables dropout, amongst
    # others
    model.eval()

    if params.device == "gpu":
        model = model.cuda()
        logging.info("Transferred the ESM-2 model to GPU.")
    elif params.device == "mps":
        model = model.to("mps")
        logging.info("Transferred the ESM-2 model to MPS.")
    
    dataset = FastaBatchedDataset.from_file(params.fasta_file)
    batches = dataset.get_batch_indices(
        params.toks_per_batch_esm, extra_toks_per_seq=1
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    logging.info(f"Read {params.fasta_file} with {len(dataset)} sequences.")

    Path(params.embed_dir).mkdir(parents=True, exist_ok=True)

    repr_layer = [(-1 + model.num_layers + 1) % (model.num_layers + 1)]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            logging.info(
                f"Processing {batch_idx + 1} of {len(batches)} batches "
                f"({toks.size(0)} sequences)."
            )

            if params.device == "gpu":
                toks = toks.to(device="cuda", non_blocking=True)
            elif params.device == "mps":
                toks = toks.to(device="mps", non_blocking=True)
            
            out = model(toks, repr_layers=repr_layer, return_contacts=False)

            representations = {
                layer: t.to(device="cpu")
                for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                params.output_file = "/".join(
                    [params.embed_dir, f"{label}.pt"]
                )
                Path(params.output_file).parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}

                truncate_len = len(strs[i])

                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                result["representation"] = {
                    -1: t[i, 1: truncate_len + 1].clone()
                    for _, t in representations.items()
                }

                torch.save(
                    result,
                    params.output_file
                )


if not os.path.exists(args.embed_dir):
    compute_embeds(args)
else:
    fasta_file = fasta.FastaFile.read(args.fasta_file)
    seq_dict = dict(fasta_file.items())
    seq_ids = list(fasta_file.keys())

    for seq_id in seq_ids:
        if os.path.exists(os.path.join(args.embed_dir, seq_id + ".pt")):
            seq_dict.pop(seq_id)
    
    if len(seq_dict) > 0:
        args_esm = copy(args)

        try:
            current_time = str(datetime.now()).replace(" ", "_")
            args_esm.fasta_file = args.fasta_file.replace(
                "fasta", current_time + ".fasta.tmp"
            )

            with open(args_esm.fasta_file, "w") as f:
                for seq_id in seq_dict.keys():
                    f.write(">" + seq_id + "\n")
                    f.write(seq_dict[seq_id] + "\n")
            
            compute_embeds(args_esm)

        except Exception as e:
            raise e
        finally:
            if os.path.exists(args_esm.fasta_file):
                os.remove(args_esm.fasta_file)
    else:
        logging.info("All ESM-2 embeddings have already been computed.")