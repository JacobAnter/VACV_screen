#!/bin/bash

for i in $(seq 1 16)
do
    # In BASH, the dollar sign in conjunction with parentheses is used
    # in order to access a command's output, whereas the dollar sign in
    # conjunction with curly braces is used to access variables
    mkdir ./chunk_${i}
    mkdir ./chunk_${i}/esm2_embs_3B
done

# Now that one separate directory has been created for each chunk, the
# respective FASTA, TSV and TXT files are assigned to them
for i in $(seq 1 16)
do
    mv \
    PPI_pairs_between_Qiagen_subset_and_VACV_WR_proteome_chunk_${i}_size_500,000.tsv\
    human_prots_Qiagen_subset_and_VACV_WR_prots_seqs_chunk_${i}_size_500,000.fasta\
    UniProt_IDs_chunk_${i}.txt ./chunk_${i} || true
done

# As a last step, for each chunk, the respective ESM2 embeddings have to
# be copied into the respective directories
# To this end, the UniProt IDs stored in the directory's TXT file are
# employed
for i in $(seq 1 16)
do
    # The read command only reads lines with line breaks at the end
    # Hence, special handling is required so as to ensure that all lines
    # are
    # For more information, see https://stackoverflow.com/questions/5249
    # 5289/last-line-of-a-file-is-not-reading-in-shell-script
    while read uniprot_id || [ -n "$uniprot_id" ]
    do
        # In order to continue script execution in spite of the
        # occurrence of errors, the `|| true` construct can be used
        # (it is thus similar to try-except blocks in Python)
        cp ./esm2_embs_3B/${uniprot_id}.pt ./chunk_${i}/esm2_embs_3B || true
    done < ./chunk_${i}/UniProt_IDs_chunk_${i}.txt
done