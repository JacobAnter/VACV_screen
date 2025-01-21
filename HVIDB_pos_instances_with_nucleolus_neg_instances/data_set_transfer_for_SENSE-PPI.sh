#!/bin/bash

# The purpose of this BASH script is do automate the transfer of the
# data set files to the SENSE-PPI directory on Hemera

# As k-fold cross-validation with k being equal to 10 is supposed to be
# performed, there are 10 different splits in total
# Additionally, there are three chunks for each split (in the case of
# the test sets)
for i in $(seq 0 9)
do
    scp data_set_splits/data_set_split_${i}/{VACV_WR_pos_and_neg_PPIs_test_set_split_${i}_chunk_0.tsv,VACV_WR_pos_and_neg_PPIs_test_set_split_${i}_chunk_1.tsv,VACV_WR_pos_and_neg_PPIs_test_set_split_${i}_chunk_2.tsv,VACV_WR_pos_and_neg_PPIs_test_set_prot_seqs_split_${i}_chunk_0.fasta,VACV_WR_pos_and_neg_PPIs_test_set_prot_seqs_split_${i}_chunk_1.fasta,VACV_WR_pos_and_neg_PPIs_test_set_prot_seqs_split_${i}_chunk_2.fasta} \
    anter87@hemera5:~/PPI_prediction/SENSE-PPI/k-fold_split_${i}
done