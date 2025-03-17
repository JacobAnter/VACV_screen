#!/bin/bash

# The purpose of this BASH script is to automate the transfer of the
# data set files to the SENSE-PPI directory on Hemera

# Usage of sshpass obviates the necessity to repeatedly enter the
# password
prompt="Enter password: "
read -p "${prompt}" -s password
echo "Password successfully entered."

# As k-fold cross-validation with k being equal to 10 is supposed to be
# performed, there are 10 different splits in total
# Additionally, there are four chunks for each split (in the case of the
# test sets)
# Iterate over the 10 splits
for i in $(seq 0 9)
do
    # Iterate over the four different chunks each test set has been
    # subdivided into
    for j in $(seq 0 3)
    do
        sshpass -p "${password}" scp data_set_splits/data_set_split_${i}/{VACV_WR_pos_and_neg_PPIs_test_set_split_${i}_chunk_${j}.tsv,VACV_WR_pos_and_neg_PPIs_test_set_prot_seqs_split_${i}_chunk_${j}.fasta} \
        anter87@hemera5:~/PPI_prediction/SENSE-PPI/k-fold_split_${i}
    done
done

echo "Data transfer accomplished."