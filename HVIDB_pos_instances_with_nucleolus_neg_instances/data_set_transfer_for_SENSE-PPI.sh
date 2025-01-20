#!/bin/bash

# The purpose of this BASH script is do automate the transfer of the
# data set files to the SENSE-PPI directory on Hemera

# As k-fold cross-validation with k being equal to 10 is supposed to be
# performed, there are 10 different splits in total
for i in $(seq 0 9)
do
    scp data_set_splits/{VACV_WR_pos_and_neg_PPIs_test_split_${i}.tsv,VACV_WR_pos_and_neg_PPIs_train_val_split_${i}.tsv} \
    anter87@hemera5:~/PPI_prediction/SENSE-PPI/k-fold_split_${i}
done