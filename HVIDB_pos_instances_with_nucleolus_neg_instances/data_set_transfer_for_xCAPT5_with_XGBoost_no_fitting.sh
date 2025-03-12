#!/bin/bash

# The purpose of this BASH script is to automate the transfer of the
# data set files to the directory for xCAPT5 with XGBoost without
# fitting on Hemera

# Usage of sshpass obviates the necessity to repeatedly enter the
# password
prompt="Enter password: "
read -p "${prompt}" -s password
echo "Password successfully entered."

# As k-fold cross-validation with k being equal to 10 is supposed to be
# performed, there are 10 different splits in total
# Iterate over the 10 splits
for i in $(seq 0 9)
do
    sshpass -p "${password}" scp data_set_splits/VACV_WR_pos_and_neg_PPIs_test_split_${i}.tsv \
    anter87@hemera5:~/PPI_prediction/xCAPT5/10-fold_cross-validation/without_training/with_XGBoost_no_fitting
done

echo "Data transfer accomplished."