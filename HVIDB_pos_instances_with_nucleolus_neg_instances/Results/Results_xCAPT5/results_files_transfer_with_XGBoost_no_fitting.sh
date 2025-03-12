#!/bin/bash

# Usage of sshpass obviates the necessity to repeatedly enter the
# password
prompt="Enter password: "
read -p "${prompt}" -s password
echo "Password successfully entered."

# Iterate over the 10 test sets
for i in $(seq 0 9)
do
    # The xCAPT5 source code has been tweaked such that one output file
    # is generated
    # Thus, for each test set, only one file has to be transferred
    sshpass -p "${password}" scp anter87@hemera5:~/PPI_prediction/xCAPT5/10-fold_cross-validation/without_training/with_XGBoost_no_fitting/xCAPT5_interaction_probs_VACV_WR_10-fold_cross-val_test_set_${i}_with_XGBoost_no_fitting.tsv \
    Results_with_XGBoost_no_fitting
done