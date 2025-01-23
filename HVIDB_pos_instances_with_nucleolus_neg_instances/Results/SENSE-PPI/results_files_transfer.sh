#!/bin/bash

# Note that in the case of SENSE-PPI, each test set has been split into
# four chunks
# Therefore, for each of the 10 splits, eight files are transferred to
# the local machine in total (SENSE-PPI generates two output files, one
# of which contains all predictions and the other of which lists
# exclusively positive interactions; as two output files are generated
# for each of four chunks, the total amount of output files per split is
# eight)

# Usage of sshpass obviates the necessity to repeatedly enter the
# password
prompt="Enter password: "
read -p "${prompt}" -s password
echo "Password successfully entered."

# Iterate over the 10 splits
for i in $(seq 0 9)
do
    # Iterate over the 4 different chunks each test set has been
    # subdivided into
    for j in $(seq 0 3)
    do
        sshpass -p "${password}" scp anter87@hemera5:~/PPI_prediction/SENSE-PPI/k-fold_split_${i}/predictions_on_VACV_WR_pos_and_neg_data_set_test_set_split_${i}_chunk_${j}_without_training_positive_interactions.tsv \
        results_split_${i}

        sshpass -p "${password}" scp anter87@hemera5:~/PPI_prediction/SENSE-PPI/k-fold_split_${i}/predictions_on_VACV_WR_pos_and_neg_data_set_test_set_split_${i}_chunk_${j}_without_training.tsv \
        results_split_${i}
    done
done

echo "Data transfer accomplished."