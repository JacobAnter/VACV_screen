#!/bin/bash

# The purpose of this BASH script is to transfer the 10 splits of the
# HVIDB VACV WR data set to the respective KSGPPI folders

# Usage of sshpass obviates the necessity to repeatedly enter the
# password
prompt="Enter password: "
read -p "${prompt}" -s password
echo "Password successfully entered."

for i in $(seq 0 9)
do
    sshpass -p "${password}" scp whole_splits_FASTA_files/split_${i}/* \
    anter87@hemera5:/bigdata/casus/MLID/Jacob/KSGPPI/HVIDB_VACV_WR_data_set/whole_split_FASTAs_split_${i}
done

echo "Data transfer accomplished."