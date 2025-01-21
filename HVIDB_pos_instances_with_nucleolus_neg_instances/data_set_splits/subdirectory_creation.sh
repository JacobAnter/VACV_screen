# The purpose of this BASH script is to create a subdirectory for each
# data set split
# The necessity to do so stems from the fact that the test set has to be
# split into multiple parts, which in turn arises from the fact that
# SENSE-PPI reads in the embeddings of all sequences at once, thereby
# potentially leading to an OutOfMemory (OOM) error

#!/bin/bash

for i in $(seq 0 9)
do
    mkdir "data_set_split_${i}"
done