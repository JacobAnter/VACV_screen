# The purpose of this BASH script is to automate the creation of
# directories to run k-fold cross-validation in
# This directory creation is accomplished for two of the three
# investigated PPI prediction models (SENSE-PPI and xCAPT) as the third
# (MaTPIP) is not run on Hemera, but on Code Ocean

for i in $(seq 0 9)
do
    mkdir ./k-fold_split_${i}
done