# Results files for chunks 0 to 2 have alreadt been transferred to the
# local machine
# Therefore, the procedure is applied from chunk 3 onwards
# Also bear in mind that two chunks have jointly been processed in one
# batch job, which is why there is one output file for two consecutive
# chunks
# The output file is transferred to the directory of the chunk with the
# odd number

for i in $(seq 3 16)
do
    # Note that in the if-statement below, the expression to be
    # evaluated is enclosed in square brackets
    # This is due to the fact that square brackets are synonymous with
    # the `test`` command; the if-statement operates on the exit status
    # of a command, so the square brackets or the test function are
    # required
    # Without square brackets (or the `test` function), the comparison
    # that remains is just an expression the if-statement cannot operate
    # on
    # Moreover, the fact that square brackets are tantamount to a
    # function also explains why there is a space immediately after the
    # opening brackets and immediately before the closing bracket,
    # respectively: the command is separated from its parameters
    if [ $(($i % 2)) -ne 0 ]
    then
        scp anter87@hemera5:~/PPI_prediction/SENSE-PPI/SENSE-PPI_on_Qiagen_subset_chunk_${i}_and_$((i + 1))_size_500,000.out \
        Results_chunk_${i}
    fi

    scp anter87@hemera5:~/PPI_prediction/SENSE-PPI/chunk_${i}/{"predictions_Qiagen_subset_chunk_${i}_size_500,000.tsv","predictions_positive_interactions_Qiagen_subset_chunk_${i}_size_500,000.tsv"} \
    Results_chunk_${i}
done