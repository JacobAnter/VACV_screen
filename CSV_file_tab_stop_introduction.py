"""
Unfortunately, when the script devoted to the gene ID and official gene
symbol check was run on the HPC cluster, the updated Pandas DataFrame
was saved to a CSV file without specifying the separator to be a tab
stop. Thus, the default separator (comma) has been used, which, however,
conflicts with the usage of commata in some entries.

Therefore, the purpose of this file is to replace the commata with tab
stops in a sophisticated manner taking account of this difference
between actual delimiters and commata which are part of entries.
"""

import numpy as np
import pandas as pd

with open(
    "Vaccinia_Report_NCBI_Gene_IDs_and_official_gene_symbols_updated.csv",
    "r"
) as f:
    file_lines = f.readlines()

# List comprising unique values of column 31, i.e. the column following
# column 30 ("Name_alternatives")
siRNA_error_options = [
    "MultipleTargets",
    "NoTargets",
    "Not available",
    "OK",
    "POOLED_SIRNA_ERROR",
    "TargetMismatch",
    "Unknown"
]

# Iterate through the lines, modify them accordingly and write the
# adjusted lines to a new output file
with open("adjusted_file.csv", "w") as f:
    for line in file_lines:
        # When employing the built-in split method for strings, the
        # separation character is not retained, but discarded
        # Hence, by employing a trick involving a nested list
        # comprehension, the separation character is added at its
        # corresponding positions
        # (https://www.geeksforgeeks.org/python-string-split-including-spaces/)
        split_line = [i for j in line.split(",") for i in (j, ",")][:-1]
        
        line_comma_indices = [
            i for i, x in enumerate(split_line) if x == ","
        ]

        # Determine the indices of commata belonging to entries in lieu
        # of being delimiters
        entry_commata_list = []

        # First, deal with column 30, i.e. "Name alternatives"
        # Keep in mind that it is iterated through the list
        # `split_line`, which encompasses both the entries as well as
        # commata
        # Therefore, the index corresponding to column 30 is not 30, but
        # 1 + 28 * 2 + 1 = 58 (counting starts with 0, hence the comma
        # of the first entry has index 1; to account for the remaining
        # 28 entries, 28 * 2 is added; finally, in order to obtain the
        # index of the 30th entry, 1 is added)
        entry_index_1 = 58
        subsequent_entry = split_line[entry_index_1]
        while subsequent_entry not in siRNA_error_options:
            entry_commata_list.append(entry_index_1 - 1)
            entry_index_1 += 2
        
        # Now, do the same thing with column 62, i.e.
        # ("Gene_Description")
        # Again, the index of the entry in `split_line` corresponding to
        # column 62 is not 62, but 1 + 60 * 2 + 1 = 122
        # Note that the index of the first entry to investigate has to
        # be adjusted according to the previous amount of "entry
        # commata"
        entry_index_2 = 122 + len(entry_commata_list) * 2
        subsequent_entry = split_line[entry_index_2]
        while (
            (subsequent_entry != "Not available")
            and
            (subsequent_entry[:4] != "ENST")
        ):
            entry_commata_list.append(entry_index_2 - 1)
            entry_index_2 += 2
        
        # Update the list harbouring the row entries along with the
        # delimiters by replacing commata with tab stops at the
        # corresponding positions
        for comma_index in line_comma_indices:
            if comma_index not in entry_commata_list:
                split_line[comma_index] != "\t"
        
        # Finally, the entries in the updated row list are concatenated
        # and the resulting string is written to the file
        # As the `.readlines()` method does not trim line endings, the
        # newline character (\n) does not have to be added
        f.write("".join(split_line))