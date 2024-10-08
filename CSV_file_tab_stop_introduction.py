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
# Note that reading all the lines into memory at once via `.readlines()`
# provokes an Out Of Memory error, which is why the file lines are
# iterated over on the fly
with open(
    "Vaccinia_Report_NCBI_Gene_IDs_and_official_gene_symbols_updated.csv",
    "r"
) as prior_tab_intro_file, open(
    "adjusted_file.csv", "w", newline=""
) as post_tab_intro_file:
    for i, line in enumerate(prior_tab_intro_file):
        # Bear in mind that the first line represents the header, i.e.
        # contains the column names
        # Thus, all commata represent actual delimiters
        if i == 0:
            split_line = [
                i for j in line.split(",") for i in (j, ",")
            ][:-1]

            # Simply replace all commata with tab stops
            split_line_with_tabs = [
                "\t" if i == "," else i for i in split_line
            ]
            
            # Concatenate the entries in the updated list and write the
            # resulting string to the file
            post_tab_intro_file.write("".join(split_line_with_tabs))
            continue

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
        # Also keep in mind that the numeric index starts with zero, not
        # 1, so that when counting in the "human" way, column 30 has
        # index 31
        # Therefore, the index corresponding to column 30 is not 30, but
        # 30 * 2 = 60 (counting starts with 0, hence the first column
        # has index 0; to account for the remaining 30 entries, 30 * 2
        # is added, yielding 60, the index of the entry corresponding to
        # column 30)
        # Also bear in mind that the column has at least one entry,
        # which is why the index of the first element to query is
        # increased by two, i.e. 62
        entry_index_1 = 62
        subsequent_entry = split_line[entry_index_1]
        while subsequent_entry not in siRNA_error_options:
            entry_commata_list.append(entry_index_1 - 1)
            entry_index_1 += 2
            subsequent_entry = split_line[entry_index_1]
        
        # Now, do the same thing with column 62, i.e. "Gene_Description"
        # Again, the index of the entry in `split_line` corresponding to
        # column 62 is not 62, but 62 * 2 = 124, and as the
        # column contains at least one entry, the index of the first
        # entry to query is increased by two (126)
        # Note that the index of the first entry to investigate has to
        # be adjusted according to the previous amount of "entry
        # commata"
        entry_index_2 = 126 + len(entry_commata_list) * 2
        subsequent_entry = split_line[entry_index_2]
        while (
            (subsequent_entry != "Not available")
            and
            (subsequent_entry[:4] != "ENST")
        ):
            entry_commata_list.append(entry_index_2 - 1)
            entry_index_2 += 2
            subsequent_entry = split_line[entry_index_2]
        
        # Update the list harbouring the row entries along with the
        # delimiters by replacing commata with tab stops at the
        # corresponding positions
        for comma_index in line_comma_indices:
            if comma_index not in entry_commata_list:
                split_line[comma_index] = "\t"
        
        # Finally, the entries in the updated row list are concatenated
        # and the resulting string is written to the file
        # As the `.readlines()` method does not trim line endings, the
        # newline character (\n) does not have to be added
        post_tab_intro_file.write("".join(split_line))