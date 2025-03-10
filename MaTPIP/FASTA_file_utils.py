"""
This Python script harbours utilities required for satisfying the
requirements MaTPIP places on FASTA files. Bundling the utilities in a
Python script allows to import them into and thus to incorporate the
preprocessing steps into the MaTPIP main script. This obviates the
occurrence of errors.
"""

def flatten_seqs_in_fasta(path_to_fasta):
    """
    This function performs flattening of the sequences in FASTA files,
    i.e. in case of the sequences spanning multiple lines, they are
    processed such that each sequence occupies only one line. The
    original FASTA file is overwritten with these changes.

    Parameters
    ----------
    path_to_fasta: str
        A string denoting the path to a FASTA file the sequences of
        which potentially span multiple lines.

    Returns
    -------
    None
    """

    # Bear in mind that in the context of working with files, the `with`
    # context manager is preferred as it automatically takes care of
    # closing files, even in case of errors/exceptions
    with open(path_to_fasta, "r") as f:
        fasta_lines = f.readlines()
    
    header_line_indices = [
        i for i, line in enumerate(fasta_lines) if line.startswith(">")
    ]
    
    one_line_per_seq_fasta_lines = []

    for i in range(len(header_line_indices)):
        # Extract the current FASTA entry
        current_start_idx = header_line_indices[i]
        try:
            current_end_idx = header_line_indices[i + 1]
        except IndexError:
            current_end_idx = len(fasta_lines)
        
        current_FASTA_entry = fasta_lines[
            current_start_idx:current_end_idx
        ]

        # Now, remove the newline character, i.e. \n from the end of the
        # sequence lines
        # The very first element of the current entry's list is always
        # the header
        sequence_lines = current_FASTA_entry[1:]
        sequence_lines = [
            line[:-1] if "\n" in line else line
            for line in sequence_lines
        ]

        # Accommodate the fact that unless the very last FASTA entry is
        # dealt with, the last sequence line of the entry at hand is
        # supposed to contain a newline character at its end
        if current_start_idx < header_line_indices[-1]:
            sequence_lines[-1] = sequence_lines[-1] + "\n"
        
        # Now, merge the processed sequence lines and append them in
        # conjunction with the header line to the
        # `one_line_per_seq_fasta_lines` list
        one_line_per_seq_fasta_lines.append(current_FASTA_entry[0])
        merged_seq = "".join(sequence_lines)
        one_line_per_seq_fasta_lines.append(merged_seq)
    
    # Finally, overwrite the original FASTA file with sequences
    # occupying only one line
    with open(path_to_fasta, "w") as f:
        f.writelines(one_line_per_seq_fasta_lines)