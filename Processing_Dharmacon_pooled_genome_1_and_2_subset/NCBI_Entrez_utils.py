"""
This Python script harbours functions adding information from NCBI
Entrez to Pandas DataFrames.
"""

import numpy as np
import pandas as pd
from biotite.sequence.io import fasta


class NCBI_Entrez_data_lookup:
    """
    ...

    Parameters
    ----------
    gene_info_file_path: str
        A string denoting the path to the `gene_info` file from NCBI
        Entrez.
    gene_history_file_path: str
        A string denoting the path to the `gene_history` file from NCBI
        Entrez.
    gene2accession_file_path: str
        A string denoting the path to the `gene2accession` file from
        NCBI Entrez.
    gene_refseq_uniprot_collab_file_path: str
        A string denoting the path to the
        `gene_refseq_uniprot_collab` file from NCBI Entrez.
    sec_ac_file_path: str
        A string denoting the path to the `sec_ac.txt` file from the
        UniProt FTP file server.
    all_human_prots_fasta_file_path: str
        A path to a FASTA file containing all human proteins in UniProt,
        i.e. from both Swiss-Prot and TrEMBL.
    """
    def __init__(
            self,
            gene_info_file_path,
            gene_history_file_path,
            gene2accession_file_path,
            gene_refseq_uniprot_collab_file_path,
            sec_ac_file_path,
            all_human_prots_fasta_file_path
    ):
        self.gene_info = pd.read_csv(
            gene_info_file_path,
            sep="\t",
            usecols=["GeneID", "Symbol", "type_of_gene"],
            dtype={"GeneID": object}
        )
        self.gene_history = pd.read_csv(
            gene_history_file_path,
            sep="\t",
            usecols=["GeneID", "Discontinued_GeneID"],
            dtype={"Discontinued_GeneID": object}
        )
        self.gene2accession = pd.read_csv(
            gene2accession_file_path,
            sep="\t",
            usecols=["GeneID", "protein_accession.version"],
            dtype={"GeneID": object}
        )
        self.gene_refseq_uniprot_collab = pd.read_csv(
            gene_refseq_uniprot_collab_file_path,
            sep="\t",
            usecols=["#NCBI_protein_accession", "UniProtKB_protein_accession"]
        )

        # Simply loading the `sec_ac.txt` file into a Pandas DataFrame
        # is unfortunately not possible as it contains explanatory text
        # at its very beginning
        # Thus, the mapping between secondary accessions and their
        # current primary accessions has to be extracted manually
        # Bear in mind that in the context of working with files, the
        # `with` context manager is preferred as it automatically takes
        # care of closing files, even in case of errors/exceptions
        with open(sec_ac_file_path, "r") as f:
            # The accession mapping starts in line 32
            lines = f.readlines()[31:]
        
        # Create a dictionary mapping secondary accessions to their
        # current primary accessions
        self.sec_to_prim_acc_dic = {}

        for line in lines:
            sec_ac, prim_ac = line.split()
            self.sec_to_prim_acc_dic[sec_ac] = prim_ac
        
        # Some supposedly current, i.e. valid primary accessions have
        # been retracted without being listed in the `delac_sp.txt` and
        # `delac_tr.txt` files
        # Thus, a FASTA file containing all human proteins from both
        # Swiss-Prot and TrEMBL is used to identify retracted accessions
        self.all_human_prots_fasta = fasta.FastaFile.read(
            all_human_prots_fasta_file_path
        )


    def check_gene_id_and_symbol(self, pd_df):
        """
        Checks both the gene ID and the official gene symbol for
        individual genes in a Pandas DataFrame.

        This method assumes that gene IDs are stored in a column named
        `ID_manufacturer`, whereas the gene names/official gene symbols
        are contained in a column named `Name`. Additionally, it expects
        the presence of a column called `ID`. Since the column
        `ID_manufacturer` is consistently populated, while the column
        `ID` is not, the entries in `ID_manufacturer` are used to
        populate the corresponding fields of `ID`. Also the presence of
        a columns bearing the names `Withdrawn_by_NCBI` and `Gene_type`
        is expected.

        Two files from NCBI Entrez are employed for cross-referencing,
        which are `gene_info` and `gene_history`. The cross-referencing
        procedure is as follows: First, the gene ID at hand is looked
        for in the `gene_info` file. If there is a match, the
        corresponding official gene symbol is retrieved. In case that
        the gene ID cannot be found in the `gene_info` file, an attempt
        is made to find it in the `gene_history` file as this indicates
        that the gene ID at hand has either been replaced or retired
        altogether. The corresponding value in the `Withdrawn_by_NCBI`
        column in set accordingly.

        Note that this method merely modifies that DataFrame without
        saving it to an e.g. TSV/CSV file. It is thus incumbent on the
        user to save the modified DataFrame.

        Parameters
        ----------
        pd_df: Pandas DataFrame
            A Pandas DataFrame containing the gene IDs and gene symbols
            to be checked/cross-referenced with the NCBI Entrez
            information. As stated above, the DataFrame is expected to
            comprise the following columns: `ID_manufacturer`, `ID`,
            `Name`, `Gene_type` and `Withdrawn_by_NCBI`.
        
        Returns
        -------
        updated_df: Pandas DataFrame
            The modified Pandas DataFrame passed as input, i.e. with the
            gene ID and official gene symbol cross-referenced with NCBI
            Entrez information.
        """
        # There are several approaches to cross-referencing data between
        # DataFrames
        # The arguably fastest approach involves capitalizing on merge
        # operations, which are implemented in C and thus extremely fast

        # First, retrieve the official gene symbols as well as the gene
        # types for gene IDs that are still valid
        updated_df = pd_df.merge(
            self.gene_info,
            left_on="ID_manufacturer",
            right_on="GeneID",
            how="left"
        )

        # Three new columns, i.e. `GeneID`, `Symbol` and `type_of_gene`
        # have been appended to the right side of the DataFrame
        # Note that for gene IDs without a match, `np.nan` is used as
        # value
        # The `Name` column is overwritten with the newly added `Symbol`
        # column, following which the `GeneID` and `Symbol` columns are
        # dropped
        updated_df["Name"] = updated_df["Symbol"]
        updated_df.drop(labels="GeneID", axis=1, inplace=True)
        updated_df.drop(labels="Symbol", axis=1, inplace=True)

        # Do the same for the `Gene_type` column and the `type_of_gene`
        # column
        updated_df["Gene_type"] = updated_df["type_of_gene"]
        updated_df.drop(labels="type_of_gene", axis=1, inplace=True)

        # Now, address gene IDs with no matches with current, valid gene
        # IDs
        # To this end, an auxiliary DataFrame is generated mapping the
        # outdated gene IDs to current gene IDs, if available
        # The merge operation below appends the `GeneID` column to the
        # right side of the DataFrame containing current gene IDs, if
        # available
        # If not, a simple hyphen is present
        hist_df = pd_df.merge(
            self.gene_history,
            left_on="ID_manufacturer",
            right_on="Discontinued_GeneID",
            how="left"
        )

        # A second merge operation is performed so as to retrieve the
        # current official gene symbol as well as the gene type, if
        # available
        # Accordingly, this operation appends the `Symbol` and
        # `type_of_gene` columns to the right side of the DataFrame
        hist_df = hist_df.merge(
            self.gene_info,
            left_on="GeneID",
            right_on="GeneID",
            how="left"
        )

        # Now, fill in the missing `Name` and `Gene_type` values for
        # replaced gene IDs
        updated_df["Name"] = updated_df["Name"].fillna(
            hist_df["Symbol"]
        )
        updated_df["Gene_type"] = updated_df["Gene_type"].fillna(
            hist_df["type_of_gene"]
        )

        # Bear in mind that the `ID_manufacturer` column still needs to
        # be updated for replaced gene IDs
        # This is accomplished in the following manner: A dictionary is
        # constructed mapping exclusively discontinued gene IDs to their
        # current gene ID (or a hyphen in case of them having been
        # withdrawn)
        # By means of this dictionary, the `ID_manufacturer` column is
        # updated with gene ID replacements
        # In the same breath, the `Withdrawn_by_NCBI` column is
        # populated with either "Yes" or "No"
        history_map = self.gene_history.set_index(
            "Discontinued_GeneID"
        )["GeneID"].to_dict()

        def _update_gene_id_and_status(gene_id):
            if gene_id in history_map:
                replacement_id = history_map[gene_id]
                if replacement_id == "-":
                    return pd.Series((gene_id, "Yes"))
                else:
                    return pd.Series((replacement_id, "No"))
            else:
                # The gene ID did not undergo any change
                return pd.Series((gene_id, "No"))
        
        updated_df[
            ["ID_manufacturer", "Withdrawn_by_NCBI"]
        ] = updated_df["ID_manufacturer"].apply(_update_gene_id_and_status)

        # Do not forget to adopt the values in `ID_manufacturer` for the
        # `ID` column
        updated_df["ID"] = updated_df["ID_manufacturer"]

        return updated_df


    def add_uniprot_ids(self, pd_df):
        """
        Maps individual gene IDs in a Pandas DataFrame to the
        corresponding UniProt accessions.

        In detail, this mapping is accomplished in two steps, the first
        of which consists of mapping gene IDs to NCBI protein accessions
        and the second of which involves mapping NCBI proteins
        accessions to UniProtKB protein accessions.

        ...

        As with the other method, this method merely modifies the
        DataFrame without saving it to an e.g. TSV/CSV file. It is thus
        incumbent on the user to save the modified DataFrame.

        Parameters
        ----------
        pd_df: Pandas DataFrame
            ...
        
        Returns
        -------
        updated_df: Pandas DataFrame
            ...
        """
        # As in the method above, fast merge operations implemented in C
        # are capitalized on
        # Conduct a merge between the `protein_accession.version` column
        # of `gene2accession` and the `#NCBI_protein_accession` column
        # of `gene_refseq_uniprot_collab`, appending the
        # `UniProtKB_protein_accession` column to the right side of the
        # resulting DataFrame
        # The resulting DataFrame has four columns, namely `GeneID`,
        # `protein_accession.version`, `#NCBI_protein_accession` and
        # `UniProtKB_protein_accession`
        gene_id_ncbi_acc_uniprot_acc_df = self.gene2accession.merge(
            self.gene_refseq_uniprot_collab,
            left_on="protein_accession.version",
            right_on="#NCBI_protein_accession",
            how="left"
        )

        # There is the possibility that a UniProtKB protein accession
        # deposited in the `gene_refseq_uniprot_collab` file turned to
        # a secondary accession, i.e. has been replaced with another
        # primary accession
        # In order to accommodate this possibility, the dictionary
        # created upon class instantiation is used
        # There is no need to define a private function for this purpose
        # as `pandas.Series.map` also accepts a dictionary as argument
        uniprot_acc_col = "UniProtKB_protein_accession"
        gene_id_ncbi_acc_uniprot_acc_df[uniprot_acc_col] = (
            gene_id_ncbi_acc_uniprot_acc_df[uniprot_acc_col]
            .map(self.sec_to_prim_acc_dic)
            .fillna(gene_id_ncbi_acc_uniprot_acc_df[uniprot_acc_col])
        )

        # Yet another layer of complexity stems from the fact that some
        # supposedly current, i.e. valid UniProtKB accessions have been
        # retracted without being listed in the `delac_sp.txt` and
        # `delac_tr.txt` files
        # Thus, the FASTA file containing all human proteins and loaded
        # upon object instantiation is used so as to identify these
        # supposedly current, but actually retracted protein accessions
        all_human_prots_accs = set(self.all_human_prots_fasta.keys())

        # Define a private function returning either the UniProtKB
        # protein accession in case of it being valid or `NaN` in case
        # of it being retracted
        def _check_acc_validity(acc):
            return acc if acc in all_human_prots_accs else np.nan
        
        gene_id_ncbi_acc_uniprot_acc_df[uniprot_acc_col] = (
            gene_id_ncbi_acc_uniprot_acc_df[uniprot_acc_col]
            .map(_check_acc_validity)
        )
        
        # Now, the `UniProt_IDs` column of `pd_df` is populated in the
        # following way:
        # For each gene ID, all UniProtKB protein accessions are
        # gathered and concatenated with semicolons as separator
        # The concatenated string is set as value for the respective
        # gene ID
        # As individual gene IDs occur multiple times, the merged
        # DataFrame is grouped by gene ID and the UniProtKB accessions
        # are collected for each gene ID once
        # The `grouped` DataFrame has two columns, which are `GeneID`
        # and `UniProtKB_protein_accession`
        grouped = (
            gene_id_ncbi_acc_uniprot_acc_df.dropna(
                subset=["UniProtKB_protein_accession"]
            )
            .groupby("GeneID")["UniProtKB_protein_accession"]
            .apply(lambda accs: ";".join(sorted(set(accs))))
            .reset_index()
        )

        # The input DataFrame is merged with the `grouped` DataFrame so
        # as to map the gene IDs to the UniProtKB protein accessions
        updated_df = pd_df.merge(
            grouped,
            left_on="ID_manufacturer",
            right_on="GeneID",
            how="left"
        )

        # A `UniProt_IDs` column has already been added; adopt the
        # values from the `UniProtKB_protein_accession` column and
        # subsequently drop both the `GeneID` and the
        # `UniProtKB_protein_accession column
        updated_df["UniProt_IDs"] = updated_df["UniProtKB_protein_accession"]
        updated_df.drop(labels="GeneID", axis=1, inplace=True)
        updated_df.drop(labels="UniProtKB_protein_accession", axis=1, inplace=True)

        return updated_df