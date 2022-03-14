"""This file contains functions used in preprocessing of sequences."""
import pandas as pd
import random

def revcompstr(seq):
    """Return the reverse complement of the given sequence."""
    rev = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join([rev[base] for base in reversed(seq)])


def get_seqdict(sequence_tbl, sequence_col="sequence", keycol="",
                ignore_missing_col=False):
    """
    Generate dictionary representation from sequence table or list.

    :param sequence_tbl: sequence table as a df or list of lists
    :param sequence_col: column name of sequences
    :param ignore_missing_col: if column name is non existent, just return keycol
        to empty string for each row
    :return: dictionary representation of sequence table
    """
    # if sequence table is neither a dataframe nor a list
    if not isinstance(sequence_tbl, pd.DataFrame) and not isinstance(sequence_tbl, list)\
       and not isinstance(sequence_tbl, dict):  # and not isinstance(sequence_tbl, dict):
        raise Exception('sequence_tbl must be either pandas data frame or a list')
    # if seqeunce table is a dataframe
    if type(sequence_tbl) == pd.DataFrame:
        # if sequence column is not found
        if sequence_col not in sequence_tbl:
            # create a list of empty strings as sequences
            if ignore_missing_col:
                if keycol:
                    kc = sequence_tbl[keycol].tolist()
                    return {k:"" for k in kc}
                else:
                    seqlist = [""] * sequence_tbl.shape[0]
            # raise exception if sequence column is not found
            else:
                raise Exception('Could not find column %s in the data frame. This can be ignored by turning on \'ignore_missing_colname\'.' % sequence_col)
        # if sequence column is found
        else:
            # return the dictionary representation of the sequences using one of the columns in the sequence table given as keys
            if keycol:
                return pd.Series(sequence_tbl[sequence_col].values, index=sequence_tbl[keycol]).to_dict()
            # get the sequences
            else:
                seqlist = sequence_tbl[sequence_col].tolist()
    # if sequence table is a list
    elif type(sequence_tbl) == dict:
        return sequence_tbl
    else:
        seqlist = list(sequence_tbl)
    # return the dictionary representation of the sequences
    return {str(k + 1): v for k, v in enumerate(seqlist)}

def gen_randseq(n):
    return''.join(random.choices('ACGT', k=n))

def itoseq(seqint, kmer):
    """Generate the sequence as a string from its integer representation."""
    nucleotides = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq = ""
    while(seqint > 0):
        seq = nucleotides[seqint & 3] + seq
        # perform bitwise right shift to get the next nucleotide
        seqint >>= 2
    # append 'A' to the left of the string until the desired length
    if len(seq) < kmer:
        seq = 'A' * (kmer - len(seq)) + seq
    # return the sequence as a string
    return seq


def seqtoi(seq):
    """Generate the integer representation of the given sequence."""
    nucleotides = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    binrep = 0
    # loop through each nucleotide
    for i in range(0, len(seq)):
        # perform bitwise left shift
        binrep <<= 2
        # add the binary representation of the current nucleotide
        binrep |= nucleotides[seq[i]]
    # return the integer representation of the sequence
    return binrep

def makefasta(sequences, path):
    """
    Make fasta file from a list of sequences or dictionary

    Args:
        sequences: list of sequences or dictionary, if list then key will be generated
            automatically
        path: where to create the fasta file
     Returns:
        NA
    """
    if isinstance(sequences, list):
        seqdict = {"seq-%d"%(i+1):sequences[i] for i in range(len(sequences))}
    elif isinstance(sequences, dict):
        seqdict =  dict(sequences)
    else:
        raise ValueError("sequences must be a list or dictionary")
    fastastr = ""
    for k,v in seqdict.items():
        fastastr += ">%s\n%s\n" % (k,v)
    with open(path, 'w') as f:
        f.write(fastastr[:-1]) # ignore last new line
