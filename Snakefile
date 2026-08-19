rule all:
    input:
        "results/final_results.csv"


rule read_fasta:
    input:
        "sample_seqs/s.fasta"
    output:
        "results/sequences.csv"
    conda:
        "environment.yml"
    shell:
        "python scripts/01_read_fasta.py {input} {output}"




