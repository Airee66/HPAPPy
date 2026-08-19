#convert fastas to dataframes
from pathlib import Path
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data_folder = Path("sample_seqs") 
fasta_dfs = {}

for fasta_file in data_folder.iterdir():
    if fasta_file.suffix.lower() not in {".fasta", ".fa", ".fas", ".fna"}:
        continue

    sequence_data = [
        {
            "id": record.id,
            "sequence": str(record.seq)
        }
        for record in SeqIO.parse(fasta_file, "fasta")
    ]

    dataframe_name = fasta_file.stem
    fasta_dfs[dataframe_name] = pd.DataFrame(sequence_data)
    #add a column for the host type based on the dataframe name
    fasta_dfs[dataframe_name]["host"] = dataframe_name

    print(f"{dataframe_name} seqs: {len(fasta_dfs[dataframe_name])}")

############## -- Combine all dataframes into a single dataframe -- ##############

combined_df = pd.concat(fasta_dfs.values(), ignore_index=True)
print(f"----Combined seqs----: {len(combined_df)}")
print(combined_df.tail(3))

############## -- convert to k-mers -- ##############
def make_kmers(seq, k=3):
    seq = seq.upper()
    return " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])

combined_df["kmers"] = combined_df["sequence"].apply(lambda x: make_kmers(x, k=3))

vectorizer = CountVectorizer()
X_ha = vectorizer.fit_transform(combined_df["kmers"]) #kmer features
y_ha = combined_df["host"] #labels

#print count of each host type
host_counts = combined_df["host"].value_counts()
print("Host counts:")

############## -- PCA -- ##############
print("Performing PCA...")

X = vectorizer.fit_transform(combined_df["kmers"])

X_dense = X.toarray()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_dense)

plot_df = combined_df.copy()
plot_df["PC1"] = X_pca[:, 0]
plot_df["PC2"] = X_pca[:, 1]

for host in plot_df["host"].unique():
    subset = plot_df[plot_df["host"] == host]
    plt.scatter(subset["PC1"], subset["PC2"], label=host, alpha=0.7)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of HA k-mer Features")
plt.legend()
#plt.xlim(-30, 30)
#plt.ylim(-30, 30)
#save the plot
plt.savefig("results/pca_plot.png")


