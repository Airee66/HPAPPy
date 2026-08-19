import streamlit as st
from pathlib import Path
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Host Prediction Explorer", layout="wide")

st.title("Host Prediction — k-mer PCA Explorer")

# Sidebar controls
st.sidebar.header("Controls")
use_samples = st.sidebar.checkbox("Use sample FASTA files (from sample_seqs)", value=True)
k = st.sidebar.slider("k-mer size (k)", min_value=2, max_value=8, value=3)
run = st.sidebar.button("Run analysis")

uploaded = st.file_uploader("Or upload one or more FASTA files", type=["fa", "fasta", "fas", "fna"], accept_multiple_files=True)

DATA_DIR = Path("sample_seqs")

def read_fasta_handles(handles):
    records = []
    for h, name in handles:
        for rec in SeqIO.parse(h, "fasta"):
            records.append({"id": rec.id, "sequence": str(rec.seq), "source": name})
    return pd.DataFrame(records)

def make_kmers(seq, k=3):
    seq = seq.upper()
    return " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])

if run:
    handles = []
    if use_samples and DATA_DIR.exists():
        for p in DATA_DIR.iterdir():
            if p.suffix.lower() in {".fasta", ".fa", ".fas", ".fna"}:
                handles.append((open(p, "r"), p.stem))
    # uploaded files override or append
    for up in uploaded or []:
        handles.append((up, Path(up.name).stem))

    if len(handles) == 0:
        st.warning("No FASTA inputs found. Enable samples or upload FASTA files.")
    else:
        with st.spinner("Reading sequences..."):
            df = read_fasta_handles(handles)
            if df.empty:
                st.warning("No sequences parsed from provided FASTA files.")
            else:
                df["kmers"] = df["sequence"].apply(lambda s: make_kmers(s, k=k))
                st.subheader("Sequence table")
                st.dataframe(df[["id", "source", "sequence"]].head(200))

                # vectorize and PCA
                st.subheader("PCA of k-mer features")
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(df["kmers"]).toarray()
                n_components = 2
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X)

                plot_df = df.copy()
                plot_df[[f"PC{i+1}" for i in range(n_components)]] = X_pca

                # matplotlib scatter
                fig, ax = plt.subplots(figsize=(6, 5))
                for src in plot_df["source"].unique():
                    subset = plot_df[plot_df["source"] == src]
                    ax.scatter(subset["PC1"], subset["PC2"], label=src, alpha=0.7)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(f"PCA (k={k})")
                ax.legend()
                st.pyplot(fig)

                # download combined CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download combined CSV", data=csv, file_name="combined_sequences.csv", mime="text/csv")

        # close any file objects opened from sample dir
        for h, _ in handles:
            try:
                if hasattr(h, "close"):
                    h.close()
            except Exception:
                pass

else:
    st.info("Choose options in the sidebar and click Run analysis.")

# Footer
st.markdown("---")
st.write("App reads FASTA(s), converts sequences to k-mers, computes PCA, and displays a scatter plot.")
