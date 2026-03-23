import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    df = pd.read_csv(path, sep=r"\s+")
    return df

dbg = load_data("all_algorithms_dbg.txt")
rel = load_data("all_algorithms_rel.txt")

best_dbg = load_data("best_results_dbg.txt")
best_rel = load_data("best_results_rel.txt")


def plot_classic(df_rel, df_dbg):
    Ns = sorted(df_rel["N"].unique())
    
    classic_rel = []
    classic_dbg = []
    
    for N in Ns:
        subset_rel = df_rel[
            (df_rel["algo"].str.contains("classic")) &
            (df_rel["N"] == N)
        ]
        
        subset_dbg = df_dbg[
            (df_dbg["algo"].str.contains("classic")) &
            (df_dbg["N"] == N)
        ]
        
        classic_rel.append(subset_rel["GFLOPS"].values[0])
        classic_dbg.append(subset_dbg["GFLOPS"].values[0])
    
    plt.figure()
    plt.plot(Ns, classic_rel, marker='o', label="classic (Release)")
    plt.plot(Ns, classic_dbg, marker='o', label="classic (Debug)")
    
    plt.xticks(Ns)
    
    plt.xlabel("N")
    plt.ylabel("GFLOPS")
    plt.title("Performance (Release vs Debug)")
    plt.legend()
    plt.grid()
    plt.show()
    

def plot_transposed(df):
    Ns = sorted(df["N"].unique())
    
    transpose = []
    transpose_full = []
    
    for N in Ns:
        subset_tr = df[
            (df["algo"].str.contains("transpose")) &
            (df["N"] == N)
        ]
        
        subset_tr_full = df[
            (df["algo"].str.contains("transpose_full")) &
            (df["N"] == N)
        ]
        
        transpose.append(subset_tr["GFLOPS"].values[0])
        transpose_full.append(subset_tr_full["GFLOPS"].values[0])
    
    plt.figure()
    plt.plot(Ns, transpose, marker='o', label="transpose")
    plt.plot(Ns, transpose_full, marker='o', label="transpose_full")
    
    plt.xticks(Ns)
    
    plt.xlabel("N")
    plt.ylabel("GFLOPS")
    plt.title("Performance (w/o vs w transposing time)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_buffer_M(df, N):
    subset = df[(df["algo"].str.contains("buffer_unroll")) & (df["N"] == N)].copy()
    
    subset["M"] = subset["algo"].str.extract(r"M(\d+)").astype(int)
    subset = subset.sort_values("M")
    
    plt.figure()
    plt.plot(subset["M"], subset["GFLOPS"], marker='o')
    plt.title(f"Buffer + Unroll: Preal vs M (N={N})")
    plt.xlabel("M")
    plt.ylabel("GFLOPS")
    plt.grid()
    plt.show()


def plot_block_S(df, N):
    subset = df[(df["algo"].str.contains("block_S")) & (~df["algo"].str.contains("unroll")) & (df["N"] == N)].copy()
    
    subset["S"] = subset["algo"].str.extract(r"S(\d+)").astype(int)
    subset = subset.sort_values("S")
    
    plt.figure()
    plt.plot(subset["S"], subset["GFLOPS"], marker='o')

    plt.xticks(subset["S"])
    plt.title(f"Block: Preal vs S (N={N})")
    plt.xlabel("S")
    plt.ylabel("GFLOPS")
    plt.grid()
    plt.show()


def plot_block_unroll(df, N, S_fixed):
    subset = df[(df["algo"].str.contains(f"block_unroll_S{S_fixed}_")) & (df["N"] == N)].copy()
    
    subset["M"] = subset["algo"].str.extract(r"M(\d+)").astype(int)
    subset = subset.sort_values("M")
    
    plt.figure()
    plt.plot(subset["M"], subset["GFLOPS"], marker='o')
    plt.title(f"Block + Unroll: Preal vs M (N={N}, S={S_fixed})")
    plt.xlabel("M")
    plt.ylabel("GFLOPS")
    plt.grid()
    plt.show()


def plot_top4(df):
    Ns = sorted(df["N"].unique())
    
    classic = []
    transpose = []
    buffer = []
    block = []
    
    for N in Ns:
        subset = df[df["N"] == N]
        
        classic.append(subset[subset["algo"] == "classic"]["GFLOPS"].values[0])
        transpose.append(subset[subset["algo"] == "transpose"]["GFLOPS"].values[0])
        
        buffer.append(subset[subset["algo"] == "buffer_unroll_M16"]["GFLOPS"].values[0])
        
        block_subset = subset[
            subset["algo"].str.contains("block_S") &
            ~subset["algo"].str.contains("unroll")
        ]
        best_block = block_subset.loc[block_subset["GFLOPS"].idxmax()]
        block.append(best_block["GFLOPS"])
    
    plt.figure()
    plt.plot(Ns, classic, marker='o', label="classic")
    plt.plot(Ns, transpose, marker='o', label="transpose")
    plt.plot(Ns, buffer, marker='o', label="buffer M=16")
    plt.plot(Ns, block, marker='o', label="block (best S)")
    
    plt.xticks(Ns)
    
    plt.xlabel("N")
    plt.ylabel("GFLOPS")
    plt.title("Performance vs Matrix Size")
    plt.legend()
    plt.grid()
    plt.show()


plot_classic(rel, dbg)
plot_transposed(rel)
plot_buffer_M(rel, 2048)
plot_block_S(rel, 2048)
plot_block_unroll(rel, 2048, S_fixed=32)
plot_top4(rel)