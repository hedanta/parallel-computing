import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

file = "results.txt"
M = 50
p = 0.95

data = np.loadtxt(file)

def min_ci(sorted_arr, I):
    n = len(sorted_arr)
    I = min(I, n)
    min_w = np.inf
    start = 0
    for i in range(n - I + 1):
        w = sorted_arr[i + I - 1] - sorted_arr[i]
        if w < min_w:
            min_w = w
            start = i
    return start, start + I - 1, min_w

for m_idx in range(data.shape[1]):
    arr = data[:, m_idx]

    # normal dist
    avg = np.mean(arr)
    var = np.var(arr, ddof=1)
    std = np.std(arr, ddof=1)
    delta = 1.96 * std / np.sqrt(len(arr))
    ci_full = (avg - delta, avg + delta)

    mask = np.abs(arr - avg) <= 3 * std
    arr1 = arr[mask]
    avg1 = np.mean(arr1)
    std1 = np.std(arr1, ddof=1)
    delta1 = 1.96 * std1 / np.sqrt(len(arr1))
    ci1 = (avg1 - delta1, avg1 + delta1)

    # not normal dist
    counts, edges = np.histogram(arr, bins=M)
    centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(8,6))
    plt.hist(arr, bins=M, edgecolor='black', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.show()

    kde = gaussian_kde(arr)
    xs = np.linspace(arr.min(), arr.max(), 1000)
    ys = kde(xs)
    peaks, props = find_peaks(ys, height=0.01 * max(ys))
    peak_vals = xs[peaks]

    mode_idx = np.argmin(peak_vals)
    mode = peak_vals[mode_idx]
    width = (arr.max() - arr.min()) / M
    arr2 = arr[(arr >= mode - width) & (arr <= mode + width)]

    avg2 = np.mean(arr2)
    sorted2 = np.sort(arr2)
    I2 = max(1, int(np.floor(p * len(sorted2))))
    s_idx, e_idx, _ = min_ci(sorted2, I2)
    ci2 = (sorted2[s_idx], sorted2[e_idx])
    avg2_ci = np.mean(sorted2[s_idx:e_idx+1])

    sorted1 = np.sort(arr1)
    I1 = max(1, int(np.floor(p * len(sorted1))))
    s1, e1, _ = min_ci(sorted1, I1)
    ci1_min = sorted1[s1]
    ci1_max = sorted1[e1]

    sorted_full = np.sort(arr)
    I_full = max(1, int(np.floor(p * len(sorted_full))))
    s_f, e_f, _ = min_ci(sorted_full, I_full)
    ci_full_min = sorted_full[s_f]
    ci_full_max = sorted_full[e_f]

    print("\nНормальное распределение")
    print(f"min = {arr.min():.3f} мс, avg = {avg:.3f} мс, var = {var:.3f}, std = {std:.3f} мс")
    print(f"CI всей выборки: [{ci_full[0]:.3f}, {ci_full[1]:.3f}]")
    print(f"После удаления выбросов 3σ: avg' = {avg1:.3f} ± {delta1:.3f}, CI = [{ci1[0]:.3f}, {ci1[1]:.3f}]")

    print("\nБез предположения о нормальности")
    print(f"Младшая мода: {mode:.3f}")
    print(f"avg'' = {avg2:.3f}, CI = [{ci2[0]:.3f}, {ci2[1]:.3f}], ширина = {ci2[1]-ci2[0]:.3f}")
    print(f"avg''' внутри CI = {avg2_ci:.3f}")

    means = [arr.min(), avg, avg1, avg2_ci]
    labels = ["t_min", "T_avg", "T_avg'", "T_avg'''"]
    errors = [
        0,
        (ci_full_max - ci_full_min)/2,
        (ci1_max - ci1_min)/2,
        (ci2[1] - ci2[0])/2
    ]
    x = range(len(means))
    plt.figure(figsize=(8,6))
    plt.errorbar(x, means, yerr=errors, fmt='o', color='black', ecolor='black', elinewidth=2, capsize=6, markersize=8)
    plt.xticks(x, labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, m in enumerate(means):
        plt.text(i, m + errors[i] + 0.05, f"{m:.2f}", ha='center', va='bottom', fontsize=10)
    plt.show()