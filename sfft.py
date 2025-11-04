import numpy as np
import matplotlib.pyplot as plt


def sfft(X_sparse, N, K, B=256):

    # Hashing: randomly permute frequency indices and bin into buckets
    perm = np.random.permutation(N)
    bucket_indices = perm % B

    # Filter: apply a flat window (can be replaced with more sophisticated filters)
    window = np.ones(N)

    # Aliased signal: simulate subsampling in time domain
    time_indices = np.arange(0, N, N // B)
    aliased_signal = np.zeros(B, dtype=complex)

    for b in range(B):
        for i in range(N):
            if bucket_indices[i] == b:
                aliased_signal[b] += X_sparse[i] * np.exp(2j * np.pi * i * time_indices[b] / N)

    # Estimate frequencies: identify top buckets and recover frequency indices
    estimated_freqs = []
    estimated_values = []

    threshold = np.percentile(np.abs(aliased_signal), 100 * (1 - K / B))
    for b in range(B):
        if np.abs(aliased_signal[b]) >= threshold:
            # Estimate frequency index from bucket
            candidates = np.where(bucket_indices == b)[0]
            # Choose the candidate with highest magnitude in original sparse signal
            best = max(candidates, key=lambda i: np.abs(X_sparse[i]))
            estimated_freqs.append(best)
            estimated_values.append(X_sparse[best])

    return estimated_values, estimated_freqs


# Parameters
N = 4096 # Signal length
K = 20                # Sparsity (number of non-zero frequencies)

# Generate a synthetic sparse signal in frequency domain
freq_indices = np.random.choice(N, K, replace=False)
freq_values = np.random.randn(K) + 1j * np.random.randn(K)

# Create sparse frequency domain signal
X_sparse = np.zeros(N, dtype=complex)
X_sparse[freq_indices] = freq_values

estimated_values, estimated_freqs = sfft(X_sparse, N, K)

# Display results
print("Original Frequencies:", sorted(freq_indices))
print("Estimated Frequencies:", sorted(estimated_freqs))

# Plot frequency domain comparison
plt.figure(figsize=(10, 5))
plt.stem(freq_indices, np.abs(X_sparse[freq_indices]), linefmt='b-', markerfmt='bo', basefmt=' ')
plt.stem(estimated_freqs, np.abs(estimated_values), linefmt='r--', markerfmt='rx', basefmt=' ')
plt.title("Original vs Estimated Sparse Frequencies")
plt.xlabel("Frequency Index")
plt.ylabel("Magnitude")
plt.legend(["Original", "Estimated"])
plt.grid(True)
plt.tight_layout()
plt.savefig("sparse_fft_comparison.png")
plt.show()