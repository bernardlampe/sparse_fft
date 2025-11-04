import numpy as np

class SparseFFT:
    """
    A practical Sparse FFT implementation based on Hassanieh et al. (2012).
    Complexity: O(k log N log (N/k))
    """

    def __init__(self, N, k, num_hashes=4, num_bins=None):
        self.N = N
        self.k = k
        self.num_hashes = num_hashes
        self.num_bins = num_bins or (1 << (k - 1).bit_length())

    def _permute(self, x, a, b):
        """Apply random affine permutation and modulation."""
        N = len(x)
        n = np.arange(N)
        idx = (a * n + b) % N
        phase = np.exp(2j * np.pi * n * np.random.rand())
        return x[idx] * phase

    def _hash_and_filter(self, x, a, b):
        """Hash the signal into bins and compute FFT."""
        permuted = self._permute(x, a, b)
        subsampled = permuted[::max(1, self.N // self.num_bins)]
        return np.fft.fft(subsampled, self.num_bins)

    def _estimate_frequencies(self, bins_list):
        """Estimate frequencies by finding peaks across multiple hashes."""
        magnitudes = np.stack([np.abs(b) for b in bins_list])
        avg_mag = np.median(magnitudes, axis=0)
        top_bins = np.argpartition(avg_mag, -self.k)[-self.k:]

        freqs = []
        coeffs = []
        for b_idx in top_bins:
            values = [bins[b_idx] for bins in bins_list]
            coeff = np.median(values)
            freqs.append(b_idx * (self.N // self.num_bins))
            coeffs.append(coeff)

        return np.array(freqs) % self.N, np.array(coeffs)

    def fit(self, x):
        """Compute the Sparse FFT estimate of x."""
        hashes = []
        for _ in range(self.num_hashes):
            a = np.random.randint(1, self.N)
            b = np.random.randint(0, self.N)
            bins = self._hash_and_filter(x, a, b)
            hashes.append(bins)

        freqs, coeffs = self._estimate_frequencies(hashes)
        return freqs, coeffs


# --- Example Usage ---
if __name__ == "__main__":
    N = 8192
    k = 10
    np.random.seed(42)
    
    # Generate k-sparse signal
    freqs_true = np.random.choice(N // 2, k, replace=False)
    coeffs_true = np.random.randn(k) + 1j * np.random.randn(k)
    x = np.zeros(N, dtype=complex)
    for f, c in zip(freqs_true, coeffs_true):
        x += c * np.exp(2j * np.pi * f * np.arange(N) / N)
    
    # Run Sparse FFT
    sfft = SparseFFT(N=N, k=k)
    freqs_est, coeffs_est = sfft.fit(x)

    print("True Frequencies:", sorted(freqs_true))
    print("Estimated Frequencies:", sorted(freqs_est))
