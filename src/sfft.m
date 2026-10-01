function [sDFT, diagnostics] = sfft(data, k)

% Sparse FFT implementation. The sDFT return variable is the estimated
% k complex values embedded in a n-length vector of n-k zeros.
%
% Requires n divisible by 64. Reliable for k up to about B/4 = 16;
% location accuracy degrades as k approaches B because the d*k selected
% bins cover a growing fraction of [n] and true locations lose their
% vote-count majority. Raise B (a power of two dividing n) for larger k.

% length of data
n = length(data);

% heuristic size of frequency bin to divide n
B = 64;

% heuristic amount of oversample when finding peaks
d = 2;

% heuristic filter parameters tuned for B = 64 and n = 4096
stddev = sqrt(50);
w = 384;

% input validation: the remap and filter geometry require these
assert(mod(n, B) == 0, 'sfft: n (%d) must be divisible by B (64)', n);
assert(mod(n, 2) == 0, 'sfft: n (%d) must be even', n);
assert(k >= 1 && k <= B, 'sfft: k (%d) must be in 1..B (%d)', k, B);
assert(w <= n, 'sfft: filter length w (%d) must be <= n (%d)', w, n);

% heuristic number of times the outer loop runs
L = ceil(log2(n));

% filter function return time domain filter and max value of G_hat
[G, G_hat] = build_filter(B, stddev, w, n);

% 4.2 Outer loop begin

% 4.2.1 Run location loop L times
Z = zeros(L, n);
z_hats = zeros(B, L);
Js = zeros(min(d*k, B), L);
for r = 1:L
    % 4.2.1 get the locations
    [J, z_hat, sigma_inv] = location_loop(data, n, G, w, B, d, k);
    z_hats(:, r) = z_hat;
    Js(:, r) = J;
    [I, z_vals] = remap_locations(d, k, n, B, sigma_inv, J, z_hat, G_hat);
    Z(r, I) = z_vals;
end

% 4.2.2 Count the occurences of each location
bin_counts = sum(Z ~= 0);
bin_counts = bin_counts(:);

% 4.2.3 Keep locations with more than L/2 occurrences (paper's rule).
% Candidates are ranked by vote count first; ties in the vote count are
% broken by the magnitude of the median estimate, so at most k locations
% survive when the threshold passes too many.
candidates = find(bin_counts > L/2);
if isempty(candidates)
    % no location passed the threshold; fall back to top-k by vote count
    [~, candidates] = sort(bin_counts, 'descend');
    candidates = candidates(1:k);
elseif numel(candidates) > k
    % rank by vote count, then by median estimate magnitude
    mags = zeros(numel(candidates), 1);
    for c = 1:numel(candidates)
        col = Z(:, candidates(c));
        col = col(col ~= 0);
        mags(c) = abs(median(col));
    end
    [~, order] = sortrows([-bin_counts(candidates(:)), -mags], [1, 2]);
    candidates = candidates(order(1:k));
end
bin_inds = candidates;

% diagnostics: per-location vote counts, the per-run subsampled FFTs with
% their voted bins, the vote matrix used for the estimation medians, and
% the loop parameters
diagnostics = struct('vote_counts', bin_counts, 'Z', Z, ...
    'z_hats', z_hats, 'Js', Js, 'L', L, 'B', B, 'd', d);

% 4.2.4 Run estimation loop L times on location that have > L/2 occurrences
% 4.2.5 Use median of complex values as estimate at each location
sDFT = zeros(n, 1);
for kk = 1:length(bin_inds)
    b = bin_inds(kk);

    % median over rows that actually voted for this location
    col = Z(:, b);
    col = col(col ~= 0);
    if isempty(col)
        continue;
    end

    sDFT(b) = median(real(col)) + 1i * median(imag(col));
end

end

% 4.1 Inner loop - remap each value in J to [n]
function [I, Z] = remap_locations(d, k, n, B, sigma_inv, J, z_hat, G_hat)
    I = zeros(d*k*n/B, 1);
    Z = zeros(d*k*n/B, 1);

    m = 1;
    for i = 1:length(J)
        % bin J(i) covers the n/B frequency slots starting half a bin below
        % the bin center: j = (J(i) - 1.5) * n/B + t, t = 0..n/B-1
        low = mod(round((J(i) - 1.5) * n / B), n);

        % each index in J maps to n/B indicies in [n]
        for t = 0:n/B-1
            j = mod(low + t, n);

            % transform the index back
            I(m) = mod(j * sigma_inv, n) + 1;

            % compute the x estimate value
            % the alias-folded tone contributes c * conj(G_hat(l)), so the
            % estimate needs the conjugate of the filter response
            Z(m) = z_hat(J(i)) ./ conj(G_hat(t + 1));

            m = m + 1;
        end
    end

    % trim to the number of votes actually written
    I = I(1:m-1);
    Z = Z(1:m-1);
end

% 4.1 Inner loop
function [J, z_hat, sigma_inv] = location_loop(x, n, G, w, B, d, k)
    % 4.1.1 choose a random sigma \in [n], with sigma odd and relatively prime
    sigma = 0;
    while gcd(sigma, n) ~= 1
        sigma = randi([0 n], 1);
    end

    % get sigma inverse
    [~, sigma_inv] = gcd(sigma, n);
    sigma_inv = mod(sigma_inv, n);

    % check that we have a valid sigma
    assert(mod(sigma * sigma_inv, n) == 1, 'sigma not relatively prime with n');

    % 4.1.2 permute the spectrum y_i = (x_i * sigma) % n
    y = zeros(w, 1);
    for i = 0:w-1
        y(i+1) = x(mod(i * sigma, n) + 1);
    end

    % 4.1.2 multiply by the window function
    y = y .* G;

    % 4.1.3 compute the B-point DFT z_hat from y
    z = zeros(B, 1);
    for i = 0:w-1
        ind = mod(i, B);
        z(ind + 1) = z(ind + 1) + y(i + 1);
    end
    z_hat = fft(z, B);

    % find dk indicies in B with largest magnitude
    [~, J] = sort(abs(z_hat), 'descend');
    J = J(1:min(d*k, B));
end

