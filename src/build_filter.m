function [G, G_hat] = build_filter(B, stddev, w, n)
    % boxcar length is twice the B
    box_length = floor(2 * B * 0.8);

    % boxcar
    g_boxcar = zeros(n, 1);
    g_boxcar((n-box_length)/2:(n+box_length)/2) = 1;

    % gaussian
    g_x = (0:n-1) - n/2;
    g_gauss = exp(-g_x.^2 / (2 * stddev * stddev))';

    % convolve the gaussian and boxcar
    g_conv = conv(g_boxcar, g_gauss, 'same');

    % normalize filters
    g_conv = g_conv ./ max(g_conv);

    % center at DC
    g_conv = circshift(g_conv, n/2+1);

    % build the time domain filter centered at t=0:
    % force the frequency response to be Hermitian so the time filter is
    % purely real, then the ifft is automatically circularly centered at
    % sample 0. No fftshift slicing and no offset phase remains.
    g_freq = g_conv;
    g_freq(n/2+2:n) = conj(g_freq(n/2:-1:2));
    G = real(ifft(g_freq, n));

    % truncate symmetrically about 0: keep samples -w/2 .. w/2-1
    half = w/2;
    G = [G(n-half+1:n); G(1:half)];

    % compute the filter response for the estimation step. The estimator
    % divides the vote at position l within a bin's sweep (l = j - c + 33
    % where c = (J-1)*n/B) by conj(G_hat(l)); the needed frequency is
    % j - c = l - 33, so the slice must place frequency 0 at element 33.
    G_hat = fftshift(fft(G, n));
    G_hat = G_hat(n/2 - B/2 + 1 : n/2 + B/2);
end
