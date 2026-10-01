function main()

% Regenerates every figure used in the report. Each section writes one PNG
% into ../images using the same filenames the LaTeX document includes.
% Run from the src directory:  octave-cli --eval "main"
%
% Figure list (report reference in parentheses):
%   gauss_boxcar     - frequency response of boxcar {*} gaussian (fig 1)
%   filter_time      - truncated time domain filter taps (fig 2)
%   filter_freq      - 64 center bins of the truncated filter (fig 3)
%   z_hat            - one location loop's B-point FFT, voted bins marked (fig 4)
%   votes            - per-location votes of the location loop (fig 5)
%   inputdata        - magnitude of the K=6 test signal (fig 6 / table I)
%   FinalResult      - sfft estimate of the test signal (fig 7 / table II)
%   kversusl1        - L1 error vs sparsity k=1..64
%   kversusfar       - false alarm rate vs sparsity k=1..64
%   snrversusl1      - L1 error vs SNR, k=8
%   snrversusfar     - false alarm rate vs SNR, k=10
%   snrversusl1_k1   - L1 error vs SNR, k=1
%   snrversusfar_k1  - false alarm rate vs SNR, k=1

more off;

% output directory next to src
imgdir = fullfile(pwd, '..', 'images');
if ~exist(imgdir, 'dir')
    imgdir = pwd;
end

% fixed parameters shared with sfft.m
n = 4096;
B = 64;
d = 2;
stddev = sqrt(50);
w = 384;
L = ceil(log2(n));

rand('seed', 7);

%% ---------------------------------------------------------------
% test signal from the report: table I
k = 6;
locs = [1 100 2048 2049 2100 4096]';
mags = [10 15 5 15 7 10]';

data_freq = zeros(n, 1);
data_freq(locs) = mags;
data_time = ifft(data_freq) * n;

%% gauss_boxcar: the filter response before truncation
[G, G_hat] = build_filter(B, stddev, w, n);

% full-length frequency response of the untruncated design, centered at DC
box_length = floor(2 * B * 0.8);
g_boxcar = zeros(n, 1);
g_boxcar((n-box_length)/2:(n+box_length)/2) = 1;
g_x = (0:n-1) - n/2;
g_gauss = exp(-g_x.^2 / (2 * stddev * stddev))';
g_conv = conv(g_boxcar, g_gauss, 'same');
g_conv = g_conv ./ max(g_conv);

figure(1); clf;
xdata = (n/2 - 70):(n/2 + 70);
plot(xdata, abs(g_conv(xdata)), 'b-', 'LineWidth', 1);
grid on;
title('Gaussian Convolved with Boxcar Filter, Width=102');
xlabel('Frequency Bins');
ylabel('Magnitude of Frequency Bins');
saveas(gcf, fullfile(imgdir, 'gauss_boxcar.png'));

%% filter_time: truncated time domain filter
figure(2); clf;
plot(0:w-1, real(G), 'b-', 'LineWidth', 1);
grid on;
title('Truncated Time Domain Filter Samples');
xlabel('Time Domain Samples');
ylabel('Real Part of Time Domain Samples');
saveas(gcf, fullfile(imgdir, 'filter_time.png'));

%% filter_freq: 64 center bins of the truncated filter
figure(3); clf;
plot(0:B-1, abs(G_hat), 'b-', 'LineWidth', 1);
grid on;
title('Magnitude of 64 Center Bins of Truncated Filter');
xlabel('Frequency Bins');
ylabel('Frequency Magnitudes');
saveas(gcf, fullfile(imgdir, 'filter_freq.png'));

%% inputdata: the test signal
figure(4); clf;
stem(0:n-1, abs(data_freq), 'b', 'Marker', 'none');
hold on;
for i = 1:k
    text(locs(i) - 40, mags(i) + 0.6, sprintf('%.0f', mags(i)), 'Color', 'k');
end
hold off;
grid on;
title('Magnitude of Input Sparse Signal, K=6');
xlabel('Frequency Bins');
ylabel('Frequency Magnitudes');
xlim([0 n]);
saveas(gcf, fullfile(imgdir, 'inputdata.png'));

%% sfft on the test signal; keep diagnostics for z_hat and votes figures
[sDFT, diag] = sfft(data_time, k);

%% z_hat: one location loop iteration's B-point FFT and the voted bins
% use the first location loop iteration
z_hat_r = diag.z_hats(:, 1);
J_r = diag.Js(:, 1);

figure(5); clf;
plot(0:B-1, abs(z_hat_r), 'b-', 'LineWidth', 1);
hold on;
plot(J_r - 1, abs(z_hat_r(J_r)), 'r*', 'MarkerSize', 8);
hold off;
grid on;
title('Magnitudes of Subsampled FFT and Top dk Points Voted');
xlabel('Frequency Bins');
ylabel('Frequency Magnitudes');
xlim([0 B]);
saveas(gcf, fullfile(imgdir, 'z_hat.png'));

%% votes: per-location vote counts from the location loop
figure(6); clf;
stem(0:n-1, diag.vote_counts, 'b', 'Marker', 'none');
hold on;
% mark locations that passed the threshold
winners = find(diag.vote_counts > L/2);
stem(winners - 1, diag.vote_counts(winners), 'ro', 'MarkerSize', 4);
hold off;
grid on;
title('Location Loop Votes for K=6');
xlabel('Frequency Bins');
ylabel('Votes');
xlim([0 n]);
ylim([0 L + 1]);
saveas(gcf, fullfile(imgdir, 'votes.png'));

%% FinalResult: the sfft estimate
figure(7); clf;
stem(0:n-1, abs(sDFT), 'b', 'Marker', 'none');
hold on;
est_locs = find(abs(sDFT) > 0);
for i = 1:numel(est_locs)
    text(est_locs(i) - 40, abs(sDFT(est_locs(i))) + 0.6, ...
        sprintf('%.4f', abs(sDFT(est_locs(i)))), 'Color', 'k');
end
hold off;
grid on;
title('Algorithm Magnitudes Output');
xlabel('Frequency Bins');
ylabel('Frequency Magnitudes');
xlim([0 n]);
saveas(gcf, fullfile(imgdir, 'FinalResult.png'));

%% kversusl1 / kversusfar: error and false alarm vs sparsity
rand('seed', 42);
maxk = 64;
ERRK = zeros(maxk, 1);
FARK = zeros(maxk, 1);
for kk = 1:maxk
    truth = zeros(n, 1);
    truth(randperm(n, kk)') = 1;
    x = ifft(truth) * n;
    est = sfft(x, kk);

    ERRK(kk) = sum(abs(est - truth)) / kk;

    tp = sum(abs(est) > 0 & truth > 0);
    fp = sum(abs(est) > 0 & truth == 0);
    tn = sum(abs(est) == 0 & truth == 0);
    FARK(kk) = fp / max(fp + tn, 1);
end

figure(8); clf;
plot(1:maxk, ERRK, 'b-', 'LineWidth', 1);
grid on;
title('L1 Error Versus Signal Sparsity');
xlabel('k-sparsity of signal');
ylabel('L1 Error');
saveas(gcf, fullfile(imgdir, 'kversusl1.png'));

figure(9); clf;
plot(1:maxk, FARK * 1e3, 'b-', 'LineWidth', 1);
grid on;
title('False Alarm Rate Versus K Sparsity');
xlabel('k-sparsity of signal');
ylabel('False Alarm Rate');
saveas(gcf, fullfile(imgdir, 'kversusfar.png'));

%% snrversusl1: L1 vs SNR at k=8
rand('seed', 42);
snrs = -20:5:100;
ERRS = zeros(numel(snrs), 1);
k8 = 8;
truth8 = zeros(n, 1);
truth8(randperm(n, k8)') = 1;
x8 = ifft(truth8) * n;
for i = 1:numel(snrs)
    [xn, scaled, ~] = addNoise(x8, snrs(i));
    est = abs(sfft(xn, k8));
    est = est / abs(scaled(1) / x8(1));
    ERRS(i) = sum(abs(est - truth8)) / k8;
end

figure(10); clf;
semilogy(snrs, ERRS, 'b-', 'LineWidth', 1);
grid on;
title('L1 Error Versus SNR, k=8');
xlabel('SNR (db)');
ylabel('L1 Error');
saveas(gcf, fullfile(imgdir, 'snrversusl1.png'));

%% snrversusfar: false alarm vs SNR at k=10
rand('seed', 42);
FARS = zeros(numel(snrs), 1);
k10 = 10;
truth10 = zeros(n, 1);
truth10(randperm(n, k10)') = 1;
x10 = ifft(truth10) * n;
for i = 1:numel(snrs)
    [xn, scaled, ~] = addNoise(x10, snrs(i));
    est = sfft(xn, k10);
    est = est / abs(scaled(1) / x10(1));
    a = abs(est);
    fp = sum(a > 0 & truth10 == 0);
    tn = sum(a == 0 & truth10 == 0);
    FARS(i) = fp / max(fp + tn, 1);
end

figure(11); clf;
plot(snrs, FARS * 1e3, 'b-', 'LineWidth', 1);
grid on;
title('False Alarm Rate Versus SNR, K=10');
xlabel('SNR (db)');
ylabel('False Alarm Rate');
saveas(gcf, fullfile(imgdir, 'snrversusfar.png'));

%% snrversusl1_k1: L1 vs SNR at k=1
rand('seed', 42);
snrs1 = -40:4:120;
ERR1 = zeros(numel(snrs1), 1);
truth1 = zeros(n, 1);
truth1(randi(n)) = 1;
x1 = ifft(truth1) * n;
for i = 1:numel(snrs1)
    [xn, scaled, ~] = addNoise(x1, snrs1(i));
    est = abs(sfft(xn, 1));
    est = est / abs(scaled(1) / x1(1));
    ERR1(i) = sum(abs(est - truth1)) / 1;
end

figure(12); clf;
plot(snrs1, ERR1, 'b-', 'LineWidth', 1);
grid on;
title('L1 Error Versus SNR, k=1');
xlabel('SNR (db)');
ylabel('L1 Error');
saveas(gcf, fullfile(imgdir, 'snrversusl1_k1.png'));

%% snrversusfar_k1: false alarm vs SNR at k=1
rand('seed', 42);
FAR1 = zeros(numel(snrs1), 1);
for i = 1:numel(snrs1)
    [xn, scaled, ~] = addNoise(x1, snrs1(i));
    est = sfft(xn, 1);
    est = est / abs(scaled(1) / x1(1));
    a = abs(est);
    fp = sum(a > 0 & truth1 == 0);
    tn = sum(a == 0 & truth1 == 0);
    FAR1(i) = fp / max(fp + tn, 1);
end

figure(13); clf;
plot(snrs1, FAR1 * 1e4, 'b-', 'LineWidth', 1);
grid on;
title('False Alarm Rate Versus SNR, K=1');
xlabel('SNR (db)');
ylabel('False Alarm Rate');
saveas(gcf, fullfile(imgdir, 'snrversusfar_k1.png'));

disp('main: all figures written');

end
