function [SNRDB, ERR, data_time] = l1_vs_snr(k, lowSNR, highSNR, snrStep)

% size of data
n = 4096;

% create data set
data_freq = zeros(n, 1);

% create k magnitudes at distinct locations
data_freq(randperm(n, k)') = 1;

% create time domain data
data_time = ifft(data_freq) .* n;

% SNRS
snrs = lowSNR:snrStep:highSNR;

% allocate
ERR = zeros(length(snrs), 1);
SNRDB = zeros(length(snrs), 1);

% loop over k
for i = 1:length(snrs)
    % create complex noise
    [signalWithNoise, scaledSignal, snr] = addNoise(data_time, snrs(i));

    % run sparse fft on the noisy scaled signal
    dft = abs(sfft(signalWithNoise, k));

    % the estimate is of the scaled signal; divide by the known scale so
    % it is comparable to the unscaled data_freq
    scale = scaledSignal(1) / data_time(1);
    dft = dft / abs(scale);

    % compute l1 error
    e = sum(abs(dft - data_freq)) ./ k;
    ERR(i) = e;

    % SNR
    SNRDB(i) = snr;
end

end
