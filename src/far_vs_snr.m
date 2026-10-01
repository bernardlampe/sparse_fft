function [SNRDB, FAR, FP, FN] = far_vs_snr(k, lowSNR, highSNR, snrStep)

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
FAR = zeros(length(snrs), 1);
FP = zeros(length(snrs), 1);
FN = zeros(length(snrs), 1);
SNRDB = zeros(length(snrs), 1);

% loop over k
for i = 1:length(snrs)
    % create complex noise
    [signalWithNoise, scaledSignal, snr] = addNoise(data_time, snrs(i));

    % run sparse fft
    dft = sfft(signalWithNoise, k);

    % divide out the signal scaling applied by addNoise
    scale = scaledSignal(1) / data_time(1);
    dft = dft / abs(scale);

    % SNR
    SNRDB(i) = snr;

    fp = 0;
    tp = 0;
    tn = 0;
    fn = 0;
    for j = 1:n
        mdft = dft(j);
        df = data_freq(j);
        if mdft == 0 && df == 0
            tn = tn + 1;
        elseif mdft > 0 && df == 0
            % false detection
            fp = fp + 1;
        elseif mdft > 0 && df > 0
            tp = tp + 1;
        else
            fn = fn + 1;
        end
    end

    FAR(i) = fp / (fp + tn);
    FP(i) = fp;
    FN(i) = fn;
end

end