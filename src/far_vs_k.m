function [FAR, FP, FN] = far_vs_k(maxk)

% size of data
n = 4096;

% allocate
FAR = zeros(maxk, 1);
FP = zeros(maxk, 1);
FN = zeros(maxk, 1);

% loop over k
for k = 1:maxk
    % create data set with k distinct locations
    data_freq = zeros(n, 1);
    data_freq(randperm(n, k)') = 1;


    % create time domain data
    data_time = ifft(data_freq) .* n;

    % run sparse fft
    dft = sfft(data_time, k);

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

    FAR(k) = fp / (fp + tn);
    FP(k) = fp;
    FN(k) = fn;
end

end
