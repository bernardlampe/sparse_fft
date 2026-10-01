function [ERR, data_time] = l1_vs_k(maxk)

% size of data
n = 4096;

% allocate
ERR = zeros(maxk, 1);

% loop over k
for k = 1:maxk
    % create data set with k distinct locations
    data_freq = zeros(n, 1);
    data_freq(randperm(n, k)') = 1;

    
    % create time domain data
    data_time = ifft(data_freq) .* n;

    % run sparse fft
    dft = sfft(data_time, k);
    
    % compute l1 error
    e = sum(abs(dft - data_freq)) ./ k;

    ERR(k) = e;
end

end
