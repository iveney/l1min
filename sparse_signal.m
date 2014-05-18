% test l1 sparse signal recovery

% for reproducibility
rng(0);

% num FFT components
N = 256;
% Non-zero components
nz = 10;
nz_idx = randperm(N);

% construct the sparse fft coefficient
coef = zeros(256, 1);
coef(nz_idx(1:nz)) = rand(nz, 1) - 0.5;

% plot the coefficient
subplot(2, 2, 1);
stem(coef);
title('Signal in frequency domain');
xlim([0, N]);

% generate the signal in time domain
x = ifft(coef);
% genearte samples
Nsample = 80;
sample_idx = randi(length(x), Nsample, 1);
samples = zeros(length(x), 1);
samples(sample_idx) = x(sample_idx);
subplot(2, 2, 2);
hold on;
stem(find(x~=0), x(x~=0), 'b');
stem(sample_idx, x(sample_idx), 'r');
% plot(1:N, x);
title('Signal in time domain');
legend('Signal', 'Sample')
xlim([0, N]);

% measurement matrix
A = randn(Nsample,N);
A = orth(A')';

% observations
y = A*coef;