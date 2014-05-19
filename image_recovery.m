addpath ../l1magic/Measurements
addpath ../l1magic/Data

I = imread('cameraman.tif');
n = size(I, 1);
N = n*n;

% number of radial lines in the Fourier domain
L = 22;

% Fourier samples (upper half plane)
[M, Mh, mh, mhi] = LineMask(L, n);
OMEGA = mhi;
K = length(OMEGA);

% Measurement matrix size = length(OMEGA) x (n*n)
A = @(z) A_fhp(z, OMEGA);
At = @(z) At_fhp(z, OMEGA, n);

% coefficients
coef = fft2(I);

% Measurement matrix (hstack real and imag of half plane)
[I, J] = ind2sub([K, N] , OMEGA);
Mreal = sparse(I, J, sqrt(2) * real(OMEGA), K, N);
Mimag = sparse(I, J, sqrt(2) * imag(OMEGA), K, N);
Mcomb = sparse(K*2, N*2);
Mcomb(1:K, 1:N) = Mreal;
Mcomb(K+1:end, N+1:end) = Mimag;

% observations
x = sqrt(2) * [real(coef(:)); imag(coef(:))];
b = Mcomb * x;

[x1] = lasso(Mcomb, b, 'Lambda', 1e-2);

% reconstruct image
c1 = x1(1:N) + x1(N+1:end) * i;
c1 = reshape(c1, n, n);
I1 = ifft2(c1);
figure, imshow(I1);