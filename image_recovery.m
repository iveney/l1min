addpath ../l1magic/Measurements
addpath ../l1magic/Data

IM = imread('cameraman.tif');
n = size(IM, 1);
N = n*n;

% number of radial lines in the Fourier domain
L = 22;

% Fourier samples (upper half plane)
[M, Mh, mh, mhi] = LineMask(L, n);
OMEGA = mhi;
K = length(OMEGA);

% Measurement matrix size = length(OMEGA) x (n*n)
% A = @(z) A_fhp(z, OMEGA);
% At = @(z) At_fhp(z, OMEGA, n);

% coefficients
coef = fft2(IM); % explain why 1/n
creal = real(coef);
cimag = imag(coef);

% make it 1D
x = [coef(1, 1); creal(:); cimag(:)];
assert(length(x) == 2*N+1);

% observations
b = [coef(1, 1); sqrt(2) * creal(OMEGA); sqrt(2) * cimag(OMEGA)] / n;
% assert(length(b) == 2*K+1);
% assert(isequal(b, A_fhp(IM(:), OMEGA)));

% Measurement matrix (hstack real and imag of half plane)
IND = sparse(1:K, OMEGA, sqrt(2), K, N);
A = sparse(K*2+1, N*2+1);
A(1,1) = 1;
A(2:K+1, 2:N+1) = IND;
A(K+2:end, N+2:end) = IND;
% assert(isequal(b, A*x/n));

% min energy reconstruction
x2 = A \ b;
c2 = sqrt(2)*reshape(x2(2:N+1) + i*x2(N+2:end), n, n);
c2(1,1) = b(1);
I2 = real(n*ifft2(c2));
I2 = mat2gray(I2);       % renormalize
% figure, imshow(I2);
% title('Min energy reconstruction');
imwrite(I2, 'I2.png');


% l1 reconstruction
[x1] = lasso(Mcomb, b, 'Lambda', 1e-2);
c1 = sqrt(2)*reshape(x1(2:N+1) + i*x1(N+2:end), n, n);
c1(1,1) = b(1);
I1 = real(n*ifft2(c1));
% figure, imshow(I1);
% title('L1 reconstruction');
imwrite(I1, 'I1.png');