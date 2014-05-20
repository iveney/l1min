% Reference: l1 magic document

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
A = @(z) A_fhp(z, OMEGA);
At = @(z) At_fhp(z, OMEGA, n);

% observations
y = A(x);

% min l2 reconstruction (backprojection)
xbp = At(y);
Xbp = reshape(xbp,n,n);
Xbp = mat2gray(Xbp);
% imshow(Xbp);
imwrite(Xbp, 'I2.png');

% min l1
xp = tveq_logbarrier(xbp, A, At, y, 1e-1, 2, 1e-8, 600);
Xtv = reshape(xp, n, n);
Xtv = mat2gray(Xtv);
% imshow(Xtv);
imwrite(Xtv, 'I1.png');