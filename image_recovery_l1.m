% Reference: l1 magic document

addpath ../l1magic/Measurements
addpath ../l1magic/Optimization

IM = imread('cameraman.tif');
n = size(IM, 1);
N = n*n;

% Uncomment the following to use the phantom image come with l1magic
% addpath ../l1magic/Data
% IM = phantom(n);

% number of radial lines in the Fourier domain
L = 22;

% Fourier samples (upper half plane)
[M, Mh, mh, mhi] = LineMask(L, n);
OMEGA = mhi;        % mhi is the non-zero index in Mh (half plane)
K = length(OMEGA);  % it is the linear index of the radial lines in Mh

% Uncomment the following to use random measurement matrix
% K = 3000;
% OMEGA = randperm(N);
% OMEGA = OMEGA(1:K)';

% Returns the observation taken on z using fourier samples OMEGA
A = @(z) A_fhp(z, OMEGA);

% Reconstruct image given the observations z using fourier samples OMEGA
At = @(z) At_fhp(z, OMEGA, n);

% observations
y = A(IM(:));

% min l2 reconstruction (backprojection)
xbp = At(y);
Xbp = reshape(xbp,n,n);
Xbp = mat2gray(Xbp);
% imshow(Xbp);
imwrite(Xbp, 'I2.png');

% min l1 reconstruction (min TV)
xp = tveq_logbarrier(xbp, A, At, y, 1e-1, 2, 1e-8, 600);
Xtv = reshape(xp, n, n);
Xtv = mat2gray(Xtv);
% imshow(Xtv);
imwrite(Xtv, 'I1.png');