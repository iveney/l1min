% Compare Ordinary Least square (no regularization), L2-reguarlized (Ridge),
% L1-regualarized (Lasso) regression in finding the sparse coefficient
% in a underdetermined linear system

rng(0);  % for reproducibility
m = 50;  % num samples
n = 200; % num variables, note that n > m

A = rand(m, n);
x = zeros(n, 1);
nz = 10; % 10 non-zeros variables (sparse)
nz_idx = randperm(n);
x(nz_idx(1:nz)) = 3 * rand(nz, 1);
y = A*x;
y = y + 0.05 * rand(m, 1); % add some noise

% plot original x
subplot(2, 2, 1);
bar(x), axis tight;
title('Original coefficients');

% OLS
x_ols = A \ y;
subplot(2, 2, 2);
bar(x_ols), axis tight;
title('Ordinary Least Square');
y_ols = A * x_ols;

% L2 (Ridge) 
x_l2 = ridge(y, A, 1e-5, 0);  % last parameter = 00 to generate intercept term
b_l2 = x_l2(1);
x_l2 = x_l2(2:end);
subplot(2, 2, 3);
bar(x_l2), axis tight;
title('L2 Regularization');
y_l2 = A * x_l2 + b_l2;

% L1 (Lasso)
[x_l1, fitinfo] = lasso(A, y, 'Lambda', 0.1);
b_l1 = fitinfo.Intercept(1);
y_l1 = A * x_l1 + b_l1;
subplot(2, 2, 4);
bar(x_l1), axis tight;
title('L1 Regularization');

% L1 (Elastic Net)
[x_en, fitinfo_en] = lasso(A, y, 'Lambda', 0.1, 'Alpha', 0.7);
b_en = fitinfo_en.Intercept(1);
y_en = A * x_en + b_en;

MSE_y = [mse(y_ols-y), mse(y_l2-y), mse(y_l1-y), mse(y_en-y)];
disp('Mean square error: ')
fprintf('%g    ', MSE_y); fprintf('\n\n');

% Plot the recovered coefficients
figure, hold on
plot(x_l1, 'b');
plot(x_en, 'r');
plot(x, 'g--');
legend('Lasso Coef', 'Elastic Net coef', 'Original Coef');