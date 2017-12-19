M = 300;
K = M; %300:-10:10; % 200;
runs = 1;
mse_orth = zeros(size(K, 2), runs);
mse_direct = zeros(size(K, 2), runs);
mse_b = zeros(size(K, 2), runs);
miss_class_runs = zeros(size(K, 2), runs);
miss_class_admm_runs = zeros(size(K, 2), runs);

edges = [50, 120, 170, 192, 220, 244, 256, 300] ;
levels = [400,  0 , 300, 0, 0, 0, 800, 0];
idxs = zeros(1, M)  ;
idxs(edges(1: end-1)+1) = 1 ;
g = levels(cumsum(idxs)+1);

F = LehmerMatrix(M);
[L, U] = lu(F);
I = eye(M);
D = inv(L);

h = cumsum(g)';

A = normrnd(0, 1/(K), [K, M]);

y = A*g';

estimate = smashed_filt_estimate(y, M, K, A, L, F, 35)

plot(estimate)

figure
plot(g)


