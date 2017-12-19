clear all;
close all;

M = 1000;
K = M/4;
Ks = [M, M/2, M/4, M/5, M/10, M/50, M/100];
edges =  [0, 50, 150, 250, 350, 450, 500, 520, 600, 780, 840, 900, 1000];
levels = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0];
%levels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
idxs = zeros(1, M)  ;
idxs(edges(1: end-1)+1) = 1 ;
g = levels(cumsum(idxs)+1);

%g = data;

F = LehmerMatrix(M);
[L, U] = lu(F);
I = eye(M);
D = inv(L);
a = D*g';
inds = find(abs(a));
h = cumsum(g)';
runs = 1;

correct_inds = zeros(1, runs);
sigmas = [0.01, 0.2, 0.5, 1, 2, 5];
pv = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
X_all = cell([length(sigmas) length(Ks)]);
Y_all = cell([length(sigmas) length(Ks)]);
auc_all = [length(sigmas), length(Ks)];
X_all_or = cell([length(sigmas), length(Ks)]);
Y_all_or = cell([length(sigmas) length(Ks)]);
auc_all_or = cell([length(sigmas) length(Ks)]);
snrs = zeros(1, length(sigmas));
scores = zeros(1, M);
oracle_scores = zeros(1, M);

A = normrnd(0, 1/(K), [K, M]);
noise = sigmas(1)*randn(1, M);
gn = g' + noise';
snr = 20*log10(norm(g)/norm(noise));
snr;
y = A*gn;

num_change_points = 50;

[estimate, est_inds] = smashed_filt_estimate(y, M, K, A, L, F, num_change_points, 0.05);
oracle = smashed_filt_oracle(y, M, K, A, L, F, inds, 0.05);
correct = ismember(inds(1), est_inds);
if correct == 1
    correct_inds(run) = 1;
end

for kk=1:M
    if estimate(kk) == g(kk)
        scores(kk) = scores(kk) + 1;
    end
    if oracle(kk) == g(kk)
        oracle_scores(kk) = scores(kk) + 1;
    end
end
scores = scores/runs;
[X, Y, T, auc] = perfcurve(g, scores, 1);
[X_or, Y_or, T_or, auc_or] = perfcurve(g, oracle_scores, 1);

plot(X,Y)


