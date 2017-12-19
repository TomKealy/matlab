clear all;
close all;
randn('seed', 12354);
chunk_length = 300;
M = chunk_length;
Ks = [15, 20, 30, 50, 100, 150, 200, 250, 300];
Ks = fliplr(Ks);

X_all = zeros(length(Ks), 100);
Y_all = zeros(length(Ks), 100);

runs = 5;
scores = zeros(length(Ks), M);

for pp=1:length(Ks)
    K = Ks(pp);
    
%     edges =  [0, 50, 150, 250, 350, 450, 500, 520, 600, 700, 840, 900, 1000];
%     levels = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0];
%     idxs = zeros(1, M)  ;
%     idxs(edges(1: end-1)+1) = 1 ;
%     g = levels(cumsum(idxs)+1);
    
    edges = [50, 120, 170, 192, 210, 240, 256, 300] ;
    levels = [1,  0 , 1, 0, 0, 1, 0, 0];
    idxs = zeros(1, M)  ;
    idxs(edges(1: end-1)+1) = 1 ;
    g = levels(cumsum(idxs)+1) ;
    
    %g = data;
    
    F = LehmerMatrix(M);
    [L, U] = lu(F);
    I = eye(M);
    D = inv(L);
    
    %h = cumsum(g)';

    pvals = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3,  0.5, 0.75, 0.8, 0.9, 0.99];
    for run = 1:runs
        for rr=1:length(pvals)
            A = normrnd(0, 1/(K), [K, M]);
            y = A*g';
            
            estimate = smashed_filt_estimate(y, M, K, A, L, F, 80, pvals(rr));
            for kk=1:M
                if estimate(kk) == g(kk)
                    scores(pp,kk) = scores(pp,kk) + 1;
                end
            end
        end
    end
    
    [X, Y] = perfcurve(g, scores(pp, :), 1);
    plot(X, Y);
    l_new = length(X);
    if l_new < 100
        num_pads = 100 - l_new;
        X = padarray(X, num_pads, NaN, 'post');
        Y = padarray(Y, num_pads, NaN, 'post');
    else
        num_pads = l_new - 100;
        X_all = padarray(X_all, abs(num_pads), NaN, 'post');
        Y_all = padarray(Y_all, abs(num_pads), NaN, 'post');
    end
    X_all(pp,:) = X;
    Y_all(pp,:) = Y;
    
end

plot(X_all', Y_all')