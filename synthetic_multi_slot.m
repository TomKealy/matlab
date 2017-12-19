close all;
randn('seed', 12354);
num_chunks = 5;
chunk_length = 300;
M = chunk_length;
Ks = [5, 10, 15, 20, 30, 50, 100, 150, 200, 250, 300];
Ks = fliplr(Ks);
runs = 10;

scores = zeros(length(Ks), 1500);
oracle_scores = zeros(length(Ks), 1500);

X_all = zeros(length(Ks), 100);
Y_all = zeros(length(Ks), 100);

for pp=1:length(Ks)
    K = Ks(pp);
    
    edges = [50, 120, 170, 192, 210, 240, 256, 300] ;
    levels = [1,  0 , 1, 0, 0, 1, 0, 0];
    idxs = zeros(1, M)  ;
    idxs(edges(1: end-1)+1) = 1 ;
    g = levels(cumsum(idxs)+1) ;
    
    edges = [50, 120, 170, 192, 220, 234, 256, 300] ;
    levels = [1,  0 , 1, 0, 0, 0, 0, 0];
    idxs = zeros(1, M)  ;
    idxs(edges(1: end-1)+1) = 1 ;
    gstar = levels(cumsum(idxs)+1) ;
    
    Fn = LehmerMatrix(M);
    [Ln, U] = lu(Fn);
    I = eye(M);
    D = inv(Ln);
    Ft = LehmerMatrix(num_chunks);
    [Lt Ut] = lu(Ft);
    Dt = inv(Lt);
    
    Dk = kron(Dt, D);
    a_g = D*g';
    a_gstar = D*gstar';
    
    F = kron(Ft, Fn);
    L = kron(Lt, Ln);
    It = eye(num_chunks);
    
    G = [g' ; g' ; g' ; gstar' ; gstar' ]';
    
    gt = G;
    allX = [;];
    allY = [;];
    pvals = [0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 0.8, 0.9, 0.99];
    noise = randn(1,length(gt));
    gt_noisy = gt ; %+ noise;
    snr = 20*log10(norm(gt)/norm(noise));
    for run = 1:runs
        for rr=1:length(pvals)
            A = normrnd(0, 1/(K), [K, M]);
            An = kron(It, A);
            An = An/norm(An);
            y = An*G';
            B = kron(Dt, eye(K));
            
            z = B*y;
            num_pieces = length(z)/K;
            
            z_pieces = zeros(num_pieces, K);
            y_pieces = zeros(num_pieces, K);
            num_groups = 0;
            
            for i=1:num_pieces
                z_pieces(i, :) = z((i-1)*K+1:(i-1)*K+K);
                y_pieces(i, :) = y((i-1)*K+1:(i-1)*K+K);
                if ~all(abs((z((i-1)*K+1:(i-1)*K+K)))) == 0.0
                    num_groups = num_groups + 1;
                end
            end
            num_groups = 5;
            groups = zeros(num_groups,K);
            
            zero_pieces = find(all(abs(z_pieces') == 0.0));
            
            a = zero_pieces;
            
            if num_groups == num_chunks
                output = {[1], [2], [3] ,[4], [5]};
            else
                output = accumarray( cumsum([0; diff(a(:))] > 1)+1, a, [], @(x) {x} );
            end
            
            estimate = zeros(num_groups, M);
            oracle_estimate = zeros(num_groups, M);
            
            for i=1:num_groups
                if ~(num_groups == num_chunks)
                    output{i} = [output{i}(1)-1 ; output{i}];
                end
                output = {[1], [2], [3] ,[4], [5]};
                candidates = [output{i}];
                %     if ~(num_groups == num_chunks)
                %         groups(i,:) = mean(y_pieces(candidates(i,:)));
                %     else
                groups(i,:) = y_pieces(candidates,:);
                
                estimate(i, :) = smashed_filt_estimate(groups(i,:)', M, K, A, Ln, Fn, 10, pvals(rr));
                %oracle_estimate(i, :) = smashed_filt_oracle(groups(i,:)', M, K, A, Ln, Fn, changepoints(i,:), pvals(pv));
            end
            % recustruct estimate
            
            recon = zeros(1, num_pieces*M);
            oracle_recon = zeros(1, num_pieces*M);
            start = 1;
            finish = M;
            for jj=1:num_groups
                groups = output{jj};
                for gg = 1:length(groups)
                    recon(1, start:finish) = estimate(jj,:);
                    start = finish + 1;
                    if finish < length(gt)
                        finish = finish + M;
                    end
                end
            end
            
            for kk=1:1500
                if recon(kk) == gt(kk)
                    scores(pp,kk) = scores(pp,kk) + 1;
                end
            end
        end
    end
    [X, Y, T, auc] = perfcurve(gt, scores(pp,:), 1);
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
figure
subplot(2,1,1)
plot(recon)
ylim([0, 1.5])
title('Reconstruction')
xlabel('Index')
ylabel('Denisty')
subplot(2,1,2)
plot(G)
ylim([0,1.5])
title('Synthetic Signal')
xlabel('Index')
ylabel('Denisty')

legend_strings = containers.Map;

for jj=1:length(Ks)
    ratio = Ks(jj)/M;
    legend_strings(num2str(Ks(jj))) = strcat(strcat('ROC curve m=', strcat(num2str(ratio), ' auc ='), num2str(auc)))
end

figure
plot(X_all', Y_all')
lengend = values(legend_strings)