close all;
randn('seed', 1234);
% data = csvread('tvws_data1.csv');
% chunk_length = 12810;
% num_chunks = 3;
% signal = zeros(num_chunks, chunk_length);
% start = 1;
%
% for ii=1:num_chunks
%     chunk = data(start:start+(chunk_length-1), 2);
%     %figure
%     %plot(chunk)
%     start = start + chunk_length;
%     signal(ii, :) = chunk;
% end

%signal(2,:) = signal(1,:);
%signal(2,6000:7200) = signal(1, 3800:5000);
%
% g = signal(1,:);
% gstar = signal(2,:);
% % experimient
% G = [signal(1,:); signal(1,:); signal(2, :)];
%
% % end exp
%
% plot(1:chunk_length, signal(1,:), 'b', 1:chunk_length, signal(2,:), 'r')

num_chunks = 5;
chunk_length = 300;
M = chunk_length;
Ks = [5, 10, 15, 20, 30, 50, 100, 150, 200, 250, 300];
Ks = fliplr(Ks);
runs = 1;

X_all = cell([1 length(Ks)]);
Y_all = cell([1 length(Ks)]);
auc_all = cell([1 length(Ks)]);

edges = [50, 120, 170, 192, 220, 234, 256, 300] ;
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

changepoints = zeros(2, 300); 
changepoints(1,:) = abs(a_g);
changepoints(2,:) = abs(a_gstar);
scores = zeros(1, 1500);
oracle_scores = zeros(1, 1500);
gt = G;
allX = [;];
allY = [;];
snrs = [.01, .05, .1, .5, 1, 1.5 2];
pvals = [0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 0.8, 0.9, 0.99];
for ss=1:length(snrs)
for kk=1:length(Ks)
    for rr = 1:runs
        for pv = 1:length(pvals)
        K = Ks(kk);
        noise = snrs(ss)*randn(1,length(gt));
        gt_noisy = gt + noise;
        snr = 20*log10(norm(gt)/norm(noise));
        
        A = normrnd(0, 1/(K), [K, M]);
        An = kron(It, A);
        An = An/norm(An);
        y = An*gt_noisy';
        B = kron(Dt, eye(K));
        
        z = B*y;
        w = Dk'*An'*y;
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
            candidates = [output{i}];
            if ~(num_groups == num_chunks)
                groups(i,:) = mean(y_pieces(candidates,:));
            else
                groups(i,:) = y_pieces(candidates,:);
            end
            estimate(i, :) = smashed_filt_estimate(groups(i,:)', M, K, A, Ln, Fn, 50, pvals(pv));
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
                oracle_recon(1, start:finish) = oracle_estimate(jj,:);
                start = finish + 1;
                if finish < length(gt)
                    finish = finish + M;
                end
            end
        end
        %
        tpr = 0;
        fpr = 0;
        
        m = length(gt);
        
        for l=1:m
            if recon(l) == gt(l)
                scores(1,l) = scores(1, l) + 1;
            end
        end
        
        for l=1:m
            if oracle_recon(l) == gt(l)
                oracle_scores(1,l) = oracle_scores(1,l) + 1;
            end
        end
        
        end
    end
    [X, Y, T, auc] = perfcurve(gt, scores, 1);
    %[X_or, Y_or, T_or, auc_or] = perfcurve(gt, oracle_scores, 1);
    %figure
    %plot(X,Y)
    %title(strcat(num2str(K), ',', num2str(auc)))
    X_all{kk} = X;
    Y_all{kk} = Y;
    auc_all{kk} = auc;
end

legendInfo = cell([1 length(Ks)]);
figure, hold on
for ii=1:length(Ks)
    plot(X_all{ii}, Y_all{ii});
    legendInfo{ii} = [num2str(Ks(ii)/M), ',', num2str(auc_all{ii})];
    t = (num2str(snr));
    title(t)
    xlabel('tprs')
    ylabel('fprs')
end
legend(legendInfo);
end