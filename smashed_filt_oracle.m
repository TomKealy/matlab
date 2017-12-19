function [estimate] = smashed_filt_oracle(y, M, K, A, L, F, change_points, alpha)

%
% y -> The compressive measurement
% M -> Size of original signal
% K -> Size of compressive signal
% L -> 2nd difference matrix
% F -> Lehmer matrix of size M
%

ghat = A'*y; 
inds = change_points;
k = nnz(inds);
% take the average of ghat between these vaules
piece_mean = zeros(1,k);
piece_var = zeros(1,k);
threshths = zeros(1,k);

inds(end+1) = M;

for l=1:k
    piece = ghat(inds(l):inds(l+1))';
    piece_ttests(l) = ttest(piece, 0, 'Tail', 'right', 'Alpha', alpha);
end

estimate = zeros(1, M);

for ii=1:k
    for jj = inds(ii):inds(ii+1)
        if piece_ttests(ii) == 1
            estimate(jj) = 1; 
        else
            estimate(jj) = 0;
        end
    end
end

end

