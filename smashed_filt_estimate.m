function [estimate, inds] = smashed_filt_estimate(y, M, K, A, L, F, num_change_points, alpha)

%
% y -> The compressive measurement
% M -> Size of original signal
% K -> Size of compressive signal
% L -> 2nd difference matrix
% F -> Lehmer matrix of size M
%
% 
% 
ghat = A'*y; 

inds1 = matching_pursuit_2ndD(A'*y, L, inv(F), num_change_points);

k = num_change_points;

ahat_big = inds1'; 

for ii = 1:M
    if ismember(ii, ahat_big)
        ii;
    else
        ahat(ii) = 0;
    end
end

% take the average of ghat between these vaules
piece_mean = zeros(1,k);
piece_var = zeros(1,k);
threshths = zeros(1,k);
inds = [1, ahat_big', M];
inds = sort(inds);
for l=1:k
    piece = ghat(inds(l):inds(l+1))';
    piece_ttests(l) = ttest(piece, 0, 'Tail', 'right', 'Alpha', alpha);
end

estimate = zeros(1, M);

for ii=1:num_change_points
    for jj = inds(ii):inds(ii+1)
        if piece_ttests(ii) == 1
            estimate(jj) = 1;
        else
            estimate(jj) = 0;
        end
    end
end

end

