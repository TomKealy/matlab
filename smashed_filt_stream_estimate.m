function [estimate, inds] = smashed_filt_stream_estimate(y, M, K, A, L, F, num_change_points, alpha)

%
% y -> The compressive measurement
% M -> Size of original signal
% K -> Size of compressive signal
% L -> 2nd difference matrix
% F -> Lehmer matrix of size M
%
% 

[inds1, recon] = matching_pursuit_2ndD_streaming(y, L, inv(F), num_change_points);

estimate = L'*recon';
inds = inds1;
end

