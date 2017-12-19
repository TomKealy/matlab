function [inds, recon] = matching_pursuit_2ndD_streaming(z0, e, Finv, num_changpoints)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

zr = z0';
inds = [];
recon = zeros(1, length(z0));
for rr=1:num_changpoints
   alpha = zr;
   [m, i] = max(abs(alpha));
   c = alpha(i);
   if ~c == 0
    inds(rr) = i;
   end
   zr(:,i) = zr(:,i) - c;
   recon(i) = c;
end

end