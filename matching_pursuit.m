function [inds] = matching_pursuit(z0, e, num_changpoints)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

zr = z0';
inds = [];
for rr=1:num_changpoints
   u = e*zr';
   [m, i] = max(abs(u));
   inds(rr) = i;
   c = m/i;
   zr = zr - c*e(i,:);
   figure
   plot(zr)
end
end

