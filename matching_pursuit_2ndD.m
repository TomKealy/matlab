function [inds, zr] = matching_pursuit_2ndD(z0, e, Finv, num_changpoints)
zr = z0';
inds = [];
for rr=1:num_changpoints
   u = e*zr';
   alpha = Finv*u;
   [m, i] = max(abs(alpha));
   c = alpha(i);
   if ~c == 0
    inds(rr) = i;
   end
   zr = zr - c*e(i,:);
end
end