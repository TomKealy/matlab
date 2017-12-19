close all;
clear all;

M = 300;

edges = [100, 200, 300] ;
levels = [0,  1 , 0];
idxs = zeros(1, M)  ;
idxs(edges(1: end-1)+1) = 1 ;
g = levels(cumsum(idxs)+1) ;

figure
plot(g)

Fn = LehmerMatrix(M);
[Ln, U] = lu(Fn);
I = eye(M);
D = inv(Ln);
Finv = inv(Fn);

inds = matching_pursuit_2ndD(g', Ln, Finv, 10)
