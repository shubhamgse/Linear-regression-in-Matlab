function [ theta ] = normalEqn( X,Y,l )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%theta = zeros(size(X,2),1);
n = size(X,2);
lambda = l;
E = eye(n);
E(1,1) = 0;

theta = inv((X'*X) + (lambda * E))*X'*Y;


end

