function [X] = poly(X_train,k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
m = length(X_train);
X = [ones(m,1) X_train];

i = 2;
while i <= k
    X(:,i+1) = X_train.^i;
    i = i+1;
end
end

