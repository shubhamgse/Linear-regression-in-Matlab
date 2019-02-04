function J = computeCost(X, Y, theta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


h = X * theta;

J = sum((Y - h).^2);

end

