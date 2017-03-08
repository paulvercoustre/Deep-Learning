function g = reluGradient(z)
%RELUGRADIENT returns the gradient of the relu function
%evaluated at z
%   g = RELU(z) computes the gradient of the relu function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.
g= z;
g(g < 0) = 0;
g(g > 0) = 1;
end