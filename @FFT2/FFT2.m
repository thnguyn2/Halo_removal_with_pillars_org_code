function p = FFT2(N)
%Return the 2D Fourier transform operator
%Author: Tan H. Nguyen
%Input:
%   N: number of points in the x-dimension. Currently, only squared image
%   is supported
%Output:
%   p: the operator

    p.adjoint = 0;
    p.N = N;
    p = class(p,'FFT2'); %Return a class instance with defined parameters
    

