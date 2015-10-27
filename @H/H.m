function op = H(N,hf,F)
%Filtering operator given the 2DFFT of the spatial correlation function hf
%Inputs:
%  N: dimension
%Outputs:
    op.N = N;
    op.hif = hf;%High-pass filter used in the operation
    op.adjoint = 0;
    op.F = F; %Fourier transform operator
    op = class(op,'H');