function res = ctranspose(a)
%Compute the adjoint operator of the Fourier transform
a.adjoint = xor(a.adjoint,1); 
res = a;
