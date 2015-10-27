function res = mtimes(f,x)
%Compute the 2D Fourier transform. Generate xf = F*x when f.adjoint = 0
% or compute the inverse Fourier transform when f.adjoint = 1;
%otherwise, perform F'*x
%Author: Tan H. Nguyen
%University of Illinois at Urbana-Champaign
if (f.adjoint) %Do inverse Fourier transform
    res = sqrt(f.N)*ifft2(x);
else
    res = 1/sqrt(f.N)*fft2(x);
end