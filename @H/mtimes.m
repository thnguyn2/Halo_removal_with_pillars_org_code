function res = mtimes(H,x)
%Compute H*x when adjoint = 0 or H'*x when adjoint = 1
hif = H.hif; %High pass filter
xf = H.F*x;
if (H.adjoint) 
    %Compute H'*x
    res = H.F'*(conj(hif).*xf);
else
    %Compute the filtering result Hx = h*x
    res = H.F'*(hif.*xf); %Fourier transform of the fil
end