function obj=ctranspose(d)
    %obj=ctranspose(d)
    d.adjoint=xor(d.adjoint,1);   
    obj=d;
end