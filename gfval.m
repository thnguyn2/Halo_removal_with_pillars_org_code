function [grad,term1_grad,term2_grad,term3_grad,term4_grad,term5_grad] = fval(gamma_os,l,f,g,params,ao2,as2)
%Compute df/dl of the following function:
%E=|gamma_os-conj(f).*gk|^2+lambda*|f-l*ho|^2+lambda*|g-l*hs|^2+...
    %beta*|ao2-conj(f).*(l*ho)|^2+ beta*|bo2-conj(g).*(l*hs)|^2
%Author: Tan H. Nguyen
    term1_grad = zeros(size(gamma_os));
    term2_grad = (params.Ho'*(params.Ho*l-f));
    term3_grad = (params.Hs'*(params.Hs*l-g));
    term4_grad = params.Ho'*((abs(f).^2).*(params.Ho*l)-f.*ao2);
    term5_grad = params.Hs'*((abs(g).^2).*(params.Hs*l)-g.*as2);
    grad = term1_grad + params.lambda*(term2_grad+term3_grad) + params.beta*(term4_grad + term5_grad);
    
    
    