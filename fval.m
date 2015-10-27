function [obj,term1,term2,term3,term4,term5] = fval(gamma_os,l,f,g,params,ao2,as2)
    %Compute the objective function
    %E=|gamma_os-conj(f).*gk|^2+lambda*|f-l*ho|^2+lambda*|g-l*hs|^2+...
    %beta*|ao2-conj(f).*(l*ho)|^2+ beta*|bo2-conj(g).*(l*hs)|^2
    
    evect1=gamma_os-conj(f).*g;
    term1 = norm(evect1(:),'fro').^2;
    Hol = params.Ho*l;
    Hsl = params.Hs*l;
    evect2=f-Hol;
    term2 = norm(evect2(:),'fro').^2;
    evect3=g-Hsl;
    term3 = norm(evect3(:),'fro').^2;
    evect4=ao2-conj(f).*Hol;
    term4 = norm(evect4(:),'fro').^2;
    evect5=as2-conj(g).*Hsl;
    term5 = norm(evect5(:),'fro').^2;
    obj = term1 + params.lambda*(term2+term3)+params.beta*(term4+term5);
end