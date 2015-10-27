function [grad_res,meas_grad,conv_grad,tv_grad] = gfval(gamma,tk,gk,params)
%Compute df/dt at tk in order to solve for tk given the current estimation
%of gk and the current solution tk
%The objective function is f = ||arg(gamma)-Hi.arg(tk)||^2 + lambda*sum_over_r{[(Dx(arg(tk))^2 + Dy(arg(tk))^2 +1e-15)]^0.5}
%   a_gamma = arg(gamma)
%   a_tk = arg(tk)
%Author: Tan H. Nguyen
    lambda = params.lambda;
    beta = params.beta;
    TV = params.TV;
    H = params.H; %Filtering operator
    %Gradient of the measurement error
    meas_grad = 2*(gamma-tk.*conj(gk)).*conj(gk);
    %Compute the derivative of the convolution term
    conv_grad = lambda*2*(H'*(H*tk-gk)); %Gradient of the ||a_gamma-H.a_tk||^2 term
    Dx =TV*tk;
    G = Dx.*(Dx.*conj(Dx) + 1e-15).^(-0.5);%Derivative of the TV
    tv_grad = beta*(TV'*G);    %See Tan's notebook for the adjoint operator of TV
    grad_res = meas_grad+conv_grad+tv_grad;
  
    
    