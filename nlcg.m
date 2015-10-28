function l = nlcg(gamma_os,l0,f,g,params,ao2,as2)
%Non-linear conjugatte gradient with line-search
%x=argmin ||y-H*x||_2^2+ alpha*TV(x) =  argmin f(x)
%However, gradient of |V*u|_p norm can not be computed easily.
%Here lambda contains the trade off between data consistency and sparseness
%constrain.
%Author: Tan H. Nguyen
%Writing date: 10/24/2011
%Email: thnguyn2@illinois.edu
%University of Illinois at Urbana-Champaign
%--------------------------------------------------------------------------
%Arguments:
%Input:
%   l0: initial estimation of the recovered image l
%   f,g,gamma_os, ao2,as2: current estimation of f,g, and the measurements
%   params: a struct containing all the constants and operators
%Reference:
%[1]. Nocedal, 'Numerical Optimization', 2006
%[2]. An introduction to the Conjugate Gradient Method without the
%Argonizaing Pain

%Parameters for Line-Search algorithm
params.LSc=0.01;%The constant c using in Line Search algorithm
params.LSrho=0.6;%Rho used in Line Search algorithm
params.LSMaxiter=100; %Max number of iteration for line-search
params.step0=1;
params.CgTol=1e-5;%Tolerence on the gradient norm to stop iterating 

%% Prepare for 1st iteration
[fk,term1,term2,term3,term4,term5] = fval(gamma_os,l0,f,g,params,ao2,as2);
gf0 = gfval(gamma_os,l0,f,g,params,ao2,as2);
pk=-gf0;%Initial searching direction, pk is dx.
xk=l0;
gfk=gf0;

k=0;
while ((k<100)&&(norm(gfk,'fro')>params.CgTol))
           
        step = params.step0; 
        %% Prepare for 1st iteration
        f0=fval(gamma_os,xk,f,g,params,ao2,as2);%New objective
        %disp(['#' num2str(k) ', step: ' num2str(step), ', obj:' num2str(f0,'%3.4f') ', measurement error: ' num2str(meas,'%0.3f'), ', convolution error: ' num2str(convv,'%0.3f') ', tv: ' num2str(tv,'%0.3f')]);
        k=k+1;
        f1 = fval(gamma_os,xk+step*pk,f,g,params,ao2,as2);
        lsiter = 0;
        while (f1 > f0 - params.LSc*step*abs(gfk(:)'*pk(:))) & (lsiter<params.LSMaxiter)%alpha = 0.01, t0 =1
        	lsiter = lsiter + 1;
            step = step * params.LSrho;
            f1 = fval(gamma_os,xk+step*pk,f,g,params,ao2,as2);
        end
         
         if (lsiter==0)
            params.step0=params.step0/params.LSrho; %If previous step can be found very quickly then increase step size
         else
            params.step0=params.step0*params.LSrho; %reduce intial searching step with knowledge given in previous step
         end
         %State update
         xk=xk+step*pk;

         %Calculate new gradient
         gfk1=gfval(gamma_os,xk,f,g,params,ao2,as2);
       
         %Updating coefficients
         beta=(gfk1(:)'*gfk1(:))/(gfk(:)'*gfk(:)+eps);
         gfk = gfk1;
         pk=-gfk1+beta*pk;
         if (mod(k,1000)==0)
             pk = -gfk1; %Restart
         end
    end
 l=xk;
  
    




