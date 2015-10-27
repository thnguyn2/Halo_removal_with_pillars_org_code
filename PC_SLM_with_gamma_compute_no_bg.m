function demo
clc;
clear all;
close all;
fname = '~/Documents/Important_stuffs/SLIM_data_for_halo_removal/SLIM_Oct_19/20_um_pillars/';
A=(im2double((imread(strcat(fname,'SNAP_image_7_0.tif')))));%Note that only a change in 4 frame order will make the dc shifted by pi/2, which will lower the phase of the DC.
B=(im2double((imread(strcat(fname,'SNAP_image_7_1.tif')))));
C=(im2double((imread(strcat(fname,'SNAP_image_7_2.tif')))));
D=(im2double((imread(strcat(fname,'SNAP_image_7_3.tif')))));
Gs=D-B; Gc=A-C;
del_phi=(atan2(Gs,Gc));
L=(A-C+D-B)./(sin(del_phi)+cos(del_phi))/4; %E0*E1
g1=(A+C)/2; %E0^2+E1^2=s
g2=L.*L; %E0^2*E1^2=p
x1=g1/2-sqrt(g1.*g1-4*g2)/2; x2=g1/2+sqrt(g1.*g1-4*g2)/2; %solutions for the 2 field, AC and DC
%Crop del_phi to correct FOV
fid = fopen(strcat(fname,'coord.txt'), 'r');
if (exist(strcat(fname,'coord.txt')))
    coord = fscanf(fid,'%d %d %d %d');
    if (~isempty(coord))
        xl = coord(1);
        xr = coord(2);
        yt = coord(3);
        yb = coord(4);
        del_phi = del_phi(yt:yb,xl:xr);
        x1 = x1(yt:yb,xl:xr);
        x2 = x2(yt:yb,xl:xr);
  end
end
as=sqrt(x1);ao = sqrt(x2)*2.3;
beta=as./ao;
phi=atan2(beta.*sin(del_phi),1+beta.*cos(del_phi));

[nrows,ncols]=size(phi);
%Crop the images to make it squared
phi = phi(round((nrows-ncols)/2):round((nrows+ncols)/2)-1,1:ncols);
ao = ao(round((nrows-ncols)/2):round((nrows+ncols)/2)-1,1:ncols);
as = as(round((nrows-ncols)/2):round((nrows+ncols)/2)-1,1:ncols);
del_phi = del_phi(round((nrows-ncols)/2):round((nrows+ncols)/2)-1,1:ncols);

[nrows,ncols]=size(phi);
Nx=min(nrows,ncols); %Dimension for Performing the Fourier transform
figure(1);
subplot(131);imagesc(phi);colormap gray;colorbar;title('Original phase map');
subplot(132);plot(phi(end/2,:));title('Profile before reconstruction');

inverse = 1;
%Step 2: create the correlation kernel
htype = 'gaussian';

%Bandwidth of the objective function
obj_bw = 90; 
[x,y]=meshgrid(linspace(-ncols/2,ncols/2-1,ncols),linspace(-nrows/2,nrows/2-1,nrows));
obj_mask = sqrt(x.^2+y.^2)<obj_bw; %
tf = fftshift(obj_mask); 
phif = fft2(phi);
phi_denoised = ifft2(phif.*tf);
subplot(133);imagesc(phi_denoised);colormap gray;colorbar;title('Denoised phase map');

%This is the TF for all the spatial frequency
switch (htype)
    case {'gaussian'} %A gaussian filter
                    bw = 20; %Bandwidth parameter
                    h=fspecial('gaussian',[round(4*bw)+1 round(4*bw)+1],bw); %Transfer function of the low-pass filter...
                    %Fourier transform the filter
                    h1 = zeros(nrows,ncols);
                    h1(1:size(h,1),1:size(h,2))=h;
                    kernel_size=size(h,1);
                    h1 = circshift(h1,[-round((kernel_size-1)/2) -round((kernel_size-1)/2)]);
                    hof = cast(tf,'double').*fft2(h1);
                    hsf = cast(tf,'double').*(ones(size(hof))-hof);
    case {'lp'} %Bandpass filter
                    lp_bw = 20;%Bandwidth of the low pass filter in the frequency domain. The smaller it is, the more coherent the field will be 
                    hf = zeros(nrows,ncols);
                    mask = sqrt(x.^2+y.^2)<lp_bw; %
                    hf(mask)=1;
                    hof = fftshift(hf);
    case {'measured'} %Measured kernel - Not tested yet.
                    load(strcat('Pillars_data/',filename,'_psf.mat'),'h_mask');
                    h = h_mask;
                    h1 = zeros(nrows,ncols);
                    h1(1:size(h,1),1:size(h,2))=h;
                    kernel_size=size(h,1);
                    h1 = circshift(h1,[-round((kernel_size-1)/2) -round((kernel_size-1)/2)]);
                    hof = fft2(h1);
                    %Normalize the kernel
                    hof = hof/max(abs(hof(:)));
end

    %Parameter definitions
    params.niter =50; %Number of iterations needed
    params.lambda = 0;
    params.beta = 1;
    params.method = 'relax';%Choose between 'relax'/'cg'/'nlcf'
    %Operator definitions
    params.F = FFT2(Nx); %Fourier transform operator
    params.Ho = H(Nx,hof,params.F); %Low-passed filtering operator
    params.Hs = H(Nx,hsf,params.F); 
    
warning('off','all'); %Disable all warnings
if (inverse)
      gpu_compute_en =0; %1-Enable GPU computing
      %Initialize our estimates
      l = ao+as.*exp(1j*del_phi); %total field estimation
      f = ao;
      g = as.*exp(1j*del_phi);
      %Compute the gamma_o,s
      gamma_os = as.*ao.*exp(i*del_phi); %Us*conj(Uo)
      ao2 = ao.^2;
      as2 = as.^2;
      [obj,term1,term2,term3,term4,term5]=objective_comp(gamma_os,l,f,g,params,ao2,as2);
      disp(['Current objective: ' num2str(obj), ', #1: ' num2str(term1) ', #2: ' num2str(term2) ...
          ', #3: ' num2str(term3) ', #4: ' num2str(term4), ', #5: ' num2str(term5)]);

      %Update f given g and l
      Hol = params.Ho*l;
      num = g.*conj(gamma_os)+params.lambda*Hol+params.beta*Hol.*ao2;
      den = abs(g).^2+params.lambda+params.beta*abs(Hol).^2;
      f = num./den;
      [obj,term1,term2,term3,term4,term5]=objective_comp(gamma_os,l,f,g,params,ao2,as2);
      disp(['    Updating f: ' num2str(obj), ', #1: ' num2str(term1) ', #2: ' num2str(term2) ...
          ', #3: ' num2str(term3) ', #4: ' num2str(term4), ', #5: ' num2str(term5)]);
        
      %Update g given f and l
      Hsl = params.Hs*l;
      num = f.*gamma_os+params.lambda*Hsl + params.beta*Hsl.*as2;
      den = abs(f).^2+params.lambda +params.beta*abs(Hsl).^2;
      g = num./den;
      [obj,term1,term2,term3,term4,term5]=objective_comp(gamma_os,l,f,g,params,ao2,as2);
      disp(['    Updating g: ' num2str(obj), ', #1: ' num2str(term1) ', #2: ' num2str(term2) ...
          ', #3: ' num2str(term3) ', #4: ' num2str(term4), ', #5: ' num2str(term5)]);
      
      %Update l given f and g
      rhs = params.lambda*(params.Ho'*f + params.Hs'*g)+...
            params.beta*(params.Ho'*(f.*ao2)+params.Hs'*(g.*as2));
      f1 = params.lambda + params.beta*abs(f).^2;
      g1 = params.lambda + params.beta*abs(g).^2;
      l = cgs(@(x)A_comp(x,f1,g1,params,nrows,ncols),rhs(:),1e-5,5);
      l = reshape(l,[nrows ncols]);
      [obj,term1,term2,term3,term4,term5]=objective_comp(gamma_os,l,f,g,params,ao2,as2);
      disp(['    Updating l: ' num2str(obj), ', #1: ' num2str(term1) ', #2: ' num2str(term2) ...
          ', #3: ' num2str(term3) ', #4: ' num2str(term4), ', #5: ' num2str(term5)]);
 
     
%      maxbg_phase = 0.5;
%      method = 'relax';
%      if (gpu_compute_en==0)
%           [gk,tk] = estimate_gt(gamma,h,niter,lambda_weight,beta_weight,tol,method,bg);
%      else %Compute gk and tk on gpu
%           d = gpuDevice();
%           reset(d); %Reset the device and clear its memmory
%          [gk,tk] = estimate_gt_gpu(gamma,h,niter,lambda_weight,beta_weight,tol,method,1,bg);
%      end
%      bgsubtract_str='_processed';
%      writeTIFF(unwrap2(cast(angle(tk),'double')),strcat(fname,bgsubtract_str,'_.tif'))

end
end 


function y=A_comp(x,f1,g1,params,nrows,ncols)
    %This function computes the results of the lhs (diag(gk.^2)+lambda*H^H*H)*x
      x=reshape(x,[nrows,ncols]);
      HoX = params.Ho*x;
      HsX = params.Hs*x;
      HoX = f1.*HoX;
      HsX = g1.*HsX;
      y = params.Ho'*HoX+params.Hs'*HsX;
      y = y(:);
end
