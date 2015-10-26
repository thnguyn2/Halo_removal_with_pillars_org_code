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
figure(1);
subplot(121);imagesc(phi);colormap gray;colorbar;title('Original phase delay');
subplot(122);plot(phi(535,:));title('Profile before reconstruction');

%Compute the gamma_o,s
gamma_os = as.*ao.*exp(i*del_phi); %Us*conj(Uo)
h_denoise = fspecial('gaussian',[9 9],0.25);
phi_denoised = imfilter(phi,h_denoise,'same');
[nrows,ncols]=size(phi);
inverse = 1;
bw = 10;
if (inverse)
      h=fspecial('gaussian',[round(6*bw)+1 round(6*bw)+1],bw); %Transfer function of the low-pass filter...
      h1 = zeros(nrows,ncols);
      h1(1:size(h,1),1:size(h,2))=h;
      kernel_size=size(h,1);
      h1 = circshift(h1,[-round((kernel_size-1)/2) -round((kernel_size-1)/2)]); 
      gpu_compute_en =0; %1-Enable GPU computing
%     %First, initialize tk and lk. Here, gk = t v h;
%     lambda_weight =5;
%     beta_weight=0;
%     tol = 1e-4; %We don't need to find the best in each step since we will tweak 2 variables t and g at the same time
%     niter=35;
%     if (gpuDeviceCount()==0)
%         gpu_compute_en = 0; %if there is no gpu, compute the result on cpu instead
%     end
% 
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