clc;
clear all;
close all;
fname='10xPh1_03';
uphi = -imread(strcat(fname,'.tif'));%The image is flipped in 10x dataset
nanidx= find(isnan(uphi)==1);
uphi(nanidx)=0;
figure(2)
imagesc(uphi);colorbar
axis image
axis off
writeTIFF(uphi,'SLIM image.tif');

load(strcat(fname,'_psf.mat'),'h_mask');
%%Added section for recovery....
[nrows,ncols]=size(uphi);
gamma_amp = ones(nrows,ncols);
%Filter the image to avoid very large value of the noise
h_denoise = fspecial('gaussian',[9 9],0.25);
uphi_denoised = imfilter(uphi,h_denoise,'same');
gamma = exp(i*(uphi_denoised));
inverse = 1;
%bw_array = linspace(15,15,1);%Specify the bandwidth 

bg.bgxx1 = 65;
bg.bgxx2 = 391;
bg.bgyy1 = 1678;
bg.bgyy2 = 1831;
   
if (inverse)
    h = h_mask;
    %h=fspecial('gaussian',[round(6*bw)+1 round(6*bw)+1],bw); %Transfer function of the low-pass filter...
    h1 = zeros(nrows,ncols);
    h1(1:size(h,1),1:size(h,2))=h;
    kernel_size=size(h,1);
    h1 = circshift(h1,[-round((kernel_size-1)/2) -round((kernel_size-1)/2)]);
    gpu_compute_en =0; %1-Enable GPU computing
    %First, initialize tk and lk. Here, gk = t v h;
    lambda_weight =5;
    beta_weight=0;
    tol = 1e-4; %We don't need to find the best in each step since we will tweak 2 variables t and g at the same time
    niter=35;
    if (gpuDeviceCount()==0)
        gpu_compute_en = 0; %if there is no gpu, compute the result on cpu instead
    end

     maxbg_phase = 0.5;
     method = 'relax';
     if (gpu_compute_en==0)
          [gk,tk] = estimate_gt(gamma,h,niter,lambda_weight,beta_weight,tol,method,bg);
     else %Compute gk and tk on gpu
          d = gpuDevice();
          reset(d); %Reset the device and clear its memmory
         [gk,tk] = estimate_gt_gpu(gamma,h,niter,lambda_weight,beta_weight,tol,method,1,bg);
     end
     bgsubtract_str='_processed';
     writeTIFF(unwrap2(cast(angle(tk),'double')),strcat(fname,bgsubtract_str,'_.tif'))
end    