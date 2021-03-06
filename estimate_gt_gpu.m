function [gk,tk] = estimate_gt_gpu(gamma,h,niter,lambda,beta_weight,tol,method,bg)
    %This function compute the estimation for gk and tk given gamma on GPU.
    %GPU is used to speed up the calculation of fft2 and ifft2
    
    %xo, yo: central location of the horizontal line to draw the
    %cross-section
  
   
    nrows = size(gamma,1);
    ncols = size(gamma,2);
    gk = ones(size(gamma)).*exp(i*angle(gamma)); %Initial estimation- not important
    gk = cast(gk,'single');
    gkd = gpuArray(gk);   
    
    gamma = cast(gamma,'single');    
    gammad = gpuArray(gamma);
    cjgammad = conj(gammad);
    
    %Fourier transform the filter
    h1 = zeros(nrows,ncols);  
    h1(1:size(h,1),1:size(h,2))=h;
    kernel_size=size(h,1);
    h1 = circshift(h1,[-round((kernel_size-1)/2) -round((kernel_size-1)/2)]);
    %Copy the filter kernel onto GPU
    h1d=gpuArray(h1);
    hfd = fft2(h1d);
    
    init_eps = 1e-3;%Smart initialization regularization factor
    hipdf = 1-hfd; %This is the fourier transform of delta - hf filter
    ang_gammadf = fft2(angle(gammad));
    ang_tkdf0=(conj(hipdf).*ang_gammadf)./(abs(hipdf).^2+init_eps);%Weiner deconvolution
    ang_tkd0 =ifft2(ang_tkdf0);
    tkd = exp(i*real(ang_tkd0));
    

    %Next, solve with the iterative method
    [obj,val1,val2]  = objective_comp(gammad,hfd,tkd,gkd,lambda,beta_weight,nrows,ncols);
     disp(['Iter ' num2str(0) ': current objective: ' num2str(obj) ', 1st: ' num2str(val1),...
       ', 2nd: ' num2str(val2)]);
    prevobj = 0;
    bestobj = obj;
    bgfilter = fspecial('gaussian',[150 150],50);
    bgfilterd = gpuArray(bgfilter);
    for iter=1:niter
        if (mod(iter,500)==0)
            tol = tol/10;
            disp('New value of tolerance');
        end
        tic;
        %First, recover g from t
        tkfd = fft2(tkd);
        gkd = (tkd.*cjgammad+lambda*ifft2(tkfd.*hfd))./(conj(tkd).*tkd+lambda+1e-8);
        
        gkd2 = imfilter(gkd,bgfilterd,'same');
        gkd = gkd./exp(i*angle(gkd2));%Get rid of the low frequency smooth variation in gk

        
        
        
        %Next, recover t from g
        switch method
            case 'relax'
                betasqr = conj(gkd(:))'*gkd(:);
                rhsd = betasqr*gammad./conj(gkd)+lambda*Hhg_comp(hfd,gkd);
                rhsfd = fft2(rhsd);
                tkfd = rhsfd./(betasqr+lambda*abs(hfd).^2+1e-8);
                tkd = ifft2(tkfd);
            case 'cg'
                 rhsd = gkd.*gammad + lambda*Hhg_comp(hfd,gkd);        
                 tkd = cgs(@(x)A_comp(x,hfd,lambda,gkd,nrows,ncols),rhsd(:),tol,30); %Just need a few step to get to the min.
                 tkd = reshape(tkd,[nrows ncols]);
        end
        
       %Cast T as the transmittance of a phase obj
       tkd = exp(i*angle(tkd));

       [obj,val1,val2] = objective_comp(gammad,hfd,tkd,gkd,lambda,beta_weight,nrows,ncols);
       relerr  = abs(obj-prevobj)/obj;
    
        te = toc;
          disp(['Iter ' num2str(iter) ': current objective: ' num2str(obj) ', 1st: ' num2str(val1),...
        ', 2nd: ' num2str(val2)]);
        prevobj = obj;
        
        %Draw the heat map for the error
         anglemap = angle(tkd)-mean(mean(angle(tkd)));
        uangle = anglemap;
       
       %Uncomment the following lines to do phase unwrapping
       % uangle = unwrap2(cast(gather(anglemap),'double'));
        figure(4)
        s=strcat('arg[T_k] iter = ',num2str(iter));
        imagesc(uangle);colorbar;title(s);colormap jet;
        colorbar
        
        drawnow;
    end
    gk = gather(gkd);
    tk = gather(besttkd);



end

function yd=A_comp(xd,hfd,lambda,gkd,nrows,ncols)
    %This function computes the results of (diag(gk.^2)+lambda*H^H*H)*x
    xd = reshape(xd,[nrows ncols]);
    xfd = fft2(xd);
    HhHfd = conj(hfd).*hfd;
    yfd = lambda*HhHfd.*xfd;
    yd = ifft2(yfd);
    yd = yd + xd.*conj(gkd).*gkd; %This one is faster than abs(gk).^2  
    yd = yd(:);
end

function Hhgd=Hhg_comp(hfd,gkd)
    %This function compute the product H^H*gk
    gkfd = fft2(gkd);
    Hhgfd=conj(hfd).*gkfd;
    Hhgd = ifft2(Hhgfd);      
end