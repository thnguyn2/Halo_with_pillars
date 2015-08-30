function [gk,tk,tk0] = estimate_gt(gamma,h,niter,lambda,beta_weight,tol,method,bg)
    %This function compute the estimation for gk and tk given gamma
    
   % maxgamma = max(gamma(:));%This value should be very close to 1
   % gamma = gamma/maxgamma;

    obj_array=zeros(0,1);
    nrows = size(gamma,1);
    ncols = size(gamma,2);
    gk = ones(size(gamma)); %Initial estimation       
    
    init_eps = 1e-3;%Smart initialization regularization factor
    %Fourier transform the filter
    h1 = zeros(nrows,ncols);
    h1(1:size(h,1),1:size(h,2))=h;
    kernel_size=size(h,1);
    h1 = circshift(h1,[-(kernel_size-1)/2 -(kernel_size-1)/2]);
    hf = fft2(h1);
    
    hipf = 1-hf; %This is the fourier transform of delta - hf filter
    ang_gammaf = fft2(angle(gamma));
    maskf =abs(hipf>0.05);
    
    ang_tkf0=(conj(hipf).*ang_gammaf)./(abs(hipf).^2+init_eps).*maskf;%Weiner deconvolution
    ang_tk0 =ifft2(ang_tkf0);
    tk0 = exp(i*real(ang_tk0));
    tk = tk0; 
     
   
    %Next, solve with the iterative method
    [obj,val1,val2] = objective_comp(gamma,hf,tk,gk,lambda,beta_weight,nrows,ncols);
    obj_array(end+1)=obj;
    cjgamma = conj(gamma);
    disp(['Iter ' num2str(0) ': current objective: ' num2str(obj) ', 1st: ' num2str(val1),...
       ', 2nd: ' num2str(val2)]);
    
    
    
    for iter=1:niter
        tic
        %First, recover g from t
        tkf = fft2(tk);
        gk = (tk.*cjgamma+lambda*ifft2(tkf.*hf))./(conj(tk).*tk+lambda+1e-8);
        gk2 = imfilter(gk,fspecial('gaussian',[150 150],50),'same');
        gk = gk./exp(i*angle(gk2));%Get rid of the low frequency smooth variation in gk
        
        switch method
            case 'relax'
                beta = norm(gk,'fro');
                betasqr = beta^2;
                rhs = betasqr*gamma./conj(gk)+lambda*Hhg_comp(hf,gk);
                rhsf = fft2(rhs);
                tkf = rhsf./(betasqr+lambda*abs(hf).^2+1e-8); %Added factor for stability
                tk = ifft2(tkf);
            case 'cg'
                rhs = gk.*gamma + lambda*Hhg_comp(hf,gk);        
                tk = cgs(@(x)A_comp(x,hf,lambda,beta_weight,gk,nrows,ncols),rhs(:),tol,20);
                tk = reshape(tk,[nrows ncols]);
        end
        [obj,val1,val2] = objective_comp(gamma,hf,tk,gk,lambda,beta_weight,nrows,ncols);
        %obj_array(end+1)=obj;
        te = toc;
        disp(['Iter ' num2str(iter) ': current objective: ' num2str(obj) ', 1st: ' num2str(val1),...
        ', 2nd: ' num2str(val2)]);
        %Make sure that our phase is not offsett
        gamma_phase = angle(tk);
        gamma_phase = gamma_phase - mean(mean(gamma_phase(bg.bgyy1:bg.bgyy2,bg.bgxx1:bg.bgxx2)));
        tk = abs(tk).*exp(i*gamma_phase);
        figure(4);
        subplot(1,3,1);imagesc(angle(tk));colorbar;title(sprintf('Current estimation tk - iter #%d',iter));
        subplot(1,3,2);imagesc(angle(gk));colorbar;title(sprintf('Current estimation -gk iter #%d',iter));
        subplot(1,3,3);imagesc(angle(gk2));colorbar;title(sprintf('Current estimation -gk iter #%d',iter));
        
        colormap jet
        figure(3);
        plot(angle(tk(1200,:)));drawnow
        

    end
    



end

function y=A_comp(x,hf,lambda,beta_weight,nrows,ncols)
    %This function computes the results of (diag(gk.^2)+lambda*H^H*H)*x
    x = reshape(x,[nrows ncols]);
    xf = fft2(x);
    HhHf = conj(hf).*hf;
    yf = lambda*HhHf.*xf;
    y = ifft2(yf);
    y = y + x.*conj(gk).*gk; %This one is faster than abs(gk).^2  
    y = y(:);
end

function Hhg=Hhg_comp(hf,gk)
    %This function compute the product H^H*gk
    gkf = fft2(gk);
    Hhgf = conj(hf).*gkf;
    Hhg = ifft2(Hhgf);      
end