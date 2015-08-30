
function [obj,term1,term2] = objective_comp(gamma,hf,tk,gk,lambda,beta,nrows,ncols)
    %Compute the objective function E=||gamma-tk.gk||^2+lambda
    evect=gamma-tk.*conj(gk);
    term1 = norm(evect(:),'fro').^2;
    tkf = fft2(tk);
    gkfiltf =hf.*tkf;
    gkfilt = ifft2(gkfiltf);
    evect = gk-gkfilt;
    term2 = lambda*norm(evect(:),'fro').^2;
    obj = term1 + term2;
       
    
end