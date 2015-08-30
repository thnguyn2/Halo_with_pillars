function demo
    %This function estimate the filter kernel from the edgemeasurement
    clc;
    clear all;
    close all;
    imname='10xPh1_03';
    %Extract by measuring the PSF at different points
    %First, load the image of the pillar
    im = im2double(imread(strcat(imname,'.tif')));
    
    %Extract the image of the edge
    im = im(898:924,1531:1619);
    im = im(:,end:-1:1);
    im = im(:,1:end-1);
    
    %%Take the average line cross-section
    profile = sum(im,1)/size(im,1); %Sum over rows
    
    npoints = size(im,2);
    scale = 3.1125;%pixels/microns
    ncols = size(im,2);
    distance_in_mu = ncols/scale;
    
    xx = linspace(-distance_in_mu/2,distance_in_mu/2,ncols);%Measured distance in microns
    %Next, filter the proflle to remove the additive noise
    profilerec = conv(profile,gauss1d(0.5,30),'same');%Avoid measurement noise
    
    %Baseline correction to correct for sample tilt
    leftmean = mean(profile(1:10)); 
    rightmean=mean(profile(end-10:end));
    baseline = (xx-xx(1))/(xx(end)-xx(1))*(rightmean-leftmean)+leftmean;
    profilerec = profilerec -baseline; %Now, perform baseline correction
    profile = profile - baseline; %Line profile after correction. The left and right end should have 0 values now.
    phi_m=profilerec; 
  
    %%Prepare the step function 
    %Find the zero-crossing around the central
    phi_abs = abs(phi_m);
    start_loc = round(npoints*0.45);
    end_loc = round(npoints*0.55);
    [min_val,idx]=min(phi_abs(start_loc:end_loc));
    zcr_coord = start_loc + idx; %Location of the zero-crossing
    disp(['Location of the zero-crossing' num2str(zcr_coord)]);
    xx = xx-xx(zcr_coord-1); %The location of the jump should be 0
    
    %Make sure the edge filter respons have the same max and min vlaue
    [phi_max,max_loc] = max(phi_m);
    [phi_min,min_loc] = min(phi_m);
    %Scale the right and the left part of the plot, make sure they are
    %symmetric across the x-axis
    lhs = phi_m(1:zcr_coord);
    rhs = phi_m(zcr_coord+1:end);
    %Normalize the max and the min for phase profile to -1 and 1
    lhs = lhs/abs(phi_min);
    rhs = rhs/abs(phi_max);
    phi_m = [lhs,rhs];
    %%-----Done with pre-processing here-----%%
    
    %Display the phase cross-section and raw cross-sections
    figure(1);
    subplot(121)
    plot(xx,phi_m,'g','linewidth',2);
    h=xlabel('Distance ({\mu}m)');
    set(h,'FontSize',14);
    set(gca,'FontSize',14);
    grid on;
    h=ylabel('Phi [rad]');
    set(h,'FontSize',14);
    title('Final phase cross-section')
    subplot(122)
    plot(xx,im(round(end/2)+5,:)-baseline,'b','linewidth',2);
    hold on;
    plot(xx,profile,'r','linewidth',2);
    axis([min(xx) max(xx) -0.3 0.3]);
    h_legend = legend('Raw profile','Mean profile');    set(h_legend,'FontSize',14);
    h=ylabel('arg(\Gamma_{r,s}(x)) [rad]');    set(h,'FontSize',14);
    h=xlabel('Distance ({\mu}m)');     set(h,'FontSize',14); 
    set(gca,'FontSize',14);
    grid on;

    
    %First result, process with Tan's edge method
    [h_mask,h_cs2,xx1d] = edge_process_tan_method(phi_m,xx);
    [xx,yy]=meshgrid(linspace(-distance_in_mu/2,distance_in_mu/2,size(h_mask,1)),...
    linspace(-distance_in_mu/2,distance_in_mu/2,size(h_mask,1)));
    figure(2);
    subplot(121);
    mesh(xx,yy,h_mask);colormap jet;title('PSF (Orignal method)')
    axis([-distance_in_mu/2,distance_in_mu/2,-distance_in_mu/2,distance_in_mu/2,min(h_cs2),max(h_cs2)]);
    axis off;
    grid on;
    save(strcat(imname,'_psf.mat'),'h_mask','h_cs2','im');
    
   
   
  
end

function [hmask] = measurepsf_directly(im,coordlist,halfwidth,smallerphaseobj)
    %Use a list of point coordinates, measure several pointspread function and take the
    %average
    %   im: input image
    %   coordlist: an array of npoints x 2 containing the centroids of the
    %   PSFs
    %   halfwidth: the output image will have dimensions of [2*halfwidth+1]
    %   x [2*halfwidth+1]
    %   smallerphaseobj: if 1 the object has smaller phase than the
    %   background and vice versa
    
    npoints = size(coordlist,1);
    fullwidth = 2*halfwidth+1;
    batch_arr = zeros(npoints,fullwidth,fullwidth);
    searching_area_dim = 5;
    for pointidx = 1:npoints
        pointidx
        currow = coordlist(pointidx,1);
        curcol = coordlist(pointidx,2);
        %Get the current batch
        curbatch = im(currow-halfwidth-searching_area_dim:currow+halfwidth+searching_area_dim,...
                      curcol-halfwidth-searching_area_dim:curcol+halfwidth+searching_area_dim);
        if (smallerphaseobj)
            curbatch = -curbatch;
        end  
        [maxval,maxcolidx]=max(max(curbatch,[],1),[],2);
        [maxval,maxrowidx]=max(max(curbatch,[],2),[],1);     
        batch_arr(pointidx,:,:)=curbatch(maxrowidx-halfwidth:maxrowidx+halfwidth,...
                                         maxcolidx-halfwidth:maxcolidx+halfwidth);
    end
    hmask = squeeze(mean(batch_arr,1));
    figure(2);
    subplot(2,2,1);imagesc(hmask);colormap jet;colorbar;
    hmaskf = fftshift(fft2(hmask));
    subplot(2,2,2);plot(real(hmaskf((end+1)/2,:)));colormap jet;colorbar;title('Fourier transform of the PSF')
    hmaskf = real(hmaskf);    %Make sure the PSF is symmetric across the center
    %Convert into a low-pass filter
    hmaskf = 1-hmaskf;
    hmask2 = ifft2(ifftshift(hmaskf));
    hmask = hmask2/abs(sum(hmask2(:)));
    subplot(2,2,3);imagesc(hmask2);colormap jet;colorbar;
    subplot(2,2,4);plot(hmask(round(end/2),:));colorbar;title('Extracted low-pass filter')
end



function [h_mask,h_cs2,xx1d] = edge_process_tan_method(phi_m,xx)
%This is the solver using Tan's method based on separability assumption
    %Crop out the section withing min_loc and max_loc
    [phi_max,max_loc] = max(phi_m);
    [phi_min,min_loc] = min(phi_m);
    phi_m = [phi_m(1:min_loc) phi_m(max_loc:end)];
    step_signal = zeros(size(phi_m));
    step_signal(min_loc+1:end)=2;
    phi_filtered = -phi_m+step_signal;
    %Adjust the coordinate of the zero-crossing
    xx=[xx(1:min_loc) xx(max_loc:end)-xx(max_loc)+xx(min_loc+1)];
    xx = xx - xx(min_loc+1);%Set the begining of the jump to be 0 coordinate
    
    %Smooth_out the signal to prevent the noise
    phi_filtered = conv1d(phi_filtered,gauss1d(10,80));%phi_filter is the original phase profile + step jump. Therefore, it is a smooth signal
    lhs = exp(i*phi_filtered);
    dlhs = lhs(2:end)-lhs(1:end-1);
    hi1 = dlhs;
    hi1f = fft(hi1);hi1f=real(hi1f);
    h_cs2 = ifft(hi1f);
    h_cs2=h_cs2(2:end);
    h_cs2 = h_cs2/max(h_cs2);
    
    xx = xx(1:length(h_cs2));
    xx1d=xx;
    %Finally, create a 2D matrix that has the specified cross-section
    f_size = 2*floor((length(h_cs2)+1)/2)-1;
    half_f_size = (f_size-1)/2;
    h_mask = zeros(f_size,f_size);
    [max_val,max_idx]=max(h_cs2);
    coord_arr = -half_f_size:half_f_size;
    [xx,yy]=meshgrid(coord_arr,coord_arr);
    distancemap = round(sqrt(xx.^2+yy.^2));
    for x_coord=1:f_size
        for y_coord = 1:f_size
            if (distancemap(x_coord,y_coord)<half_f_size)
                h_mask(y_coord,x_coord)=real(h_cs2(max_idx+distancemap(x_coord,y_coord)));
            end
        end
    end

end

function h=gauss1d(std_x,npoints)
    %Geneate 1d gaussian filter
    x=linspace(-npoints/2,npoints/2,npoints);
    h = exp(-x.^2/2/std_x^2);
    h = h./sum(h);
end
function y=conv1d(x,h)
    %Produce 1D convolution with replicate
    m = length(x);
    n = length(h(:));
    x_pad = ones(m+n-1,1);
    x_pad((n+1)/2:(n+1)/2+m-1)=x;
    x_pad(1:(n+1)/2)=mean(x(1:10));
    x_pad((n+1)/2+m:m+n-1)=mean(x(end-10:end));
    
    y = conv(x_pad,h,'same');
    y = y((n+1)/2:(n+1)/2+m-1);
end