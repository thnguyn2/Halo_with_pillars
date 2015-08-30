function demo
    %This function estimate the filter kernel from the edgemeasurement
    clc;
    clear all;
    close all;
    imname='10xPh1_03';
    %Extract by measuring the PSF at different points
    %First, load the image of the pillar
    im = im2double(imread(strcat(imname,'.tif')));
   
    h = fspecial('gaussian',10,0.5);
    im = imfilter(im,h,'same');
   
    figure(1);
    imagesc(im);title('Input image');
    %Add coordinates of the points to evaluate the PSF
    coordlist = [77 1560;
                 156 1560;
                 235 1560;
                 314 1560;
                 394 1561;
                 552 1561;
                 631 1562;
                 393 1798;
                 234 1876;
                 156 1718;
                 472 1719;
                 156 1718;
                 235 1639;
                 551 1640;
                 393 1640;
                 472 1719;
                 235 1639;
                 472 1956;
                 631 1799];%rows, cols
    
    
    %First result, process with Tan's edge method
    halfwidth  = 30;
    [h_mask] = measurepsf_directly(im,coordlist,halfwidth,1);
    save(strcat(imname,'_psf.mat'),'h_mask','im');
    
  
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