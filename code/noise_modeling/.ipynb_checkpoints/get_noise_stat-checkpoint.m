clearvars;
%delete(gcp('nocreate'));
%p = parpool(24);

addpath(strcat('../denoise/matlab/utils/'));
for idx = 1:100
    datasets(idx) = "avg" + num2str(idx);
end

time_start = datestr(now,'dd-mm-yyyy HH:MM:SS FFF');
for dataset_idx = 1:numel(datasets)
    
    dataset = datasets(dataset_idx)
    D = convertStringsToChars(strcat('../../data/all/', dataset, '/'));
    S = dir(fullfile(D,'*.png'));
    D_gt = convertStringsToChars(strcat('../../data/all/', 'avg400', '/'));
    S_gt = dir(fullfile(D_gt,'*.png'));

    for img_idx = 1:numel(S)
    
        %Reading both the image and the ground-truth (to save PSNR/SSIM)
        F = fullfile(D,S(img_idx).name);
        I = imread(F);
        y = im2double(I);
        F_gt = fullfile(D_gt,S_gt(img_idx).name);
        I_gt = imread(F_gt);
        y_gt = im2double(I_gt);


        PSNR(img_idx) = 10*log10( 1/ mean( (y(:)-y_gt(:)).^2 ) );
        SSIM(img_idx) = ssim(y, y_gt);
        
        %Estimating the noise
        fitparams = estimate_noise(y);
        a = fitparams(1);
        b = fitparams(2);
        %if a<0
        %    a = eps;
        %end
        %if b<0
        %    b = eps;
        %end
        % sigma = sqrt(b);
        
        NoiseAB(img_idx, 1) = a;
        NoiseAB(img_idx, 2) = b;
        
    end
    
    save_name = strcat(dataset, '_PSNR.mat');
    save(save_name,'PSNR');
    save_name = strcat(dataset, '_SSIM.mat');
    save(save_name,'SSIM');
    save_name = strcat(dataset, '_NoiseAB.mat');
    save(save_name,'NoiseAB');
    
end



time_end = datestr(now,'dd-mm-yyyy HH:MM:SS FFF');
disp(time_start);
disp(time_end);