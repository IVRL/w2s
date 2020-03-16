clearvars;
%delete(gcp('nocreate'));
%p = parpool(24);
addpath(genpath(pwd));


denoiser_method = "BM3D"
datasets = ["avg1", "avg2", "avg4", "avg8", "avg16"];


time_start = datestr(now,'dd-mm-yyyy HH:MM:SS FFF');
disp(time_start);
for dataset_idx = 1:numel(datasets)
    
    dataset = datasets(dataset_idx)
    D = convertStringsToChars(strcat('../../../data/all/', dataset, '/'));
    S = dir(fullfile(D,'*.png'));
    D_gt = convertStringsToChars(strcat('../../../data/all/', 'avg400', '/'));
    S_gt = dir(fullfile(D_gt,'*.png'));

    for img_idx = 1:numel(S)
    
        %Reading both the image and the ground-truth (to save PSNR/SSIM)
        F = fullfile(D,S(img_idx).name);
        I = imread(F);
        y = im2double(I);
        F_gt = fullfile(D_gt,S_gt(img_idx).name);
        I_gt = imread(F_gt);
        y_gt = im2double(I_gt);


        if denoiser_method == "BM3D"
            [y_denoised, time_avg] = denoise_VST_BM3D(y);
        elseif denoiser_method == "EPLL"
            [y_denoised, time_avg] = denoise_VST_EPLL(y);
        elseif denoiser_method == "PURELET"
            [y_denoised, time_avg] = denoise_PURE_LET(y);
        end
        
        
        %SAVE IMAGE
        foldername = strcat('../../../results/', denoiser_method, '/', dataset);
        if ~exist(foldername, 'dir')
            mkdir(foldername);
        end
        filename = strcat(foldername, '/', S(img_idx).name);
        imwrite(y_denoised, convertStringsToChars(filename));


        PSNR(img_idx) = 10*log10( 1/ mean( (y_denoised(:)-y_gt(:)).^2 ) );
        SSIM(img_idx) = ssim(y_denoised, y_gt);
        
    end
    
    save_name = strcat(foldername, '/', 'PSNR.mat');
    save(save_name,'PSNR');
    save_name = strcat(foldername, '/', 'SSIM.mat');
    save(save_name,'SSIM');

    time_end = datestr(now,'dd-mm-yyyy HH:MM:SS FFF');
    disp(time_end);
end

time_end = datestr(now,'dd-mm-yyyy HH:MM:SS FFF');
disp(time_start);
disp(time_end);