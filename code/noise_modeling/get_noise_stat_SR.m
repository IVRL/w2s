clearvars;
%delete(gcp('nocreate'));
%p = parpool(24);

addpath(strcat('../denoise/matlab/utils/'));

datasets(1) = "City100/City100_NikonD5500";
datasets(2) = "City100/City100_iPhoneX";
datasets(3) = "realSR/Canon/Test/2";
datasets(4) = "realSR/Nikon/Test/2";


time_start = datestr(now,'dd-mm-yyyy HH:MM:SS FFF');
for dataset_idx = 1:numel(datasets)
    
    dataset = datasets(dataset_idx)
    D = convertStringsToChars(strcat('../../data/', dataset, '/'));
    S = dir(fullfile(D,'*L*'));
    
    for img_idx = 1:numel(S)
    
        %Reading the image
        F = fullfile(D,S(img_idx).name);
        I = imread(F);
        y = im2double(I);
        
        for channel = 1:3 %loop over RGB
            %Estimating the noise
            fitparams = estimate_noise(y(:,:,channel));
            a = fitparams(1);
            b = fitparams(2);
            %if a<0
            %    a = eps;
            %end
            %if b<0
            %    b = eps;
            %end
            % sigma = sqrt(b);

            NoiseAB(img_idx, channel, 1) = a;
            NoiseAB(img_idx, channel, 2) = b;
        end
    end
    
    save_name = strcat('SR/dataset', num2str(dataset_idx), '_NoiseAB.mat');
    save(save_name,'NoiseAB');
    
end



time_end = datestr(now,'dd-mm-yyyy HH:MM:SS FFF');
disp(time_start);
disp(time_end);