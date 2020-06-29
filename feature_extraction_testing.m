clc;
clear;
close all;

%% Initialization

No_of_train = 824;
batch_size = 200;
patch_size = 51;
frame_spacing = 10;

OV = 2;                         % overlap factor of 2 (4 is also often used)
wshift = 160;                   % set frame increment in samples
wlen = wshift*OV;               % DFT window length
W = hamming(wlen);              % window
N_fft = 256;                    % n-point FFT
beta = 1;
batch_ind = 0;

%% make directories
addpath(genpath('.../path/of/gammatone_folder/'));
clean_path = '.../path/of/testing_clean_data/';
noisy_path = '.../path/of/testing_noisy_data/';
files = dir([clean_path,'/*.wav']);

% mkdir to save testing mat files
mkdir '.../path/to_save/testing_data' 'folder_name_testing'
save_path = '.../path/to_save/testing_data/folder_name_testing/';

%%
for i = 1:824
    
    disp(['Processing file : ', num2str(i)])
    clean_file = [clean_path,files(i).name];
    noisy_file = [noisy_path,files(i).name];
    [clean,fs] = audioread(clean_file);
    noisy = audioread(noisy_file);

    % extract features and get no.of frames to be added
    [ ~,Clean_gtm,Noisy_gtm,N_gtm,~ ] = my_gammatone( clean, noisy, fs, wlen, wshift,0.95);
    Min_frame = min([length(Clean_gtm(1,:))]);
    multiple = ceil(Min_frame/patch_size);
    add_frames = abs(multiple*patch_size-Min_frame);

    % append columns for getting columns equal to multiple of patch_size
    Clean_gtm = [Clean_gtm Clean_gtm(:,1:add_frames)];
    Noisy_gtm = [Noisy_gtm Noisy_gtm(:,1:add_frames)];
    N_gtm = [N_gtm N_gtm(:,1:add_frames)];

    % get IRM and clean training labels
    IRM = (Clean_gtm.^2)./(Clean_gtm.^2 + N_gtm.^2);
    RM = Clean_gtm./Noisy_gtm;
    RM(RM>1)=1;
    Clean_gtm = RM.*Noisy_gtm;

    % apply log to gammatone features and then normalize them
    Log_Noisy_gtm = log(Noisy_gtm);
    Log_Noisy_gtm = (Log_Noisy_gtm-mean(Log_Noisy_gtm(:)))/std(Log_Noisy_gtm(:));

    Feat = Log_Noisy_gtm';
    clean_cent = Clean_gtm';
    noisy_cent = Noisy_gtm';

    save([save_path, erase(files(i).name, '.wav'),'.mat'],'noisy_cent', 'clean_cent', 'Feat', 'IRM');
    batch_ind = batch_ind+1;

end
