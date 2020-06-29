clc;
clear;
close all;

%% Initialization
addpath(genpath('.../path/of/gammatone_folder/'));

patch_size = 51;
number = 824;
OV = 2;                         % overlap factor of 2 (4 is also often used)
wshift = 160;                   % set frame increment in samples
wlen = wshift*OV;               % DFT window length
W = hamming(wlen);              % window
N_fft = 256;                    % n-point FFT
beta = 1;

%% path of actual clean and noisy database
clean_path = '.../path/of/testing_clean_data/';
noisy_path = '.../path/of/testing_noisy_data/';
file_clean = dir([clean_path,'*.wav']);
file_noisy = dir([noisy_path,'*.wav']);

%% path of stored mask, after getting predicted masks
pathpredicted_mask = '.../path/of/enhanced/features/';
list_mask = dir([pathpredicted_mask,'*.mat']);

%% make different directories
mkdir '.../path/to_save/enhanced_data' 'folder_name_waveforms'

%% path to store enhanced waveforms
clean_wavform = '.../path/to_save/enhanced_data/folder_name_waveforms/';

%%

for i=1:length(file_noisy)
    disp(['Processing file : ', num2str(i)])

    % read clean and noisy file
    clean_file = [clean_path, file_clean(i).name];
    noisy_file = [noisy_path, file_noisy(i).name];
    [clean, fs] = audioread(clean_file);
    noisy = audioread(noisy_file);

    % get oracle mask
    [impulse_response, Clean_gtm, Noisy_gtm, N_gtm, Y_filtered ] = my_gammatone(clean, noisy, fs, wlen, wshift, 0.95);
    IRM = (Clean_gtm.^2)./(Clean_gtm.^2 + N_gtm.^2);

    % load enhanced mask
    predictedMasks = load([pathpredicted_mask, list_mask(i).name]);
    Min_frame = min([length(Clean_gtm(1,:))]);
    multiple = ceil(Min_frame/patch_size);
    subtract_frames = abs(multiple*patch_size-Min_frame);
    Pred_mask = predictedMasks.PRED_IRM;
    Pred_mask = Pred_mask';
    Pred_mask = Pred_mask(:,1:end-subtract_frames);

    % get oracle and enhanced waveforms
    pred_recon = synthesis(W, wshift, Pred_mask, Y_filtered, impulse_response, 0.95);

    % mean-var normalizing all the waveforms
    pred_recon = (pred_recon'-mean(pred_recon))/std(pred_recon)/12;

    % store the waveforms
    audiowrite([gan_wavform, erase(list_mask(i).name, '.mat'), '.wav'],pred_recon(1:end-1),fs)

end
