function [ impulse_response,S_gtm,Y_gtm,N_gtm,Y_filtered ] = my_gammatone( clean, noisy, fs, wlen, wshift, Alpha)

fl = 50; fc = 1500;
fh = fs/2;
filters_per_ERB = 2.05;
W=hamming(wlen);              % window

addpath(genpath('.../path/of/gammatone_folder/'));
B = [1 -Alpha];
clean= filter(B,1,clean);
noisy= filter(B,1,noisy);

noise = clean-noisy;

analyzer = Gfb_Analyzer_new(fs, fl, fc, fh, filters_per_ERB);
bands = length(analyzer.center_frequencies_hz);
impulse = [1, zeros(1,799)];
[impulse_response, analyzer] = Gfb_Analyzer_process(analyzer, impulse);
frequency_response = fft(real(impulse_response)');
frequency = [0:799] * fs / 800;
impulse_response = real(impulse_response);


S_filtered = Gammatone_filter(clean,impulse_response);
S_gtm = energies( S_filtered,W,wshift );

Y_filtered = Gammatone_filter(noisy,impulse_response);
Y_gtm = energies( Y_filtered,W,wshift );

N_filtered = Gammatone_filter(noise,impulse_response);
N_gtm = energies( N_filtered,W,wshift );

end
