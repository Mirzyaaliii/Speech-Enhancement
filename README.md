# Speech-Enhancement
Speech Enhancement - Remove the noise from audio files to produce more clear output.

This is a implementation of speech enhancemnet project using signal-processing concepts and Deep Learning (DL)-based mapping technique. The implementation done using [Pytorch](https://pytorch.org/) open-source library and publicly available dataset.


## Algorithm
In this project, Time-Frequency (T-F) masking-based approach is used<sup>1</sup>, which works on supervised-learning. In its simplest form, supervised model estimates an ideal T-F mask of a noisy mixture using a trained classifier, typically, any DL-based architecture, e.g, DNN, CNN. An ideal T-F mask indicates whether, or to what extent, each T-F unit is dominated by target speech.


## Dataset
Dataset used in this project, namely, **SUPERSEDED**, which is Publicly available<sup>2</sup>. Dataset contains separate data for training and testing for both kind of speech, i.e., clean and noisy speech. Training and testing data contain 11572 and 824 wave files, respectively. 

Link of the dataset - [https://datashare.is.ed.ac.uk/handle/10283/1942](https://datashare.is.ed.ac.uk/handle/10283/1942)


## Implementation
For implementation, follow the following 5 steps:
### Feature Extraction
All the experiments are done on 8k sampling frequency. As given dataset is originally in 16k, first, all the files are down-sampled to 8k. To down sample the file [SoX](http://sox.sourceforge.net/)-player is used.

1. To downsample the data, give according path and run the following script - downsampled.sh

In this project, **Gammatone FilterBank** feature are used<sup>3</sup>. In feature extraction parameters used are as follow, 51-channels with 20 ms Hamming window and 10 ms overlap between consecutive frames.

2. To extract the features, give according path and run the following scripts - feature_extraction_training.m & feature_extraction_testing.m


### Mapping Technique
To enhance the noisy speech, DL-based mapping technique is used. Particulary, Generative Adversarial Network (GAN)<sup>4</sup> is used to map the noisy speech to the clean speech. Here, one extra loss added as regularization in the implemenation of GAN, which tries to minimize the distance betwwen the predicted and target spectrum. Mean Square Error (MSE) loss is used to satisty this condition. 

3. For training, give according path and run the following command in terminal:
```sh
python3 training.py -sm .../path/to/save/model -trn .../path/of/training/data
```

4. For testing, give according path and run the following command in terminal:
```sh
python3 training.py -sm .../path/of/saved/model -tst .../path/of/testing/data -pfp .../path/to/save/predicted/features
```


### Resynthesis
To resynthesize the wave file from predicted features, Inverse Gammatone filterbank is applied<sup>2</sup>.

5. To synthesize file, give according path and run the following script - synthesize_enhanced_file.m


## Sources
1. [Soni, Meet H., Neil Shah, and Hemant A. Patil. "Time-frequency masking-based speech enhancement using generative adversarial network." In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5039-5043. IEEE, 2018.](https://ieeexplore.ieee.org/abstract/document/8462068)
2. [Valentini-Botinhao, Cassia. (2016). Noisy speech database for training speech enhancement algorithms and TTS models, [dataset]. University of Edinburgh. School of Informatics. Centre for Speech Technology Research (CSTR).](https://www.research.ed.ac.uk/portal/en/datasets/noisy-speech-database-for-training-speech-enhancement-algorithms-and-tts-models(60d13dd9-9f7d-41f8-8743-dafc20078b43).html)
3. [Hohmann V. Frequency analysis and synthesis using a Gammatone filterbank. Acta Acustica united with Acustica. 2002 May 1;88(3):433-42.](https://www.researchgate.net/publication/230554893_Frequency_analysis_and_synthesis_using_a_Gammatone_filterbank)
4. [Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative adversarial nets." In Advances in neural information processing systems, pp. 2672-2680. 2014.](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
