# PAV Speaker Classification and Verification with Deep Neural Networks

This is an implementation of https://github.com/santi-pdp/pav_spkid_pytorch, The objetcive of this project is to implement a speaker identifier using Neural Networks over a previously calculated feature vector. Additionally to the classifier function for which that program was designed, a verification function has been added.

## Trainning
The Trainning in our program is performed though the command : `FEAT=<feature> run_spkid train_nn`. Where <feature> must be one of the three choices for the feaure vector we implemented:
  - `lp` for Linear Prediction Coeficients.
  - `lpcc` for Linear Prediction Cespstral Coeficients.
  - `mfcc`for Mel Frequency Cepstral Coeficientes.
  
Training a neural network is not an easy task, as overfitting and underfitting problems are pretty common [6]. The objective is to design a model that is able to achive the best results on a given task and that can be later generalized. The problems faced here are those of escaping local minima and dealing with the previously mentioned overfitting and underfitting problems. 

For our proposed model, we used 256 hidden layers with 20 feature units per layer. The idea is that a deeper network will allow to use more non-linear information. A deeper or wider network did not improve the results, actually, as there are more parameters to optimize, it turned out to perform worse. A bigger network also proved to suffer from overfitting, which we could detect on the training results. When overfitting, the loss and accuracy of the training data will shoot up, getting up to near 0 loss and 100% accuracy. At the same time, validation loss will start to grow, and validation accuracy will not have the same 100% value. This means that the networks is "memorizing" the training data, and so not able to propely generalize what it has learned. More on that topic on []. An example of overfitting case would look something like this:

<img src="log_plots_overfitting.png" align="center">

We performed the trainning with a batch size of 1000 using the Adam optimizer and a 0.0001 learning rate. The data we used is that of the SPEECOn database [1], which was divided on a 75% training, 12.5% validation and 12.5% test. The results of the trainning process can be observed on the following plot:

<img src="log_plots.png" align="center">
  
### CUDA Acceleration
To train the model, we needed more performance than the CPU can offer, as training these models can take quite a lot of teime, especially when you take on consideration the amount of times you want to perform the training in order the achive the optimal hyperparameters. So, the best solution was to train it on our GPU, a Nvidia RTX 2060. By using the graphics card, we managed to reduce the computing teme between three and four times.

The problem we faced here was that we are using WSL 2 for programming on linux, whose support for GPUs is still being developed. For that reason, we had to update to the latest build on development, only available though Windows Insider Program (and on the dev channel). We do not recommend anyone this option currently, as this version may have instabitlities and some bugs. For more information about the procedure to install CUDA on WSL, follow this link: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

## Classification
The Classification task can be performed with the command `FEAT=<feature> run_spkid test_nn`. Then, the command `FEAT=<feature> run_spkid classerr_nn` must be executed to calculate the error rate of the classification.

With our model we obtained a 1.28% error rate.

## Verification
To Perform the verification, the command `FEAT=<feature> run_spkid verify_nn` must be run, followed by the command `FEAT=<feature> run_spkid verif_err_nn` to evaluate the verification results.

To implement this option, which was not present in the original `pav_spkid_pytorch` code, a modified version of the classification script was implemented. On this version, instead of returning the maximum of the last layer after the softmax algorithm, the script will return the value corresponding to the probability of a given class.

On the verification task, our model managed to achive a 0.0 cost detection score.


## Optimization


## Bibliography:

[1] ISKRA, Dorota, et al. Speecon-speech databases for consumer devices: Database specification and validation. 2002.

[2] G. Heigold, I. Moreno, S. Bengio and N. Shazeer, "End-to-end text-dependent speaker verification," 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Shanghai, 2016, pp. 5115-5119, doi: 10.1109/ICASSP.2016.7472652.

[3] Bing Xiang and T. Berger, "Efficient text-independent speaker verification with structural Gaussian mixture models and neural network," in IEEE Transactions on Speech and Audio Processing, vol. 11, no. 5, pp. 447-456, Sept. 2003, doi: 10.1109/TSA.2003.81582

[4] D. Snyder, P. Ghahremani, D. Povey, D. Garcia-Romero, Y. Carmiel and S. Khudanpur, "Deep neural network-based speaker embeddings for end-to-end speaker verification," 2016 IEEE Spoken Language Technology Workshop (SLT), San Diego, CA, 2016, pp. 165-170, doi: 10.1109/SLT.2016.7846260.

[5] SNYDER, David, et al. Deep Neural Network Embeddings for Text-Independent Speaker Verification. En Interspeech. 2017. p. 999-1003.

[6] GOODFELLOW, Ian J.; VINYALS, Oriol; SAXE, Andrew M. Qualitatively characterizing neural network optimization problems. arXiv preprint arXiv:1412.6544, 2014.

[7] D. Stathakis (2009) How many hidden layers and nodes?, International Journal of Remote Sensing, 30:8, 2133-2147, DOI: 10.1080/01431160802549278

[8] M. Bahaghighat, F. Abedini, M. S’hoyan and A. Molnar, "Vision Inspection of Bottle Caps in Drink Factories Using Convolutional Neural Networks," 2019 IEEE 15th International Conference on Intelligent Computer Communication and Processing (ICCP), Cluj-Napoca, Romania, 2019, pp. 381-385, doi: 10.1109/ICCP48234.2019.8959737.

--

//Tournament

YANG, Jiaping; SOH, Chee Kiong. Structural optimization by genetic algorithms with tournament selection. Journal of Computing in Civil Engineering, 1997, vol. 11, no 3, p. 195-200.

//PSO

BAI, Qinghai. Analysis of particle swarm optimization algorithm. Computer and information science, 2010, vol. 3, no 1, p. 180.

CHEN, Ching-Yi; YE, Fun. Particle swarm optimization algorithm and its application to clustering analysis. En 2012 Proceedings of 17th Conference on Electrical Power Distribution. IEEE, 2012. p. 789-794.

//ABSO

AKBARI, Reza; MOHAMMADI, Alireza; ZIARATI, Koorush. A novel bee swarm optimization algorithm for numerical function optimization. Communications in Nonlinear Science and Numerical Simulation, 2010, vol. 15, no 10, p. 3142-3155.

AKBARI, Reza; MOHAMMADI, Alireza; ZIARATI, Koorush. A novel bee swarm optimization algorithm for numerical function optimization. Communications in Nonlinear Science and Numerical Simulation, 2010, vol. 15, no 10, p. 3142-3155.

