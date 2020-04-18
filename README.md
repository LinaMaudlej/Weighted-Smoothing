# Weighted-Smoothing
Deep Learning Project 236781.

Paper: https://arxiv.org/pdf/1911.07198.pdf
Repo: https://github.com/yanemcovsky/SIAM


A README file (plain text/markdown) explaining:
The structure of the code in the src/ folder: What is implemented in each package/module.
Steps to reproduce your results using this code: Where to get and place the data, how to run all the data processing steps, how to run training and evaluation

# Implementation folders and what is implemented
####  src/run_attack.py 
The basic implementation for the flags inputs:
--vote_thresh 
--k_predic 
--close_pred_thresh
--repeat
####  src/train.py
The basic implementation for the flags inputs:
--vote_thresh 
--k_predic 
--close_pred_thresh
--repeat

####  src/models/resnet_cpni_smooth_predict.py
We implemented  ResNet_Cifar_FilteredMonteCarlo class, that inherits from the ResNet_Cifar regular class its methods, overrides the method  monte_carlo_predict with our code. 
The main difference is that the class gets new input class w_assign_fn, FilterByThresholdSoftmax and FilterByThresholdKPredictions and close prediction, which calculates the threshold and the k predictions and the cpt. 

# Running steps
After regular training as mentioned in the repository https://github.com/yanemcovsky/SIAM

#### 1) run the adversrial attack with first method ( --vote_thresh)
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --vote_thresh <threshold> --attack epgd --epochs <epochs_num> --m_test  <test_num> --m_train <train_num> --noise_sd <noise_num> --repeat  <repeat_num> --resume <trained_path> 

#### Example for noise_sd range [0,0.5]:
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict  --vote_thresh 0.05 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.1 --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/Threshold_outputs/out1_t005_attack.txt  
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict  --vote_thresh 0.05 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.2 --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/Threshold_outputs/out2_t005_attack.txt  
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict  --vote_thresh 0.05 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.3 --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/Threshold_outputs/out3_t005_attack.txt  
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict  --vote_thresh 0.05 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.4 --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/Threshold_outputs/out4_t005_attack.txt  
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict  --vote_thresh 0.05 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.5 --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/Threshold_outputs/out5_t005_attack.txt 

#### 2) run the adversrial attack with second method (--k_predic flag):
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic <k_num> --attack epgd --epochs <epochs_num> --m_test <test_num> --m_train <train_num> --noise_sd <noise_num>  --repeat <repeat_num> --resume <trained_path> 
#### Example for noise_sd range [0,0.5]:
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic 1 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.1  --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/KPredictions_outputs/out1_k1_attack.txt
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic 1 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.2  --repeat 5 --resume src/results/2020-04-07_19-21-44 > Weighted-Smoothing/results\ analysis/KPredictions_outputs/out2_k1_attack.txt
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic 1 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.3  --repeat 5 --resume src/results/2020-04-08_11-43-28 > Weighted-Smoothing/results\ analysis/KPredictions_outputs/out3_k1_attack.txt
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic 1 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.4  --repeat 5 --resume src/results/2020-04-09_03-59-23 > Weighted-Smoothing/results\ analysis/KPredictions_outputs/out_k1_attack.txt
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic 1 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.5  --repeat 5 --resume src/results/2020-04-09_21-48-05 > Weighted-Smoothing/results\ analysis/KPredictions_outputs/out5_k1_attack.txt


#### 3) run the adversrial attack with first method (--close_pred_thresh)
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --close_pred_thresh <cpt> --attack epgd --epochs <epochs_num> --m_test  <test_num> --m_train <train_num> --noise_sd <noise_num> --repeat  <repeat_num> --resume <trained_path> 
#### Example for noise_sd range [0,0.5]:
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict  --vote_thresh 0.05 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.1 --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/ClosePrediction_outputs/out1_cpt05_attack.txt  
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict  --vote_thresh 0.05 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.2 --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/ClosePrediction_outputs/out2_cpt05_attack.txt  
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict  --vote_thresh 0.05 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.3 --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/ClosePrediction_outputs/out3_cpt05_attack.txt  
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict  --vote_thresh 0.05 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.4 --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/ClosePrediction_outputs/out4_cpt05_attack.txt  
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict  --vote_thresh 0.05 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.5 --repeat 5 --resume src/results/2020-04-07_03-22-06 > Weighted-Smoothing/results\ analysis/ClosePrediction_outputs/out5_cpt05_attack.txt 

# Results and Evaluations 
#### 1) Results for first method: 
    Weighted-Smoothing/results\ analysis/Threshold_outputs 
    python3 threshold_draw.py
#### 2) Results for second method: 
    Weighted-Smoothing/results\ analysis/KPredictions_outputs 
    python3 kpredictions_draw.py
#### 3) Results for third method: 
    Weighted-Smoothing/results\ analysis/ClosePrediction_outputs/
    python3 close_draw.py



