# Weighted-Smoothing
Deep Learning Project 236781.

Paper: https://arxiv.org/pdf/1911.07198.pdf
Repo: https://github.com/yanemcovsky/SIAM


A README file (plain text/markdown) explaining:
The structure of the code in the src/ folder: What is implemented in each package/module.
Steps to reproduce your results using this code: Where to get and place the data, how to run all the data processing steps, how to run training and evaluation

# Implementation folders
####  src/run_attack.py 

####  src/train.py

####  src/models/resnet_cpni_smooth_predict.py

# Running steps
After regular training as mentioned in the repository https://github.com/yanemcovsky/SIAM

#### 1) run the adversrial attack with first method ( --)

#### 2) run the adversrial attack with second method ( --k_predic flag):
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic <k_num> --attack epgd --epochs <epochs_num> --m_test <test_num> --m_train <train_num> --noise_sd <noise_num>  --repeat <repeat_num> --resume <trained_path> 
#### Example for noise_sd range [0,0.5]:
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic 1 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.1  --repeat 5 --resume src/results/2020-04-07_03-22-06 > out_k1_attack.txt
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic 1 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.2  --repeat 5 --resume src/results/2020-04-07_19-21-44 > out_k1_attack.txt
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic 1 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.3  --repeat 5 --resume src/results/2020-04-08_11-43-28 > out_k1_attack.txt
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic 1 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.4  --repeat 5 --resume src/results/2020-04-09_03-59-23 > out_k1_attack.txt
    python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --k_predic 1 --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.5  --repeat 5 --resume src/results/2020-04-09_21-48-05 > out_k1_attack.txt

# Results and Evaluations 
#### 1) Results for first method: 
    Weighted-Smoothing/Threshold_outputs 
    python3 threshold_draw.py
#### 2) Results for second method: 
    Weighted-Smoothing/KPredictions_outputs 
    python3 kpredictions_draw.py



