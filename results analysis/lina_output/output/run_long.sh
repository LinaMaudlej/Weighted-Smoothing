#!/bin/bash

# Setup env
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw

unset XDG_RUNTIME_DIR
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --vote_thresh 0.10 --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.1 > out_boris1.txt
python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --vote_thresh 0.10 --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.2 > out_boris2.txt
python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --vote_thresh 0.10 --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.3 > out_boris3.txt
python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --vote_thresh 0.10 --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.4 > out_boris4.txt
python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --vote_thresh 0.10 --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.5 > out_boris5.txt

python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --vote_thresh 0.05 --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.1 > out_boris1.txt
python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --vote_thresh 0.05 --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.2 > out_boris2.txt
python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --vote_thresh 0.05 --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.3 > out_boris3.txt
python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --vote_thresh 0.05 --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.4 > out_boris4.txt
python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --vote_thresh 0.05 --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.5 > out_boris5.txt


#python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 20 > out.txt

#python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 20 --m_test 8 --m_train 8 --noise_sd 0.2 > out1.txt

#python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 20 --m_test 16 --m_train 16 --noise_sd 0.2 > out2.txt

#python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 20 --m_test 16 --m_train 8 --noise_sd 0.2 > out3.txt

#################################

#python3 train.py -e --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 20 --m_test 8 --m_train 8 --noise_sd 0.2 --resume /home/lina.maudlej/project/SIAM/results/2020-03-25_20-29-06 > out1_e.txt

#python3 train.py -e --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 20 --m_test 16 --m_train 16 --noise_sd 0.2 --resume /home/lina.maudlej/project/SIAM/results/2020-03-26_00-25-40  > out2_e.txt

#python3 train.py -e --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 20 --m_test 16 --m_train 8 --noise_sd 0.2 --resume /home/lina.maudlej/project/SIAM/results/2020-03-26_07-54-48 > out3_e.txt

#################################

#python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --attack epgd --epochs 20 --m_test 8 --m_train 8 --noise_sd 0.2 --resume /home/lina.maudlej/project/SIAM/results/2020-03-25_20-29-06 > out1_e_attack.txt

#python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --attack epgd --epochs 20 --m_test 16 --m_train 16 --noise_sd 0.2 --resume /home/lina.maudlej/project/SIAM/results/2020-03-26_00-25-40  > out2_e_attack.txt

#python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --attack epgd --epochs 20 --m_test 16 --m_train 8 --noise_sd 0.2 --resume /home/lina.maudlej/project/SIAM/results/2020-03-26_07-54-48 > out3_e_attack.txt

#################################
#python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.1 > out_sd_1.txt
#python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.2 > out_sd_2.txt
#python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.3 > out_sd_3.txt
#python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.4 > out_sd_4.txt
#python3 train.py --weight-noise --adv --cpni  --smooth mcpredict --attack epgd --schedule 200 300 --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.5 > out_sd_5.txt

##################################
##2020-03-29_05-27-50/
##2020-03-29_23-47-12/

#python3 run_attack.py --weight-noise  --cpni  --smooth mcpredict --attack epgd --epochs 50 --m_test 16 --m_train 16 --noise_sd 0.2 --resume /home/lina.maudlej/project/SIAM/results/2020-03-29_04-20-12 > out1_e_attack.txt
