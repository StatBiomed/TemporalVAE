# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：starts_exps_vae_humanEmbryo_kFold.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/4 17:32 

cd /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/
source ~/.bashrc
nohup python -u project_mouse_Yijun/starts_exps_vae_humanEmbryo_kFold.py >> logs/starts_exps_vae_humanEmbryo_kFold.py 2>&1 &


"""
import os, sys
import time
import yaml
import argparse


def main():
    # -------------------------------------------------------------------------------------------------------------------
    for i in range(90, 150):
        cmd = f"nohup python -u project_mouse_Yijun/vae_humanEmbryo_adversarial_mulitDataset.py --result_save_path 240409_human_2data_inter --train_epoch_num {i} >> logs/240409_human_2data_inter_epoch{i}.log 2>&1 &"
        print(cmd)
        os.system(cmd)
        time.sleep(20)
        if (i + 1) % 5 == 0:
            time.sleep(60)
        if i > 120:
            time.sleep(60)

        sys.stdout.flush()


if __name__ == "__main__":
    main()
