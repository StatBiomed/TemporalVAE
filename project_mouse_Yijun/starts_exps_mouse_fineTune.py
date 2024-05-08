# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：starts_exps_mouse_fineTune.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/12/7 21:55 

cd /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan
source ~/.bashrc
nohup python -u project_mouse_Yijun/starts_exps_mouse_fineTune.py >> logs/starts_exps_mouse_fineTune.log 2>&1 &

"""
import os, sys
import time
import yaml
import argparse
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils/")
from utils.GPU_manager_pytorch import check_memory


def main():
    # time.sleep(60*60)
    # python_file = "project_mouse_Yijun/VAE_mouse_fineTune_Train_on_U_concat_S.py"
    # special_path_str = "231214_trainOnUconcatS_withMoment_"
    python_file = "project_mouse_Yijun/VAE_mouse_fineTune_Train_on_U_pairs_S.py"
    special_path_str = "240420_trainOnX_withMoment_finetune_withoutUtInLoss_withRNAVelocity"
    # special_path_str = "240318_trainOnX_withMoment_finetune_withoutUtInLoss_withRNAVelocity"

    sc_file_name_list = [
        # "240108mouse_embryogenesis/hematopoiesis",
        "240108mouse_embryogenesis/neuron",
        # "MouseTracheaE16_muhe/E16_Dec7v3_merged",
        # "mouse_erythoid/erythroid_lineage",
    ]
    fine_tune_mode_list = [
        # "withoutCellType",
        # "withCellType",
        # "withMoreFeature",
        "focusEncoder"
    ]
    # clf_weight_list = [0.1]
    clf_weight_list = [0.1,0.2,0.3,0.4, 0.5]
    detT_list = [0.01, 0.001, 0.1, 1]
    # detT_list = [0.001, 0.001]

    # -------------------------------------------------------------------------------------------------------------------
    cmd_list = []

    for clf_weight in clf_weight_list:
        for sc_file_name in sc_file_name_list:
            for fine_tune_mode in fine_tune_mode_list:
                for detT in detT_list:
                    args = dict()
                    args["special_path_str"] = special_path_str
                    args["sc_file_name"] = sc_file_name
                    args["fine_tune_mode"] = fine_tune_mode
                    args["clf_weight"] = str(clf_weight)
                    args["detT"] = str(detT)

                    cmd = "nohup python -u " + python_file
                    for key, value in args.items():
                        cmd = cmd + " --" + str(key) + "=" + str(value)
                    log_file_name = "_".join(args.values()).replace("/", "_").replace(" ", "_").replace("__", "_") + ".log"
                    while len(log_file_name) >= 254:
                        log_file_name = log_file_name[1:]
                    cmd = cmd + f" > logs/data{log_file_name} 2>&1 &"

                    cmd_list.append(cmd)
    for i in range(len(cmd_list)):
        print("#--- Start: {}/{}".format(i + 1, len(cmd_list)))
        print(cmd_list[i])
        check_memory(free_thre=10, max_attempts=1000000)
        os.system(cmd_list[i])
        time.sleep(60)
        # if i > 2:
        #     time.sleep(60 * 5)
        if i == 5:
            time.sleep(60 * 15)
        if i == 10:
            time.sleep(60 * 15)
        if i == 15:
            time.sleep(60 * 15)
        if i == 20:
            time.sleep(60 * 15)
        # if (i + 1) % 10 == 0:
        #     time.sleep(60 * 60)
        # if (i + 1) % 9 == 0:
        #     time.sleep(60 * 20)

        sys.stdout.flush()


if __name__ == "__main__":
    main()
