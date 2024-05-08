# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：starts_exps_mouse_kFoldOnStereoAndAtlas_withAdversarial.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/30 18:49

nohup python -u starts_exps_mouse_kFoldOnStereoAndAtlas_withAdversarial.py >> logs/starts_exps_trainOn_mouse_adversarial.log 2>&1 &

"""
import os, sys
import time
import argparse
import subprocess

def main():
    # time.sleep(60*5)
    fast_check_bool = False
    dataset_list = ["/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/"]
    external_dataset_list = ["/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/"]
    result_save_path = f"/231001_mouse_atlas_stereo_kFold_adversarial_{str(fast_check_bool)}FastCheck_CyclicLR/"
    vae_param_file_list_re = [
        "supervise_vae_regressionclfdecoder_adversarial0121212_231001_01",
        "supervise_vae_regressionclfdecoder_adversarial0121212_231001_02",
        # "supervise_vae_regressionclfdecoder_adversarial_0816_0121212_version0912_lessCell_dropoutLayer",
        # "supervise_vae_regressionclfdecoder_adversarial_0816_0121212_version0913_lessCell_dropoutLayer",
        # "supervise_vae_regressionclfdecoder_adversarial_0816_0121212_version091402_lessCell_dropoutLayer",
        # "supervise_vae_regressionclfdecoder_adversarial_0816_0121212_version0914_dropoutLayer",
        # "supervise_vae_regressionclfdecoder_adversarial_0816_0121212_version091203_lessCell_dropoutLayer"
    ]
    adversarial_train_bool_list = ["True"]
    epoch_num = 300
    if fast_check_bool:
        python_file = "VAE_mouse_kFoldOnStereoAndAtlas_withOrWithoutAdversarial_lessCellForFastCheck.py"
    else:
        python_file = "VAE_mouse_kFoldOnStereoAndAtlas_withOrWithoutAdversarial.py"

    time_type_list_re = ["embryoneg5to5"]
    parallel_bool = "True"
    works = 4
    cmd_list = []
    print("start regression model")

    for adversarial_train_bool in adversarial_train_bool_list:
        for time_type in time_type_list_re:
            for vae_param_file in vae_param_file_list_re:
                for external_dataset in external_dataset_list:
                    for dataset in dataset_list:
                        args = dict()
                        args["vae_param_file"] = vae_param_file
                        args["file_path"] = dataset
                        args["external_test_path"] = external_dataset
                        args["time_standard_type"] = time_type
                        args["train_epoch_num"] = str(epoch_num)
                        args["result_save_path"] = result_save_path
                        args["adversarial_train_bool"] = adversarial_train_bool
                        args["parallel_bool"] = parallel_bool
                        args["works"] = works
                        cmd = "nohup python -u " + python_file
                        for key, value in args.items():
                            cmd = cmd + " --" + str(key) + "=" + str(value)

                        cmd = cmd + f" > logs/{args['vae_param_file']}_{args['train_epoch_num']}Epoch_{str(fast_check_bool)}FastCheck.log" + " 2>&1 &"
                        # cmd = cmd + " > logs/" + "_".join(args.values()).replace("/", "_").replace(" ", "_").replace("_","").replace("__","_") + ".log" + " 2>&1 &"
                        cmd_list.append(cmd)
    for i in range(len(cmd_list)):
        print(f"--- Start: {i+1}/{len(cmd_list)}")

        # process = subprocess.run(cmd_list[i], shell=True)
        os.system(cmd_list[i])
        # pid = process.pid
        # print(f"PID:{pid}")
        print(f"{cmd_list[i]}")


        # time.sleep(60)
        # if (i + 1) % 2 == 0:
        #     time.sleep(60 * 20)
        # if (i + 1) % 6 == 0:
        #     time.sleep(60 * 20)
        # if (i + 1) % 9 == 0:
        #     time.sleep(60 * 20)

        sys.stdout.flush()


if __name__ == "__main__":
    main()
