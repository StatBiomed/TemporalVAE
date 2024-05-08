# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：starts_exps_mouse_kFoldOnStereoAndAtlas_withAdversarial.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/30 18:49

nohup python -u starts_exps_mouse_kFoldOnStereoAndAtlas_withoutAdversarial.py > logs/starts_exps_mouse_kFoldOnStereoAndAtlas_withoutAdversarial.log 2>&1 &

"""
import os, sys
import time
import yaml
import argparse


def main():
    # dataset_list = ["/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene50_minCell50_hvg1000/"]
    dataset_list = ["/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/"]
    external_dataset_list = ["/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/"]
    result_save_path = "/231004_kFoldOnStereo_trainOnAtlasSubStereo_noAdversarial/"
    result_save_path = "/240416_noAdversarial_testOnMouseStereo_withBatch200000/"
    vae_param_file_list_re = ["supervise_vae_regressionclfdecoder_mouse_stereo"]
    # vae_param_file_list_re = ["supervise_vae_regressionclfdecoder_mouse_atlas_stereo_noAdversarial"]
    # vae_param_file_list_re = ["supervise_vae_regressionclfdecoder"]
    adversarial_train_bool="False"
    epoch_num = 100
    # python_file = "VAE_mouse_kFoldOnStereoAndAtlas_withOrWithoutAdversarial_lessCellForFastCheck.py"
    python_file = "project_mouse_Yijun/VAE_mouse_kFoldOnStereoAndAtlas_withOrWithoutAdversarial.py"
    time_type_list_re = ["embryoneg5to5"]
    parallel_bool = "True"
    works = 4
    cmd_list = []
    print("start regression model")

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
                    cmd = cmd + " > logs/" + args["vae_param_file"] + ".log" + " 2>&1 &"
                    # cmd = cmd + " > logs/" + "_".join(args.values()).replace("/", "_").replace(" ", "_").replace("_","").replace("__","_") + ".log" + " 2>&1 &"
                    cmd_list.append(cmd)
    for i in range(len(cmd_list)):
        print("#--- Start: {}/{}".format(i, len(cmd_list)))
        print(cmd_list[i])
        # os.system(cmd_list[i])
        # time.sleep(60)
        # if (i + 1) % 3 == 0:
        #     time.sleep(60 * 20)
        # if (i + 1) % 6 == 0:
        #     time.sleep(60 * 20)
        # if (i + 1) % 9 == 0:
        #     time.sleep(60 * 20)
        #
        # sys.stdout.flush()


if __name__ == "__main__":
    main()
