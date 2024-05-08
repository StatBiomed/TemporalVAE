# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：starts_exps_mouse_kFoldOn_oneDataset.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/4 17:32 

cd /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/
source ~/.bashrc
nohup python -u project_mouse_Yijun/starts_exps_mouse_kFoldOn_oneDataset.py >> logs/starts_exps_mouse_kFoldOn_oneDataset.log 2>&1 &
"""
import os, sys
import time
import yaml
import argparse


def main():
    # time.sleep(60*30)
    dataset_list = [
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_hvg1000_patternhvg+atlas_hvg_ofLiver/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_hvg1000_patternhvg+atlas_hvg_ofHeart/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_hvg1000_patternhvg+atlas_hvg_ofBrain/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_hvg1000_ofBrain/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_hvg1000_ofHeart/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_hvg1000_ofLiver/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofHeart/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofLiver/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofMesothelium/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofAdiposetissue/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofDorsalrootganglion/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofMeninges/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofLung/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofJawandtooth/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofEpidermis/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofSmoothmuscle/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofCartilage/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofKidney/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofGItract/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofCavity/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofConnectivetissue/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofMuscle/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofSpinalcord/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50/",
        "/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/",
        # "mouse_embryonic_development/preprocess_adata_JAX_dataset_4",
        # "mouse_embryonic_development/preprocess_adata_JAX_dataset_2",
        # "mouse_embryonic_development/preprocess_adata_JAX_dataset_3",
        # "mouse_embryonic_development/preprocess_adata_JAX_dataset_1"
    ]

    min_gene_num_list = [100]
    epoch_num = 100
    batch_size = 100000
    python_file = "project_mouse_Yijun/VAE_mouse_kFoldOn_oneDataset.py"
    time_type_list_re = ["embryoneg5to5"]
    result_save_path = f"240328_kFold_mouse_atlas_data_onlyTestTime"
    use_checkpoint_bool = "False"  # 2024-03-21 14:12:55 take care to use the pre-train model

    # -------------------------------------------------------------------------------------------------------------------
    cmd_list = []
    print("start regression model")
    # vae_param_file_list_re = ["supervise_vae_regressionclfdecoder", "supervise_vae_regressionclfdecoder_mouse_stereo"]
    vae_param_file_list_re = ["supervise_vae_regressionclfdecoder_mouse_stereo"]
    # time_type_list = ["log2", "0to1", "neg1to1", "sigmoid", "logit"]
    for time_type in time_type_list_re:
        for vae_param_file in vae_param_file_list_re:
            for dataset in dataset_list:
                for min_gene_num in min_gene_num_list:
                    args = dict()
                    args["vae_param_file"] = vae_param_file
                    args["file_path"] = dataset
                    args["time_standard_type"] = time_type
                    args["train_epoch_num"] = str(epoch_num)
                    args["result_save_path"] = result_save_path
                    args["min_gene_num"] = str(min_gene_num)
                    args["batch_size"] = str(batch_size)
                    args["use_checkpoint_bool"] = use_checkpoint_bool
                    cmd = "nohup python -u " + python_file
                    for key, value in args.items():
                        cmd = cmd + " --" + str(key) + "=" + str(value)
                    log_file_name = "_".join(args.values()).replace("/", "_").replace(" ", "_").replace("__", "_") + ".log"
                    while len(log_file_name) >= 254:
                        log_file_name = log_file_name[1:]
                    # cmd = cmd + f" --train_whole_model  > logs/{log_file_name} 2>&1 &"
                    cmd = cmd + f" --train_whole_model --kfold_test  > logs/{log_file_name} 2>&1 &"

                    cmd_list.append(cmd)
    for i in range(len(cmd_list)):
        print("#--- Start: {}/{}".format(i + 1, len(cmd_list)))
        print(cmd_list[i])
        # os.system(cmd_list[i])
        # time.sleep(60)
        # if (i + 1) % 3 == 0:
        #     time.sleep(60 * 30)
        # if (i + 1) % 6 == 0:
        #     time.sleep(60 * 30)
        # if (i + 1) % 9 == 0:
        #     time.sleep(60 * 20)

        sys.stdout.flush()


if __name__ == "__main__":
    main()
