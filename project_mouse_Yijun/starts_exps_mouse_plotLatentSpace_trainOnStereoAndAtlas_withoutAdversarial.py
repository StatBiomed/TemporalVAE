# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：starts_exps_mouse_plotLatentSpace_trainOnStereoAndAtlas_withoutAdversarial.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/10/10 21:35 

cd /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/
source ~/.bashrc
nohup python -u project_mouse_Yijun/starts_exps_mouse_plotLatentSpace_trainOnStereoAndAtlas_withoutAdversarial.py > logs/starts_exps_mouse_plotLatentSpace_trainOnStereoAndAtlas_withoutAdversarial.log 2>&1 &

"""
import os, sys
import time
import yaml
import argparse
def main():
    # time.sleep(60*60*10)
    dataset_list = ["/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/"]
    external_dataset_list = [
        "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofHeart/",
        # "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofLiver/",
    ]
    min_gene_num = 100
    result_save_path = f"/231127_plotLatentSpace_trainOnStereoAndAtlas_minGeneNum{min_gene_num}_withoutAdversarial/"
    vae_param_file_list_re = ["supervise_vae_regressionclfdecoder_mouse_stereo"]
    adversarial_train_bool = "False"
    epoch_num = 300
    python_file = "project_mouse_Yijun/VAE_mouse_plotLatentSpace_trainOnStereoAndAtlas_withOrWithoutAdversarial.py"

    time_type_list_re = ["embryoneg5to5"]
    parallel_bool = "False"
    works = 3
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
                    args["min_gene_num"] = str(min_gene_num)
                    cmd = "nohup python -u " + python_file
                    for key, value in args.items():
                        cmd = cmd + " --" + str(key) + "=" + str(value)

                    cmd = cmd + f" > logs/trainOnAtlas_kFoldOnStereo_{args['vae_param_file']}_{args['train_epoch_num']}Epoch_minGene100_{adversarial_train_bool}Adversarial.log" + " 2>&1 &"
                    # cmd = cmd + " > logs/" + "_".join(args.values()).replace("/", "_").replace(" ", "_").replace("_","").replace("__","_") + ".log" + " 2>&1 &"
                    cmd_list.append(cmd)
    for i in range(len(cmd_list)):
        print(f"--- Start: {i + 1}/{len(cmd_list)}")

        # process = subprocess.run(cmd_list[i], shell=True)
        os.system(cmd_list[i])
        # pid = process.pid
        # print(f"PID:{pid}")
        print(f"{cmd_list[i]}")

        # time.sleep(60*60*1)
        # time.sleep(30)
        # if (i + 1) % 2 == 0:
        #     time.sleep(60 * 20)
        # if (i + 1) % 6 == 0:
        #     time.sleep(60 * 20)
        # if (i + 1) % 9 == 0:
        #     time.sleep(60 * 20)

        sys.stdout.flush()




if __name__ == "__main__":
    main()
