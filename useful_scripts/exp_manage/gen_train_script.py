import os


def gen_exp_train_script(dset_path="./data/semi_processed",
                         labeled_num=500,
                         wafm=0.9, wu=0.1, num_classes=7, suffix="acfi_1000",
                         check_flag="run_check 1000_acfi_0901"):
    script_str = "\n\n"
    script_str += "python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \\\n"
    script_str += "--src-dset-path \'{dset_path}\' \\\n".format(dset_path=dset_path)
    script_str += "--labeled_num {labeled_num} \\\n".format(labeled_num=labeled_num)
    script_str += "--weight-afm {wafm} \\\n".format(wafm=wafm)
    script_str += "--weight-u {wu} \\\n".format(wu=wu)
    script_str += "--num_classes {num_classes} \\\n".format(num_classes=num_classes)
    script_str += "--suffix \'{suffix}\' \\\n".format(suffix=suffix)

    script_str = script_str[:-3] + "\n"
    script_str += check_flag
    script_str += "\n"
    return script_str


with open("chain.script.base.sh", "r") as f:
    base_script = f.read()
    with open("chain.generated.sh", "w") as out_f:
        out_f.write(base_script)
        out_f.write(gen_exp_train_script())
