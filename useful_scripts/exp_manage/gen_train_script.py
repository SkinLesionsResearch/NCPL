import os


def gen_exp_train_script(dset_path="./data/semi_processed",
                         labeled_num=500, threshold=0.99,
                         wafm=0.5, wu=0.5, num_classes=7, suffix=None):
    if suffix is None:
        print("please input valid suffix")
        return
    script_str = "\n\n"
    script_str += "python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \\\n"
    script_str += "--src-dset-path \'{dset_path}\' \\\n".format(dset_path=dset_path)
    script_str += "--labeled_num {labeled_num} \\\n".format(labeled_num=labeled_num)
    script_str += "--threshold {threshold} \\\n".format(threshold=threshold)
    script_str += "--weight-afm {wafm} \\\n".format(wafm=wafm)
    script_str += "--weight-u {wu} \\\n".format(wu=wu)
    script_str += "--num_classes {num_classes} \\\n".format(num_classes=num_classes)
    script_str += "--suffix \'{suffix}\' \\\n".format(suffix=suffix)

    script_str = script_str[:-3] + "\n"
    script_str += "run_check {suffix}".format(suffix=suffix)
    script_str += "\n"
    return script_str


with open("chain.script.base.sh", "r") as f:
    base_script = f.read()
    with open("chain.generated.sh", "w") as out_f:
        out_f.write(base_script)
        out_f.write(gen_exp_train_script(labeled_num=500, suffix="plgs_500_099", threshold=0.99))
        out_f.write(gen_exp_train_script(labeled_num=500, suffix="plgs_500_075", threshold=0.75))
        out_f.write(gen_exp_train_script(labeled_num=500, suffix="plgs_500_065", threshold=0.65))
        out_f.write(gen_exp_train_script(labeled_num=500, suffix="plgs_500_055", threshold=0.55))
        out_f.write(gen_exp_train_script(labeled_num=500, suffix="plgs_500_050", threshold=0.50))

        out_f.write(gen_exp_train_script(labeled_num=1000, suffix="plgs_1000_099", threshold=0.99))
        out_f.write(gen_exp_train_script(labeled_num=1000, suffix="plgs_1000_075", threshold=0.75))
        out_f.write(gen_exp_train_script(labeled_num=1000, suffix="plgs_1000_065", threshold=0.65))
        out_f.write(gen_exp_train_script(labeled_num=1000, suffix="plgs_1000_055", threshold=0.55))
        out_f.write(gen_exp_train_script(labeled_num=1000, suffix="plgs_1000_050", threshold=0.50))

        out_f.write(gen_exp_train_script(labeled_num=1500, suffix="plgs_1500_099", threshold=0.99))
        out_f.write(gen_exp_train_script(labeled_num=1500, suffix="plgs_1500_075", threshold=0.75))
        out_f.write(gen_exp_train_script(labeled_num=1500, suffix="plgs_1500_065", threshold=0.65))
        out_f.write(gen_exp_train_script(labeled_num=1500, suffix="plgs_1500_055", threshold=0.55))
        out_f.write(gen_exp_train_script(labeled_num=1500, suffix="plgs_1500_050", threshold=0.50))

        out_f.write(gen_exp_train_script(labeled_num=2000, suffix="plgs_2000_099", threshold=0.99))
        out_f.write(gen_exp_train_script(labeled_num=2000, suffix="plgs_2000_075", threshold=0.75))
        out_f.write(gen_exp_train_script(labeled_num=2000, suffix="plgs_2000_065", threshold=0.65))
        out_f.write(gen_exp_train_script(labeled_num=2000, suffix="plgs_2000_055", threshold=0.55))
        out_f.write(gen_exp_train_script(labeled_num=2000, suffix="plgs_2000_050", threshold=0.50))

        out_f.write(gen_exp_train_script(labeled_num=2500, suffix="plgs_2500_099", threshold=0.99))
        out_f.write(gen_exp_train_script(labeled_num=2500, suffix="plgs_2500_075", threshold=0.75))
        out_f.write(gen_exp_train_script(labeled_num=2500, suffix="plgs_2500_065", threshold=0.65))
        out_f.write(gen_exp_train_script(labeled_num=2500, suffix="plgs_2500_055", threshold=0.55))
        out_f.write(gen_exp_train_script(labeled_num=2500, suffix="plgs_2500_050", threshold=0.50))
