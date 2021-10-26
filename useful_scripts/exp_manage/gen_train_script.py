import os


def gen_exp_train_script(dset_path="./data/semi_processed",
                         labeled_num=500, threshold=0.99, train_path="ckps",
                         wafm=0.5, wu=0.5, num_classes=7, suffix=None):
    if suffix is None:
        print("please input valid suffix")
        return
    script_str = "\n\n"
    script_str += "python -u /home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/object/train.py \\\n"
    script_str += "--src-dset-path \'{dset_path}\' \\\n".format(dset_path=dset_path)
    script_str += "--train_path \'{train_path}\' \\\n".format(train_path=train_path)
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
        out_f.write(gen_exp_train_script(labeled_num=500, suffix="neg_500_neg", train_path="ckps_neg"))
        out_f.write(gen_exp_train_script(labeled_num=1000, suffix="neg_1000_neg", train_path="ckps_neg"))
        out_f.write(gen_exp_train_script(labeled_num=1500, suffix="neg_1500_neg", train_path="ckps_neg"))
        out_f.write(gen_exp_train_script(labeled_num=2000, suffix="neg_2000_neg", train_path="ckps_neg"))
        out_f.write(gen_exp_train_script(labeled_num=2500, suffix="neg_2500_neg", train_path="ckps_neg"))
