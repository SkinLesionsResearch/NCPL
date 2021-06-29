import pandas as pd


def process_file_two_cate(base_path, src_file_path, target_file_path):
    def process_img_path(row):
        path = "./" + base_path + "/" + row.split("\\")[5]
        return path

    def process_label(label):
        # change label to two categories, 1 for 4(mel), 0 for others
        return 1 if label == 4 else 0

    input_df = pd.read_csv(src_file_path, delimiter=" ", index_col=None, header=None)
    input_df.iloc[:, 0] = input_df.iloc[:, 0].map(process_img_path)
    input_df.iloc[:, 1] = input_df.iloc[:, 1].map(process_label)
    with open(target_file_path, 'w') as f:
        input_df.to_csv(target_file_path, sep=" ", index=False, header=None)


def process_file(base_path, src_file_path, target_file_path):
    def process_img_path(row):
        path = "./" + base_path + "/" + row.split("\\")[5]
        return path

    def process_label(label):
        return label + 1
    print("executed path: ", src_file_path)
    input_df = pd.read_csv(src_file_path, delimiter=" ", index_col=None, header=None)
    input_df.iloc[:, 0] = input_df.iloc[:, 0].map(process_img_path)
    with open(target_file_path, 'w') as f:
        input_df.to_csv(target_file_path, sep=" ", index=False, header=None)
