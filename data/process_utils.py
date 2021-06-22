import pandas as pd


def process_file(base_path, src_file_path, target_file_path):
    def process_row(row):
        path = "./" + base_path + "/" + row.split("\\")[5]
        return path

    def process_label(label):
        return label + 1

    input_df = pd.read_csv(src_file_path, delimiter=" ", index_col=None, header=None)
    input_df.iloc[:, 0] = input_df.iloc[:, 0].map(process_row)
    with open(target_file_path, 'w') as f:
        input_df.to_csv(target_file_path, sep=" ", index=False, header=None)
