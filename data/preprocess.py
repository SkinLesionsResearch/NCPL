from process_utils import process_file
from pathlib import Path

rootdir = "./semi/train"
target_dir = "./semi_processed"
target_base_path = "../ups/data_ham10000/datasets"


def replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail

test_path = Path("./semi/test.txt")
if test_path.is_file():
    test_file_path = str(test_path.absolute())
    target_file_path = replace_last(test_file_path, "semi", "semi_processed")
    print(test_file_path)
    process_file(target_base_path, test_file_path, target_file_path) 
    print(target_file_path)
"""
for path in Path("./semi/train").iterdir():
    if path.is_dir():
        train_labeled_path = path.joinpath("train_labeled.txt")
        if train_labeled_path.is_file():
            src_file_path = str(train_labeled_path.absolute())
            print(src_file_path)
            target_file_path = replace_last(src_file_path, "semi", "semi_processed")
            print(target_file_path)
            process_file(target_base_path, src_file_path, target_file_path)
        train_unlabeled_path = path.joinpath("train_unlabeled.txt")
        if train_unlabeled_path.is_file():
            src_file_path = str(train_unlabeled_path.absolute())
            print(src_file_path)
            target_file_path = replace_last(src_file_path, "semi", "semi_processed")
            print(target_file_path)
            process_file(target_base_path, src_file_path, target_file_path)
            print(train_unlabeled_path)
"""
