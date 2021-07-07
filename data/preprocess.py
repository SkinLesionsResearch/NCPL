from process_utils import process_file, process_file_two_cate
from pathlib import Path

rootdir = "./semi/train"
# target_dir = "./semi_processed"
# target_dir_name = "semi_processed_two_cates"
target_dir_name = "semi_processed_absolute"
target_data_base_path = "/home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer/data_ham10000/datasets"

# make the target transferred directories
target_test_path = Path(target_dir_name)
target_test_path.mkdir(exist_ok=True)
target_test_path = Path(target_dir_name + "/train")
target_test_path.mkdir(exist_ok=True)


def replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail


def process_test_data():
    test_path = Path("./semi/test.txt")

    if test_path.is_file():
        test_file_path = str(test_path.absolute())
        target_file_path = replace_last(test_file_path, "semi", target_dir_name)
        print(test_file_path)
        # process_file_two_cate(target_data_base_path, test_file_path, target_file_path)
        process_file(target_data_base_path, test_file_path, target_file_path)
        print(target_file_path)


def process_train_two_cate():
    for path in Path("./semi/train").iterdir():
        if path.is_dir():
            train_labeled_path = path.joinpath("train_labeled.txt")
            if train_labeled_path.is_file():
                src_file_path = str(train_labeled_path.absolute())
                print(src_file_path)
                target_file_path = replace_last(src_file_path, "semi", target_dir_name)
                Path(target_file_path).mkdir(exist_ok=True)
                print(target_file_path)
            train_unlabeled_path = path.joinpath("train_unlabeled.txt")
            if train_unlabeled_path.is_file():
                src_file_path = str(train_unlabeled_path.absolute())
                print(src_file_path)
                target_file_path = replace_last(src_file_path, "semi", target_dir_name)
                Path(target_file_path).mkdir(exist_ok=True)
                print(target_file_path)


def process_train_data():
    for path in Path("./semi/train").iterdir():
        print(path.absolute())
        target_section_path = Path(replace_last(str(path.absolute()), "semi", target_dir_name))
        target_section_path.mkdir(exist_ok=True)
        if path.is_dir():
            train_labeled_path = path.joinpath("train_labeled.txt")
            if train_labeled_path.is_file():
                src_file_path = str(train_labeled_path.absolute())
                print(src_file_path)
                target_file_path = replace_last(src_file_path, "semi", target_dir_name)
                print(target_file_path)
                # process_file_two_cate(target_data_base_path, src_file_path, target_file_path)
                process_file(target_data_base_path, src_file_path, target_file_path)
            train_unlabeled_path = path.joinpath("train_unlabeled.txt")
            if train_unlabeled_path.is_file():
                src_file_path = str(train_unlabeled_path.absolute())
                print(src_file_path)
                target_file_path = replace_last(src_file_path, "semi", target_dir_name)
                print(target_file_path)
                # process_file_two_cate(target_data_base_path, src_file_path, target_file_path)
                process_file(target_data_base_path, src_file_path, target_file_path)


def main():
    process_train_data()
    process_test_data()


if __name__ == '__main__':
    main()
