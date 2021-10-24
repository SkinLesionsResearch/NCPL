from pylatex import Document, Section, Subsection, Table, Tabular
from pylatex import MultiRow, MultiColumn
from pylatex.utils import italic, NoEscape
import pandas as pd
import numpy as np
import os


def create_tabular(df):
    df_max_acc_idx = df.groupby(by=["num_labeled", "lambda_param", "w"])["Accuracy"].idxmax()
    df_max_acc = df.iloc[df_max_acc_idx].reset_index(drop=True)

    tabular = Tabular(table_spec="|c|c|c|l|l|l|l|l|")
    tabular.add_hline()

    tabular.add_row(["The number of labeled samples",
                     NoEscape("$\eta$"), NoEscape("$w$"), "Accuracy", "Kappa",
                     NoEscape("F1-score"), "Recall", "Precision"])
    tabular.add_hline()

    labeled_num_set = [500, 1000, 1500, 2000]
    for num_labeled_cur in labeled_num_set:
        df_cur = df_max_acc[df_max_acc["num_labeled"] == num_labeled_cur].reset_index(drop=True)

        for i in range(0, len(df_cur)):
            cur_row = [""] if i > 0 else [MultiRow(size=len(df_cur), data=num_labeled_cur)]
            cur_row.extend(list(df_cur.iloc[i, 1:]))
            tabular.add_row(cur_row)
            # tabular.add_hline() if i == len(df_cur) - 1 else tabular.add_hline(start=2, end=8)
            if i == len(df_cur) - 1:
                tabular.add_hline()
    return tabular


def create_table(table_name):
    table = Table()
    table.add_caption(table_name)

    tabular = create_tabular()
    table.append(tabular)
    return table


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)


def fill_document(doc):
    # table = create_table("Parameter Analysis - ACFI Module")
    obj = create_tabular()
    obj.generate_tex("tabular.example")



doc = Document('table.example')
df = pd.read_csv('acfi_res.csv')
df["num_labeled"] = df["num_labeled"].astype(np.dtype("int64"))
tabular = create_tabular(df)
tabular.generate_tex('tabular.example')
inplace_change("tabular.example.tex", "%", "")
# fill_document(doc)
# doc.generate_pdf(clean_tex=False)
# doc.generate_tex()
