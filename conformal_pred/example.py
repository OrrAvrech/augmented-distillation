import torch
from arc.methods import SplitConformal
from arc.black_boxes import TabularAutoML, SVC
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    csv_path = Path(r"C:\Users\orrav\Documents\Datasets\Adult Census Income\adult_train.csv")
    test_csv_path = Path(r"C:\Users\orrav\Documents\Datasets\Adult Census Income\adult_aug_samples_test.csv")
    # model_path = r"C:\Users\orrav\Documents\Github\augmented-distillation\ck\adults_income\adults_income_features.pt"
    label = 'income'
    alpha = 0.1
    # cond_enc = torch.load(model_path)
    df = pd.read_csv(csv_path, index_col=0)
    x_test = pd.read_csv(test_csv_path, index_col=0)
    x = df.drop([label], axis=1)
    y = df[label]
    bb = TabularAutoML(label=label, metric='roc_auc')
    # bb = SVC()

    method_sc = SplitConformal(x, y, bb, alpha)
    s_sc = method_sc.predict(x_test)
    test_sc = [(xx, s_sc[i][0]) for i, xx in enumerate(x_test.to_numpy()) if len(s_sc[i]) == 1]
    x_test_sc, y_test_sc = zip(*test_sc)
    test_sc_df = pd.DataFrame(x_test_sc, columns=x_test.columns)
    test_sc_df[label] = y_test_sc
    test_sc_df.to_csv(test_csv_path.parent.joinpath("adult_test_sc_roc_auc.csv"))


if __name__ == '__main__':
    main()
