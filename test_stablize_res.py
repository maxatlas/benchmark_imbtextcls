import pandas as pd

import vars
from tablize_res import *
import numpy as np
from sklearn.metrics import RocCurveDisplay

idx = pd.IndexSlice


def test_curve_display():
    import matplotlib.pyplot as plt
    roc = dill.load(open("results/banking77/bert.roc", "rb"))
    roc_list = list(list(roc.values())[0].values())[0]
    get_roc_curve_by_class(*roc_list)
    curves = get_roc_curve_by_class(*roc_list)
    display = RocCurveDisplay(**list(curves.values())[2])
    display.plot()
    plt.show()


def test_matrix():
    df = get_res_multi_file("merged")
    df = df.loc[idx[:, :, "single", :, :, "AUC", :], idx[:, :, :, :, :, :]]
    return create_matrix_from_df(df, 4, 1)


def get_df_ci(df, col_levels, row_levels):
    def _get_ci(x):
        x = x.to_numpy().reshape(-1)
        x = x[~np.isnan(x)]
        return [(x.mean() + 1.96 * x.std() / np.sqrt(len(x)),
                x.mean() - 1.96 * x.std() / np.sqrt(len(x)))]

    out = []
    for name, col in df.groupby(col_levels, axis=1):
        rows = []
        for name, group in col.groupby(row_levels):
            ci = group.apply(_get_ci)
            cell_index = [[group.index[0][group.index.names.index(l)]] for l in row_levels]
            ci = ci.set_index(cell_index).rename_axis(row_levels)
            rows.append(ci)

        rows = pd.concat(rows, axis=0)
        out.append(rows)

    out = pd.concat(out, axis=1)

    return out


if __name__ == "__main__":
    # df = get_res_multi_file("merged")
    df = dill.load(open("merged.df", "rb"))
    df = get_df_ci(df, col_levels=["num_layer", ""], row_levels=["Metrics"])
    # test_curve_display()
    # test_curve_display()
    # folder = "../results_wiener"
    # df = get_res_multi_file("merged", sort=False)
    # df = test_get_metric_value()
    # df = get_res_per_file("merged/poem_sentiment/lstmattn")
    with pd.option_context('display.max_columns', None, 'display.max_rows', None):
        print(df)
    # out = df.agg(lambda l: get_ci(l.dropna()), axis=0)
    # with pd.option_context('display.max_columns', None, 'display.max_rows', None):
    #     print(out)