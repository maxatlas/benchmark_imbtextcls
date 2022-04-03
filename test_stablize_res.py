import pandas as pd

import task_utils
import vars
from tablize_res import *
import numpy as np
from sklearn.metrics import RocCurveDisplay
from scipy.stats import linregress, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols

idx = pd.IndexSlice


def df_anova1(df, axis, level):
    x = []
    for name, group in df.groupby(axis=axis, level=level):
        group = group.to_numpy().reshape(-1)
        group = group[~np.isnan(group)]
        x.append(group.tolist())

    return f_oneway(*x)


def df_lm(df, axis, level):
    x, y = [], []
    for name, group in df.groupby(axis=axis, level=level):
        group = group.to_numpy().reshape(-1)
        group = group[~np.isnan(group)]
        x += group.tolist()
        y += [name] * len(group.tolist())

    return linregress(x, y, alternative="two-sided")


def df_anova2(df, col_levels, row_levels):
    if type(col_levels) is not list or type(col_levels) is not tuple:
        col_levels = [col_levels]
    if type(row_levels) is not list or type(row_levels) is not tuple:
        row_levels = [row_levels]

    assert len(col_levels) + len(row_levels) == 2
    if not row_levels:
        return _df_anova2(df, 1, col_levels)
    elif not col_levels:
        return _df_anova2(df, 0, row_levels)
    else:
        if type(col_levels[0]) is int:
            col_levels = [df.columns.names[level] for level in col_levels]
        if type(row_levels[0]) is int:
            row_levels = [df.index.names[level] for level in row_levels]
        df_dict = {"x": [], col_levels[0]: [], row_levels[0]: []}
        for col_name, group in df.groupby(axis=1, level=col_levels):
            for row_name, group in group.groupby(axis=0, level=row_levels):
                group = group.to_numpy().reshape(-1)
                group = group[~np.isnan(group)]
                df_dict["x"] += group.tolist()
                df_dict[col_levels[0]] += [col_name] * len(group)
                df_dict[row_levels[0]] += [row_name] * len(group)
    df = pd.DataFrame(df_dict)

    model = ols('x ~ C(%s) + C(%s) + C(%s):C(%s)' % (col_levels[0], row_levels[0], col_levels[0], row_levels[0]),
                data=df).fit()
    out = sm.stats.anova_lm(model, type=2)
    return out


def _df_anova2(df, axis, levels):
    assert (type(levels) is list or type(levels) is tuple) and len(levels) == 2
    df_dict = {"x": []}
    if type(levels[0]) is int:
        names = df.columns.names if axis == 1 else df.index.names
        levels = [names[level] for level in levels]
    for level in levels:
        df_dict[level] = []

    for name, group in df.groupby(axis=axis, level=levels):
        group = group.to_numpy().reshape(-1)
        group = group[~np.isnan(group)]
        df_dict['x'] += group.tolist()
        df_dict[levels[0]] += [name[0]] * len(group)
        df_dict[levels[1]] += [name[1]] * len(group)

    df = pd.DataFrame(df_dict)
    model = ols('x ~ C(%s) + C(%s) + C(%s):C(%s)' % (levels[0], levels[1], levels[0], levels[1]),
                data=df).fit()
    out = sm.stats.anova_lm(model, type=2)

    return out


def test_curve_display():
    import matplotlib.pyplot as plt
    roc = dill.load(open("results/banking77/bert.roc", "rb"))
    roc_list = list(list(roc.values())[0].values())[0]
    curves = get_roc_curve_by_class(*roc_list)
    display = RocCurveDisplay(**list(curves.values())[2])
    display.plot()
    roc_list = list(list(roc.values())[0].values())[1]
    curves = get_roc_curve_by_class(*roc_list)
    display = RocCurveDisplay(**list(curves.values())[2])
    display.plot()
    plt.show()


def test_matrix(df):
    df = df.loc[idx[:, :, "single", :, :, "AUC", :], idx[:, :, :, :, :, :]]
    return create_matrix_from_df(df, 4, 1)  # Loss function comparison


def get_df_ci(df, col_levels, row_levels):
    def _get_ci(x):
        x = x.to_numpy().reshape(-1)
        x = x[~np.isnan(x)]
        return (x.mean() + 1.96 * x.std() / np.sqrt(len(x)),
                x.mean() - 1.96 * x.std() / np.sqrt(len(x)))

    if type(col_levels) is int or type(col_levels[0]) is int:
        col_levels = [df.columns.names[l] for l in col_levels]
    if type(row_levels) is int or type(row_levels[0]) is int:
        row_levels = [df.index.names[l] for l in row_levels]

    out = task_utils.create_empty_df(row_levels, col_levels)
    for col_name, col in df.groupby(col_levels, axis=1):
        for row_name, group in col.groupby(row_levels):
            ci = _get_ci(group)
            out.loc[row_name, col_name] = ci
    out = out.dropna(0, "all")
    return out


if __name__ == "__main__":
    df = dill.load(open("merged.df", "rb"))
    df = df.dropna(0, "all")
    df = df.dropna(1, "all")
    df = df.loc[idx[:, :, :, :, :, "AUC", :], idx[:, :, :, :, :, :, :]]
    # index: dataset, cls_type, label_type, proofread, text_length, metrics, random_seed
    # column: model, num_layer, pretrained, balance_strategy, loss, qkv_size, tokenizer_pretrained
    # df = layer_value(df)
    # df = df_anova1(df, 0, 0)
    # df = df_anova2(df, 0, 0)
    # df = _df_anova2(df, axis=1, levels=[0, 4])
    # df = hypo_test(df, 4, 1)
    df = get_df_ci(df, col_levels=["num_layer", ""], row_levels=["proofread", "metrics"])
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