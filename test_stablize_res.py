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
    df = df.loc[idx[:, "AUC", :], idx[:, :, :, :, :, :]]
    return create_matrix_from_df(df, 0, 0)


if __name__ == "__main__":
    df=test_matrix()
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