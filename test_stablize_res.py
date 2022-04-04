from tablize_res import *
from sklearn.metrics import RocCurveDisplay

idx = pd.IndexSlice


def test_curve_display():
    import matplotlib.pyplot as plt
    roc = dill.load(open("results/ag_news/bert.roc", "rb"))
    roc_list = list(list(roc.values())[0].values())[0]
    curves = get_roc_curve_by_class(*roc_list)
    display = RocCurveDisplay(**list(curves.values())[2])
    display.plot()
    roc_list = list(list(roc.values())[0].values())[0]
    curves = get_roc_curve_by_class(*roc_list)
    display = RocCurveDisplay(**list(curves.values())[2])
    display.plot()
    plt.show()


def test_matrix(df):
    df = df.loc[idx[:, :, "single", :, :, "AUC", :], idx[:, :, :, :, :, :]]
    return create_matrix_from_df(df, 4, 1)  # Loss function comparison


if __name__ == "__main__":
    df = dill.load(open("merged.df", "rb"))
    df = df.dropna(0, "all")
    df = df.dropna(1, "all")
    df = df.round(2)
    df = df.drop(100, axis=1, level=5)
    model_names = vars.model_names
    model_names.remove("xlnet")
    pretrains = df.drop(model_names, axis=1, level=0)
    pretrains = pretrains.drop([3, 5], axis=1, level=1)
    pretrains = pretrains.drop(False, axis=1, level=2)
    pretrains = pretrains.drop(["oversample", "undersample"], axis=1, level=3)
    pretrains = pretrains.drop("gpt2", axis=1, level=6)
    df = df.drop("xlnet-base-cased", axis=1, level=6)
    df = pd.concat([df, pretrains], axis=1)
    # df = get_res_multi_file("../results_dracula")
    # df = df.loc[idx[:, :, "single", :, :, "AUC", :], idx[:, 1, False, None, ["DiceLoss()", "TverskyLoss()"], :, :, "Macro"]]
    df = df.loc[idx[:, :, "single", :, :, "AUC", :], idx[:, 1, False, None, ["DiceLoss()", "TverskyLoss()"], :, :, "Macro"]]
    # df = create_matrix_from_df(df, 4, 1).round(2)
    # index: dataset, cls_type, label_type, proofread, text_length, metrics, random_seed
    # column: model, num_layer, pretrained, balance_strategy, loss, qkv_size, tokenizer_pretrained, Macros/Micro
    # df = layer_value(df)
    # df = get_df_ci(df, col_levels=["model", "loss_func", ""], row_levels=[ "metrics"]).round(2)
    # df = df_anova1(df, axis=1, level=0)
    df = df_anova2_cr(df, col_levels=[4], row_levels=[0])
    # df = df_anova2(df, axis=1, levels=[0, 4])
    # df = hypo_test(df, axis=0, level=1).round(2).to_latex()
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