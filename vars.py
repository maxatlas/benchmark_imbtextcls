hpc_folder = "/scratch/itee/uqclyu1/"
current = hpc_folder
current = ""
parameter_folder = "%sparams" % current
results_folder = "%sresults" % current
cache_folder = "%s.cache" % current
trained_model_folder = "%strained" % current
results_cur_folder = "%sresults/CUR" % current
trained_model_cur_folder = "%strained/CUR" % current

balanced_ds = ["md_gender_bias", "sst", "imdb", "glue_sst2",
               "ag_news", "amazon_reviews_multi", "dbpedia_14",
               "yelp_review_full", "yahoo_answers_topics",
               "amazon_polarity", "banking77", "amazon_reviews_multi_en"]
imbalanced_ds = ["poem_sentiment", "sms_spam", "lex_glue_scotus", "glue_cola",
                 "lex_glue_ecthr_a",
                 "lex_glue_ecthr_b", "hate_speech18", "emotion",
                 "ade_corpus_v2_Ade_corpus_v2_classification",
                 "hate_speech_offensive", "go_emotions", "tweet_eval_emoji"]
binary_ds = ["poem_sentiment", "sms_spam", "sst", "glue_cola",
             "ade_corpus_v2_Ade_corpus_v2_classification", "imdb",
             "glue_sst2", "amazon_polarity", ]
multilabel_ds = ["md_gender_bias", "go_emotions", "reuters21578_ModLewis"]
proofread_ds = ["poem_sentiment", "md_gender_bias", "sst", "glue_cola", "ade_corpus_v2_Ade_corpus_v2_classification", "glue_sst2", "ag_news", "dbpedia_14"]
twitter_ds = ["emotion", "hate_speech_offensive"]

hf_cache_folder = "%s.cache/huggingface" % current

model_names = ["lstm", "lstmattn", "cnn", "rcnn", "mlp", "han",
               "bert", "gpt2", "xlnet",]
transformer_names = model_names[-3:]
transformer_pretrain = \
    ["bert-base-uncased", "xlnet-base-cased", "gpt2"]
customized_model_names = model_names[3:]

customized_tokenizer_names = ["nltk", "spacy"]

cutoff = 400_000

split_strategies = ["undersample", "oversample"]


imb_ratio = {
    "threshold": 0.6,
    "tolerance": 0.3,
    "glue_sst2": (0.5, 0.6),
    "imdb": (0.5, 0.5),
    "sst": (0.5, 0.5),
    "amazon_polarity": (0.5, 0.53),
    "ag_news": (0.5, 0.53),
    "yelp_review_full": (0.5, 0.42),
    "dbpedia_14": (0.5, 0.53),
    "amazon_reviews_multi_en_stars": (0.5, 0.42),
    "banking77": (0.3, 0.6),
    "yahoo_answers_topics": (0.4, 0.42),
    "md_gender_bias": (0.8, 0.46)
}

split_ratio = {
    "train": 0.75,
    "test": 0.2,
    "validation": 0.05,
}

kvtypes = {
    "glove": "glove-wiki-gigaword-300",
    "word2vec": 'word2vec-google-news-300',
    "fasttext": "fasttext-wiki-news-subwords-300"
}

datasets_meta = [
# 1
{
    "huggingface_dataset_name": ["poem_sentiment"],
    "label_field": "label",
    "text_fields": ["verse_text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 2
{
    "huggingface_dataset_name": ["md_gender_bias"],
    "label_field": "labels",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.8,
    "sample_ratio_to_imb": 0.46,
    "multi_label": True,
    },
# 3
{
    "huggingface_dataset_name": ["sms_spam"],
    "label_field": "label",
    "text_fields": ["sms"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 4
{
    "huggingface_dataset_name": ["sst"],
    "label_field": "label",
    "text_fields": ["sentence"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.5,
    },
# 5
{
    "huggingface_dataset_name": ["glue", "cola"],
    "label_field": "label",
    "text_fields": ["sentence"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 6
{
    "huggingface_dataset_name": ["banking77"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.3,
    "sample_ratio_to_imb": 0.6,
    },
# 7
{
    "huggingface_dataset_name": ["hate_speech18"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 8
{
    "huggingface_dataset_name": ["emotion"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 9
{
    "huggingface_dataset_name": ["ade_corpus_v2", "Ade_corpus_v2_classification"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 10
{
    "huggingface_dataset_name": ["hate_speech_offensive"],
    "label_field": "class",
    "text_fields": ["tweet"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 11
{
    "huggingface_dataset_name": ["go_emotions"],
    "label_field": "labels",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    "multi_label": True,
    },
# 12

{
    "huggingface_dataset_name": ["ag_news"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    },
# 13

{
    "huggingface_dataset_name": ["imdb"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.5,
},
# 14
    {
        "huggingface_dataset_name": ["reuters21578", "ModLewis"],
        "label_field": "topics",
        "text_fields": ["text"],
        "cls_ratio_to_imb": 0.5,
        "sample_ratio_to_imb": 0.53,
        "multi_label": True
    },
{
    "huggingface_dataset_name": ["yahoo_answers_topics"],
    "label_field": "topic",
    "text_fields": ["question_title", "question_content", "best_answer"],
    "cls_ratio_to_imb": 0.4,
    "sample_ratio_to_imb": 0.42,
    },
# 15

{
    "huggingface_dataset_name": ["glue", "sst2"],
    "label_field": "label",
    "text_fields": ["sentence"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 16
{
    "huggingface_dataset_name": ["amazon_reviews_multi", "en"],
    "label_field": "stars",
    "text_fields": ["review_title", "review_body"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.42,
    },
# 17
{
    "huggingface_dataset_name": ["dbpedia_14"],
    "label_field": "label",
    "text_fields": ["title", "content"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    },
# 18
{
    "huggingface_dataset_name": ["yelp_review_full"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.42,
    },
# 19

{
    "huggingface_dataset_name": ["tweet_eval", "emoji"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 20
{
    "huggingface_dataset_name": ["amazon_polarity"],
    "label_field": "label",
    "text_fields": ["title", "content"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    },
# 21
{
    "huggingface_dataset_name": ["lex_glue", "ecthr_a"],
    "label_field": "labels",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    "multi_label": True,
    },
# 22
{
    "huggingface_dataset_name": ["lex_glue", "ecthr_b"],
    "label_field": "labels",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    "multi_label": True,
    },
# 23
{
    "huggingface_dataset_name": ["lex_glue", "scotus"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    },
]

datasets_meta = datasets_meta[: 13]
dataset_names = ["_".join(ds['huggingface_dataset_name']) for ds in datasets_meta]

text_lengths = {'poem_sentiment': {0: {0.25: 7.0, 0.5: 8.0, 0.75: 10.0}}, 'md_gender_bias': {0: {0.25: 11.0, 0.5: 14.0, 0.75: 17.0}}, 'sms_spam': {0: {0.25: 9.0, 0.5: 15.0, 0.75: 27.0}}, 'sst': {0: {0.25: 12.0, 0.5: 18.0, 0.75: 25.0}}, 'glue_cola': {0: {0.25: 6.0, 0.5: 8.0, 0.75: 11.0}}, 'banking77': {0: {0.25: 8.0, 0.5: 11.0, 0.75: 14.0}}, 'hate_speech18': {0: {0.25: 9.0, 0.5: 15.0, 0.75: 24.0}}, 'emotion': {0: {0.25: 11.0, 0.5: 17.0, 0.75: 25.0}}, 'ade_corpus_v2_Ade_corpus_v2_classification': {0: {0.25: 14.0, 0.5: 19.0, 0.75: 26.0}}, 'hate_speech_offensive': {0: {0.25: 11.0, 0.5: 18.0, 0.75: 27.0}}, 'go_emotions': {0: {0.25: 9.0, 0.5: 15.0, 0.75: 22.0}}, 'imdb': {0: {0.25: 151.0, 0.5: 210.0, 0.75: 343.0}}, 'ag_news': {0: {0.25: 36.0, 0.5: 43.0, 0.75: 50.0}}, 'reuters21578_ModLewis': {0: {0.25: 49.0, 0.5: 93.0, 0.75: 172.0}}}

header_meta = {
    "model": {
        "values": model_names,
        "func": (lambda x: x[0]['task']['model_config']['model_name'], []),
    },
    "num_layer": {
        "values": [1, 3, 5],
        "func": (lambda x: x[0]['task']['model_config']['num_layers'], []),
    },
    "pretrained": {
        "values": [True, False],
        "func": (lambda x: True if x[0]['task']['model_config']['pretrained_model_name'] else False, []),
    },
    "balance_strategy": {
        "values": ["oversample", "undersample", None],
        "func": (lambda x: x[0]['task']['data_config']['balance_strategy'], []),
    },
    "loss_func": {
        "values": ["CrossEntropyLoss()", "DiceLoss()", "TverskyLoss()", "FocalLoss()", "BCEWithLogitsLoss()"],
        "func": (lambda x: x[0]['task']['loss_func'], []),
    },
    "qkv_size": {
        "values": [100, 768],
        "func": (lambda x: 768 if not x[0]['task']['model_config'].get("qkv_size") else x[0]['task']['model_config'].get("qkv_size"), []),
    },
    "pretrained_tokenizer": {
        "values": transformer_pretrain,
        "func": (lambda x: x[0]['task']['model_config'].get("pretrained_tokenizer_name"), []),
    },
    "": {
        "values": ["Macro", "Micro"],
        "func": (lambda x: None, [])
    }
}


def label_text_length(inputs):
    res = inputs[0]
    ds_name = "_".join(res['task']['data_config']['huggingface_dataset_name'])
    length_dict = text_lengths[ds_name][0]
    if length_dict[0.75] <= 10:
        return "small"
    else:
        if length_dict[0.25] < 10:
            return "small"
        elif length_dict[0.25] > 10 and length_dict[0.75] <= 50:
            return "medium"
        elif length_dict[0.75] > 50:
            return "large"


index_meta = {
    "dataset": {
        "values": dataset_names,
        "func": (lambda x: "_".join(x[0]['task']['data_config']['huggingface_dataset_name']), []),
    },
    "cls_type": {
        "values": ["binary", "multiclass"],
        "func": (lambda x: "binary" if "_".join(x[0]['task']['data_config']['huggingface_dataset_name']) in binary_ds else "multiclass", []),
    },
    "label_type": {
        "values": ["single", "multilabel"],
        "func": (lambda x: "single" if "_".join(x[0]['task']['data_config']['huggingface_dataset_name']) not in multilabel_ds else "multilabel", []),
    },
    "proofread": {
        "values": [True, False],
        "func": (lambda x: "_".join(x[0]['task']['data_config']['huggingface_dataset_name']) in proofread_ds, []),
    },
    "text_length": {
        "values": ["small", "medium", "large"],
        "func": (label_text_length, []),
    },
    "metrics": {
        "values": ["AUC", "F1", "Epochs", "Seconds/e"],
        "func": (lambda x: None, [])
    },
    "random_seed": {
        "values": [129, 29, 444],
        "func": (lambda x: None, [])
    }

}