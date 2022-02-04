hpc_folder = "/scratch/itee/uqclyu1/"
current = hpc_folder
current = ""
parameter_folder = "%sparams" % current
results_folder = "%sresults" % current
cache_folder = "%s.cache" % current
trained_model_folder = "%strained" % current

hf_cache_folder = "%s.cache/huggingface" % current

model_names = ["bert", "xlnet", "gpt2",
               "lstm", "lstmattn", "cnn", "rcnn", "mlp", "han"]
transformer_names = model_names[:3]
transformer_pretrain = \
    ["bert-base-uncased", "xlnet-base-cased", "roberta-base", "gpt2"]
customized_model_names = model_names[4:]

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
    "huggingface_dataset_name": ["imdb"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.5,
},
# 12
{
    "huggingface_dataset_name": ["go_emotions"],
    "label_field": "labels",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    "multi_label": True,
    },
# 13
{
    "huggingface_dataset_name": ["tweet_eval", "emoji"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 14
{
    "huggingface_dataset_name": ["glue", "sst2"],
    "label_field": "label",
    "text_fields": ["sentence"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 15
{
    "huggingface_dataset_name": ["ag_news"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
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
    "huggingface_dataset_name": ["amazon_reviews_multi", "en"],
    "label_field": "product_category",
    "text_fields": ["review_title", "review_body"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.6,
    },
# 18
{
    "huggingface_dataset_name": ["dbpedia_14"],
    "label_field": "label",
    "text_fields": ["title", "content"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    },
# 19
{
    "huggingface_dataset_name": ["yelp_review_full"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.42,
    },
# 20
{
    "huggingface_dataset_name": ["yahoo_answers_topics"],
    "label_field": "topic",
    "text_fields": ["question_title", "question_content", "best_answer"],
    "cls_ratio_to_imb": 0.4,
    "sample_ratio_to_imb": 0.42,
    },
# 21
{
    "huggingface_dataset_name": ["amazon_polarity"],
    "label_field": "label",
    "text_fields": ["title", "content"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    },
# 22
{
    "huggingface_dataset_name": ["lex_glue", "ecthr_a"],
    "label_field": "labels",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    "multi_label": True,
    },
# 23
{
    "huggingface_dataset_name": ["lex_glue", "ecthr_b"],
    "label_field": "labels",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    "multi_label": True,
    },
# 24
{
    "huggingface_dataset_name": ["lex_glue", "scotus"],
    "label_field": "label",
    "text_fields": ["text"],
    "cls_ratio_to_imb": 0.5,
    "sample_ratio_to_imb": 0.53,
    },
]