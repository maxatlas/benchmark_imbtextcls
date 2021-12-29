

training_config = {
    "cutoff": 400_000,
    "split_strategy": "uniform"
}

split_strategies = ["follow known", "uniform"]


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
    "train" : 0.75,
    "test" : 0.2,
    "validation": 0.05,
}

kvtypes = {
    "glove": "glove-wiki-gigaword-300",
    "word2vec": 'word2vec-google-news-300',
    "fasttext": "fasttext-wiki-news-subwords-300"
}

datasets_meta = [
{
    "dname": ["glue", "sst2"],
    "label_field": "label",
    "text_fields": ["sentence"]
    },

{
    "dname": ["sst"],
    "label_field": "label",
    "text_fields": ["sentence"]
    },

{
    "dname": ["glue", "cola"],
    "label_field": "label",
    "text_fields": ["sentence"]
    },

{
    "dname": ["ag_news"],
    "label_field": "label",
    "text_fields": ["text"]
    },

{
    "dname": ["emotion"],
    "label_field": "label",
    "text_fields": ["text"]
    },

{
    "dname": ["hate_speech18"],
    "label_field": "label",
    "text_fields": ["text"]
    },

{
    "dname": ["hate_speech_offensive"],
    "label_field": "class",
    "text_fields": ["tweet"]
    },

{
    "dname": ["tweet_eval", "emoji"],
    "label_field": "label",
    "text_fields": ["text"]
    },

{
    "dname": ["banking77"],
    "label_field": "label",
    "text_fields": ["text"]
    },

{
    "dname": ["sms_spam"],
    "label_field": "label",
    "text_fields": ["sms"]
    },

{
    "dname": ["ade_corpus_v2", "Ade_corpus_v2_classification"],
    "label_field": "label",
    "text_fields": ["text"]
    },

{
    "dname": ["poem_sentiment"],
    "label_field": "label",
    "text_fields": ["verse_text"]
    },

{
    "dname": ["go_emotions"],
    "label_field": "labels",
    "text_fields": ["text"]
    },

{
    "dname": ["md_gender_bias"],
    "label_field": "labels",
    "text_fields": ["text"]
    },

{
    "dname": ["lex_glue", "ecthr_a"],
    "label_field": "labels",
    "text_fields": ["text"]
    },

{
    "dname": ["lex_glue", "ecthr_b"],
    "label_field": "labels",
    "text_fields": ["text"]
    },

{
    "dname": ["lex_glue", "scotus"],
    "label_field": "label",
    "text_fields": ["text"]
    },

{
    "dname": ["dbpedia_14"],
    "label_field": "label",
    "text_fields": ["title", "content"]
    },

{
    "dname": ["amazon_reviews_multi", "en"],
    "label_field": "stars",
    "text_fields": ["review_title", "review_body"]
    },

{
    "dname": ["amazon_reviews_multi", "en"],
    "label_field": "product_category",
    "text_fields": ["review_title", "review_body"]
    },


{
    "dname": ["yahoo_answers_topics"],
    "label_field": "topic",
    "text_fields": ["question_title", "question_content", "best_answer"]
    },

{
    "dname": ["yelp_review_full"],
    "label_field": "label",
    "text_fields": ["text"]
    },

{
    "dname": ["imdb"],
    "label_field": "label",
    "text_fields": ["text"],

},

{
    "dname": ["amazon_polarity"],
    "label_field": "label",
    "text_fields": ["title", "content"]
    },
]