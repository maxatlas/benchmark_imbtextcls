from Config import ModelConfig

import build_model


def build(model_name, tokenizer_name, pretrained_model_name, pretrained_tokenizer_name):
    mc = ModelConfig(model_name, n_labels,
                     tokenizer_name=tokenizer_name,
                     pretrained_model_name=pretrained_model_name,
                     pretrained_tokenizer_name=pretrained_tokenizer_name,
                     word_max_length=50,
                     )
    return build_model.main(mc)


if __name__ == "__main__":
    n_labels = 6

    print("Scenario 1. Pretrained model.")
    model_name = "xlnet"
    pretrained_model_name = "xlnet-base-cased"
    # build(model_name, None, pretrained_model_name, None)

    print("Scenario 2. Transformer with pretrained tokenizer.")
    model_name = "roberta"
    pretrained_tokenizer_name = "roberta-base"
    # build(model_name, None, None, pretrained_tokenizer_name)

    print("Scenario 3. Transformer with customized tokenizer.")
    model_name = "xlnet"
    tokenizer_name = "xlnet"
    try:
        build(model_name, tokenizer_name, None, None)
    except Exception as e:
        print(e)

    print("Scenario 4. Model with pretrained tokenizer.")
    model_name = "lstmattn"
    pretrained_tokenizer_name = "xlnet-base-cased"
    build(model_name, None, None, pretrained_tokenizer_name)

    print("Scenario 5. Model with customized transformer tokenizer.")
    model_name = "lstmattn"
    tokenizer_name = "roberta"
    try:
        build(model_name, tokenizer_name, None, None)
    except Exception as e:
        print(e)

    print("Scenario 6. Model with customized tokenizer.")
    model_name = "cnn"
    tokenizer_name = "nltk"

    mc = ModelConfig(model_name, n_labels,
                     tokenizer_name=tokenizer_name,
                     word_max_length=50,
                     emb_path="parameters/emb_layer_glove",
                     word_index_path="parameters/word_index"
                     )
    model = build_model.main(mc)


