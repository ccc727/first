# -*- coding: utf-8 -*-
"""Build local char-level tokenizer + tiny random BERT/GPT2/senti-2cls models."""
import os
import json
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "_smoke", "models")
os.makedirs(MODELS, exist_ok=True)

TEXT_FILES = [
    "text_matching/supervised/data/comment_classify/train.txt",
    "text_matching/supervised/data/comment_classify/dev.txt",
    "text_matching/supervised/data/comment_classify/types_desc.txt",
    "text_classification/data/comment_classify/train.txt",
    "text_classification/data/comment_classify/dev.txt",
    "text_classification/data/comment_classify/label_mapping.txt",
    "prompt_tasks/PET/data/comment_classify/train.txt",
    "prompt_tasks/PET/data/comment_classify/dev.txt",
    "prompt_tasks/PET/data/comment_classify/prompt.txt",
    "prompt_tasks/PET/data/comment_classify/verbalizer.txt",
    "prompt_tasks/p-tuning/data/comment_classify/train.txt",
    "prompt_tasks/p-tuning/data/comment_classify/dev.txt",
    "prompt_tasks/p-tuning/data/comment_classify/verbalizer.txt",
    "UIE/data/DuIE/train.txt",
    "UIE/data/DuIE/dev.txt",
    "RLHF/data/reward_datasets/sentiment_analysis/train.tsv",
    "RLHF/data/reward_datasets/sentiment_analysis/dev.tsv",
]
EXTRA = ["刚收到货，感觉", "这部电影很", "说实话，真的很", "这次购物总的来说体验很",
         "这是一个测试句子", "这个破笔记本卡的不要不要的，差评",
         "地理环境不错，但对面一直在盖楼，门前街道上打车不方便。",
         "跟好朋友一起凑单买的，很划算，洗发露是樱花香的，挺好的。。。"]


def scan_chars():
    chars = set()
    for rel in TEXT_FILES:
        fp = os.path.join(ROOT, rel)
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                chars.update(c for c in line if c not in "\t\r\n")
    for t in EXTRA:
        chars.update(t)
    return chars


def build_vocab():
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + \
               ["[unused%d]" % i for i in range(1, 51)]
    chars = sorted(scan_chars())
    vocab = specials + [c for c in chars if c not in specials]
    return vocab, {w: i for i, w in enumerate(vocab)}


def make_tokenizer_files(vocab, vocab_index, model_dir):
    from tokenizers import Tokenizer, AddedToken
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
    from tokenizers.normalizers import BertNormalizer

    tok = Tokenizer(WordLevel(vocab=vocab_index, unk_token="[UNK]"))
    tok.normalizer = BertNormalizer(clean_text=True, handle_chinese_chars=True,
                                    strip_accents=False, lowercase=False)
    tok.pre_tokenizer = BertPreTokenizer()
    from tokenizers import decoders
    tok.decoder = decoders.WordPiece(cleanup=True)
    special_ids = {k: vocab_index[k] for k in ("[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]")}
    for name, _id in special_ids.items():
        tok.add_special_tokens([AddedToken(name, normalized=False)])
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", special_ids["[CLS]"]), ("[SEP]", special_ids["[SEP]"])],
    )
    tok.enable_padding(pad_id=vocab_index["[PAD]"], pad_token="[PAD]")
    tok.save(os.path.join(model_dir, "tokenizer.json"))

    with open(os.path.join(model_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")

    cfg = {"tokenizer_class": "BertTokenizerFast", "do_lower_case": False,
           "unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]",
           "cls_token": "[CLS]", "mask_token": "[MASK]"}
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    smap = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]",
            "cls_token": "[CLS]", "mask_token": "[MASK]"}
    with open(os.path.join(model_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(smap, f, ensure_ascii=False, indent=2)


def make_bert_model(model_dir, vocab_size, num_labels=None):
    from transformers import BertConfig
    config = BertConfig(
        vocab_size=vocab_size, hidden_size=768, num_hidden_layers=2,
        num_attention_heads=12, intermediate_size=3072, hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, max_position_embeddings=512,
        type_vocab_size=2, pad_token_id=0,
        **({"num_labels": num_labels} if num_labels else {}))
    return config


def make_gpt2_model(model_dir, vocab_size):
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(vocab_size=vocab_size, n_positions=256, n_ctx=256,
                        n_embd=128, n_layer=2, n_head=4, resid_pdrop=0.0,
                        embd_pdrop=0.0, attn_pdrop=0.0, pad_token_id=0,
                        eos_token_id=0, bos_token_id=2)
    GPT2LMHeadModel(config).save_pretrained(model_dir)
    return config


def main():
    vocab, vocab_index = build_vocab()
    print("vocab size:", len(vocab))

    bert_dir = os.path.join(MODELS, "tiny_bert_char")
    gpt2_dir = os.path.join(MODELS, "tiny_gpt2_char")
    senti_dir = os.path.join(MODELS, "senti_2cls")
    for d in (bert_dir, gpt2_dir, senti_dir):
        os.makedirs(d, exist_ok=True)

    from transformers import BertModel, BertForSequenceClassification
    make_tokenizer_files(vocab, vocab_index, bert_dir)
    BertModel(make_bert_model(bert_dir, len(vocab))).save_pretrained(bert_dir)
    BertForSequenceClassification(make_bert_model(senti_dir, len(vocab), num_labels=2)).save_pretrained(senti_dir)
    make_gpt2_model(gpt2_dir, len(vocab))
    for extra in (gpt2_dir, senti_dir):
        for fn in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            shutil.copy(os.path.join(bert_dir, fn), os.path.join(extra, fn))
    print("models ready")


if __name__ == "__main__":
    main()

