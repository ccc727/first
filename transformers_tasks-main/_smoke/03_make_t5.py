# -*- coding: utf-8 -*-
"""Build a tiny random T5 model dir (tokenizer = local char-level BERT tokenizer)."""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERT_DIR = os.path.join(ROOT, "_smoke", "models", "tiny_bert_char")
T5_DIR = os.path.join(ROOT, "_smoke", "models", "tiny_t5")
os.makedirs(T5_DIR, exist_ok=True)


def main():
    from transformers import T5Config, T5ForConditionalGeneration
    import json
    cfg = json.load(open(os.path.join(BERT_DIR, "config.json"), "r", encoding="utf-8"))
    vocab_size = cfg["vocab_size"]
    t5 = T5Config(
        vocab_size=vocab_size,
        d_model=128,
        d_kv=64,
        d_ff=512,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="relu",
        pad_token_id=0,
        eos_token_id=3,
        decoder_start_token_id=2,
    )
    model = T5ForConditionalGeneration(t5)
    model.save_pretrained(T5_DIR)
    for fn in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(BERT_DIR, fn), os.path.join(T5_DIR, fn))
    print("tiny t5 saved at", T5_DIR)


if __name__ == "__main__":
    main()
