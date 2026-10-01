# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

Copyright (c) 2022 ByteDance.com, Inc. All Rights Reserved
测试训练好的打分模型。

Author: pankeyu(pankeyu@bytedance.com)
Date: 2022/12/30
"""
import os
import torch
from rich import print
from transformers import AutoTokenizer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'    # CPU环境自适应
model_dir = os.environ.get('RM_MODEL_PATH', './checkpoints/reward_model/sentiment_analysis/model_best/')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = torch.load(os.path.join(model_dir, 'model.pt'), map_location='cpu', weights_only=False)
model.to(device).eval()

texts = [
    '买过很多箱这个苹果了，一如既往的好，汁多味甜～',
    '一台充电很慢，信号不好！退了！又买一台竟然是次品。。服了。。'
]
inputs = tokenizer(
    texts, 
    max_length=128,
    padding='max_length', 
    return_tensors='pt'
)
r = model(**inputs)
print(r)