"""自建中文 SFT / DPO 数据（Qwen 对话格式，行分隔 JSON）。

- SFT：data/sft_samples.jsonl  每条 {"id","messages":[{role,content},...]}
- DPO：data/dpo_pairs.jsonl    每条 {"id","prompt","chosen","rejected"}

主题：深度学习与机器学习概念、Python 编程、经典唐诗/通识、简单逻辑与数学。
rejected 版本：短答 / 空泛 / 明显错误，用于偏好对齐。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _q(topic: str, question: str, answer: str):
    return {"topic": topic, "question": question, "answer": answer}


QA = [
    # 深度学习概念
    _q("dl", "什么是反向传播？",
       "反向传播是一种计算神经网络参数梯度的算法。它利用链式法则，从输出层的损失出发，"
       "逐层向前计算每个参数的偏导数，再结合优化器更新权重，从而最小化损失函数。"),
    _q("dl", "Transformer 中的自注意力是如何计算的？",
       "自注意力先对输入做 Q、K、V 三个线性投影，然后计算 Q 与 K 的点积并除以√d_k 缩放，"
       "经 softmax 得到权重后再对 V 加权求和。这样每个位置都能直接看到序列中其它位置的信息。"),
    _q("dl", "什么是过拟合？如何缓解？",
       "过拟合指模型在训练集上表现好、但在未见数据上表现差，通常因为模型容量过大或数据不足。"
       "缓解方法包括增大数据、数据增强、正则化（L2/ dropout）、早停、降低模型容量等。"),
    _q("dl", "解释一下梯度消失问题。",
       "梯度消失指深层网络中靠前层参数的梯度因反向传播的连乘而趋近于零，导致前层几乎无法更新。"
       "常见缓解方法有残差连接、ReLU 族激活、合理的初始化、LayerNorm 以及归一化的权重初始化。"),
    _q("dl", "LoRA 的原理是什么？",
       "LoRA（低秩适配）在冻结的预训练权重旁边加入低秩分解矩阵 A、B，使 h = Wx + (α/r)·BAx，"
       "只训练 A、B。这样既大幅减少可训练参数量，又能在下游任务上接近全量微调的效果。"),
    _q("dl", "什么是 dropout？",
       "dropout 在训练时按一定概率随机把神经元的输出置零，使模型不依赖个别神经元，起到正则化作用；"
       "推理时按保留概率缩放权重，保证输出期望一致。"),
    _q("dl", "embedding 和 one-hot 的区别？",
       "one-hot 是稀疏的定长向量，维度等于词表大小且不含语义；embedding 是稠密的低维可学习向量，"
       "相近语义的词在空间中距离更近，能捕捉词汇关系。"),
    _q("dl", "RLHF 与 DPO 有何不同？",
       "RLHF 需要先训练一个奖励模型，再用强化学习（如 PPO）优化策略；DPO 直接从偏好数据推导出"
       "隐式奖励，用简单的二分类式损失同时训练策略，省去奖励模型和强化学习的复杂度。"),
    # Python 编程
    _q("py", "Python 中列表和元组的区别？",
       "列表是可变的，可增删改元素；元组不可变，创建后不能修改。因此元组可做字典键、占用更小。"
       "语法上列表用 []、元组用 ()。"),
    _q("py", "怎么用 Python 读取一个文本文件？",
       "使用 with open('file.txt', 'r', encoding='utf-8') as f: content = f.read() 即可安全读取，"
       "with 语句会自动关闭文件。"),
    _q("py", "解释一下 Python 的装饰器。",
       "装饰器是一个接收函数并返回新函数的可调用对象，用 @decorator 语法糖包裹原函数，"
       "常用于在不改函数体的情况下增加日志、计时、鉴权等横切逻辑。"),
    _q("py", "生成器和迭代器有什么区别？",
       "迭代器实现了 __iter__ 与 __next__ 协议；生成器是用 yield 写成的迭代器，"
       "能惰性逐个产出值、节省内存，比如逐行读大文件时非常有用。"),
    # 通识 / 唐诗
    _q("cn", "“床前明月光”的作者是谁？",
       "李白。这是《静夜思》的前两句，全诗为：床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
    _q("cn", "简单介绍一下机器学习里的监督学习。",
       "监督学习使用带标签的样本 (x, y) 训练模型，学习输入到输出的映射，"
       "典型任务包括分类（判断类别）和回归（预测连续值）。"),
    _q("cn", "2 的 10 次方等于多少？",
       "2 的 10 次方等于 1024。"),
    _q("cn", "什么是梯度下降？",
       "梯度下降是沿损失函数负梯度方向迭代更新参数的优化方法：θ ← θ − η·∇L(θ)。"
       "学习率 η 控制每步大小，太大震荡、太小收敛慢。"),
    _q("cn", "CPU 和 GPU 的区别是什么？",
       "CPU 核少但单核强、擅长复杂串行控制；GPU 有大量简单计算核心，擅长大规模并行计算，"
       "因此深度学习的矩阵运算主要用 GPU 加速。"),
    _q("cn", "“举头望明月，低头思故乡”出自哪首诗？",
       "出自李白《静夜思》。这首诗借明月抒发游子思乡之情，语言朴素而意境深远。"),
]

# 补充：把每条答案打乱做 rejected 版本
REJECT_REASONS = {
    "short": lambda a: "我不太清楚。",
    "vague": lambda a: "这是一个很复杂的问题，涉及很多方面，需要深入分析才能回答。",
    "wrong": lambda a: a[::-1][:len(a)//3] if len(a) > 20 else "答案：不知道。",
    "off": lambda a: "这个问题我暂时无法回答，请换个问题试试。",
}


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    sft = DATA / "sft_samples.jsonl"
    dpo = DATA / "dpo_pairs.jsonl"
    with sft.open("w", encoding="utf-8") as fs, dpo.open("w", encoding="utf-8") as fd:
        for i, item in enumerate(QA):
            # SFT：单轮 或 双轮（追加追问）
            msg = [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]
            fs.write(json.dumps({"id": f"sft-{i}", "messages": msg},
                                ensure_ascii=False) + "\n")
            # 追问一轮（无上下文也成立）
            follow = f"再详细一点讲一下“{item['question']}”里的核心思想好吗？"
            fs.write(json.dumps(
                {"id": f"sft-{i}-f", "messages": [
                    {"role": "user", "content": follow},
                    {"role": "assistant",
                     "content": "好的。" + item["answer"]}]},
                ensure_ascii=False) + "\n")

            # DPO 对
            fd.write(json.dumps(
                {"id": f"dpo-{i}", "prompt": item["question"],
                 "chosen": item["answer"],
                 "rejected": REJECT_REASONS["short"](item["answer"])},
                ensure_ascii=False) + "\n")
            fd.write(json.dumps(
                {"id": f"dpo-{i}-b", "prompt": f"请用中文回答：{item['question']}",
                 "chosen": item["answer"],
                 "rejected": REJECT_REASONS["vague"](item["answer"])},
                ensure_ascii=False) + "\n")
            fd.write(json.dumps(
                {"id": f"dpo-{i}-c", "prompt": item["topic"] + "：" + item["question"],
                 "chosen": item["answer"],
                 "rejected": REJECT_REASONS["wrong"](item["answer"])},
                ensure_ascii=False) + "\n")
    print("SFT samples:", sum(1 for _ in sft.open(encoding="utf-8")))
    print("DPO samples:", sum(1 for _ in dpo.open(encoding="utf-8")))


if __name__ == "__main__":
    main()
