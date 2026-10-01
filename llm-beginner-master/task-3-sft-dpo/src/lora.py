"""手写 LoRA（不依赖 peft）。

原理：冻结原权重 W0，在旁边挂低秩分支 B*A：
    h = W0 x + (alpha/r) * B(A x)
- A: in x r，kaiming 初始化；B: r x out，零初始化（保证注入瞬间输出不变）；
- 只训练 A/B，可训练参数占比 < 5%（自检 lora_param_count）；
- merge_lora：把 W0 + (alpha/r)*B@A 合回原 Linear 的权重并卸载 LoRA 分支。

对外契约：
    inject_lora(model, target_modules, r, alpha) -> model
    merge_lora(model) -> model
    save_lora(model, path) / load_lora(model, path)
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """把 nn.Linear 包一层：原权重冻结 + 可训练的低秩旁路。"""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.base.weight.requires_grad_(False)
        if base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0

        out_f, in_f = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(in_f, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base(x)
        if not self.merged and self.r > 0:
            # LoRA: h = W x + (α/r)·(x @ A) @ B
            # A: (in, r), B: (r, out)
            delta = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B)
            h = h + delta * self.scaling
        return h

    def merge(self) -> nn.Linear:
        """把 LoRA 增量合入 base.weight，返回纯 Linear（卸载旁路）。"""
        if not self.merged:
            # 有效权重增量 Δ = A@B (in,out)；W 存储形状是 (out,in)
            delta = (self.lora_A @ self.lora_B).detach().t() * self.scaling
            new_w = (self.base.weight.data + delta).to(self.base.weight.dtype)
            self.base.weight.data = new_w
            self.merged = True
        return self.base

    def state_dict_lora(self) -> dict:
        return {
            "r": self.r, "alpha": self.alpha,
            "lora_A": self.lora_A.detach().cpu(),
            "lora_B": self.lora_B.detach().cpu(),
            "merged": self.merged,
        }


def _match(name: str, target_modules) -> bool:
    return any(name.endswith(m) or name == m for m in target_modules)


def inject_lora(model, target_modules, r: int = 8, alpha: float = 16) -> nn.Module:
    """冻结整个基座，只把匹配 target_modules 的 nn.Linear 替换为带 LoRA 旁路的版本。

    LoRA 的常规做法：全模型 requires_grad=False，仅 LoRA 的 A/B 可训练，
    这样可训练参数占比才能 < 5%。
    """
    model.requires_grad_(False)
    for name, module in list(model.named_modules()):
        if not _match(name, target_modules):
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        wrapped = LoRALinear(module, r, alpha).to(device=module.weight.device,
                                                   dtype=module.weight.dtype)
        if isinstance(parent, nn.ModuleList):
            parent[int(leaf)] = wrapped
        else:
            setattr(parent, leaf, wrapped)
    return model


def merge_lora(model) -> nn.Module:
    """把已注入的 LoRA 增量合并回原权重，并卸载 LoRA 旁路。"""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear) and not module.merged:
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            linear = module.merge()
            if isinstance(parent, nn.ModuleList):
                parent[int(leaf)] = linear
            else:
                setattr(parent, leaf, linear)
    return model


def _module_full_names(model) -> list[str]:
    return [name for name, _ in model.named_modules()]


def save_lora(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd = module.state_dict_lora()
            payload[name] = {
                "r": sd["r"], "alpha": sd["alpha"],
                "A": sd["lora_A"], "B": sd["lora_B"], "merged": sd["merged"],
            }
    torch.save(payload, path)


def load_lora(model, path: str | Path) -> nn.Module:
    """按模块名给 model 注入 LoRA 并装载权重（需要先 inject_lora 出同名模块）。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for name, sd in payload.items():
        module = model.get_submodule(name)
        if isinstance(module, LoRALinear):
            module.r = sd["r"]
            module.alpha = sd["alpha"]
            module.scaling = sd["alpha"] / max(1, sd["r"])
            with torch.no_grad():
                module.lora_A.copy_(sd["A"].to(device=module.lora_A.device,
                                               dtype=module.lora_A.dtype))
                module.lora_B.copy_(sd["B"].to(device=module.lora_B.device,
                                               dtype=module.lora_B.dtype))
            module.merged = bool(sd.get("merged", False))
    return model
