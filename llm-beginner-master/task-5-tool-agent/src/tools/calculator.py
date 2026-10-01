"""计算器工具：只允许算术/数学函数白名单（不用裸 eval）。"""
from __future__ import annotations

import ast
import math
import operator as op

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "四则运算与常用数学函数（sqrt/log/sin/cos/abs/pow 等），"
                       "返回计算结果字符串。",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string",
                                          "description": "数学表达式，如 2 + 3 * 4"}},
            "required": ["expression"],
        },
    },
}

_ALLOWED = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Mod: op.mod,
    ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos,
    ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le,
    ast.Gt: op.gt, ast.GtE: op.ge,
}

_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "floor": math.floor, "ceil": math.ceil,
    "round": round, "pow": pow, "min": min, "max": max, "factorial": math.factorial,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数值常量")
    if isinstance(node, ast.BinOp):
        return _ALLOWED[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in _FUNCS:
            raise ValueError(f"不支持的函数: {func_name}")
        args = [_eval_node(a) for a in node.args]
        return _FUNCS[func_name](*args)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left = _eval_node(node.left)
        right = _eval_node(node.comparators[0])
        return _ALLOWED[type(node.ops[0])](left, right)
    raise ValueError(f"不支持的表达式: {type(node).__name__}")


def run(args: dict) -> str:
    expr = str(args.get("expression", "")).strip()
    if not expr:
        raise ValueError("expression 不能为空")
    tree = ast.parse(expr, mode="eval")
    val = _eval_node(tree)
    # 归一化输出便于关键词校验（去千分位等）
    return format(val, ".10f").rstrip("0").rstrip(".") if isinstance(val, float) \
        else str(val)
