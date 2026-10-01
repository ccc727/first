"""faiss 索引的跨平台读写。

faiss 底层 C++ 用 fopen 打开文件，路径会按本地代码页解释；当项目目录含非 ASCII
字符（如中文的 E:\\桌面\\...）时，直接 faiss.write_index / read_index 会抛
"could not open ... for writing/reading"。这里改走内存字节（serialize / deserialize），
由 Python 负责读写文件，天然支持 unicode 路径。

注意：serialize_index 产出的字节流与 write_index 落盘格式一致，故历史索引也能读。
"""
from __future__ import annotations

from pathlib import Path

import faiss


def write_index(index, path: str | Path) -> None:
    """把 faiss 索引写入 path（支持含中文的路径）。"""
    Path(path).write_bytes(faiss.serialize_index(index))


def read_index(path: str | Path):
    """从 path 读入 faiss 索引（支持含中文的路径）。

    deserialize_index 在本机 faiss 版本下要求 numpy 数组而非 bytes，故用
    np.frombuffer 转换。
    """
    import numpy as np

    data = Path(path).read_bytes()
    return faiss.deserialize_index(np.frombuffer(data, dtype="uint8"))
