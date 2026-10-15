"""模型資料夾名稱的慣例：顯示用的短名稱、快取鍵用的標記、各個系列的提示詞差異。

資料夾名稱（也是 ``translation.offline.model_name`` 的值）：

- ``qwen2.5-3b-instruct-gguf-q4_k_m``：Qwen2.5-Instruct 系列
- ``qwen3-1.7b-gguf-q8_0``：Qwen3 系列（混合思考模型，翻譯用非思考模式）
"""

from __future__ import annotations

import re

# Qwen3 是「思考／非思考」混合模型：回答之前會先寫一段 <think>…</think> 推理，翻譯字幕不需要，而且很慢。
# 官方的做法是在輪到模型回答的位置先放一個空的思考區塊（聊天範本的 enable_thinking=False），模型就直接寫答案。
QWEN3_NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_STRAY_THINK_RE = re.compile(r"</?think>")


def model_family(name: str) -> str:
    return "qwen3" if name.lower().startswith("qwen3") else "qwen2.5"


def answer_prefix(name: str) -> str:
    """接在最後一個 ``<|im_start|>assistant\\n`` 後面、由我們先寫好的字（Qwen3：空的思考區塊；其他系列：沒有）。"""
    return QWEN3_NON_THINKING_PREFIX if model_family(name) == "qwen3" else ""


def prompt_style(name: str) -> str:
    """提示詞的風格：Qwen3 用英文（中文版對它的台詞反而變差），其餘（Qwen2.5 系列與自訂模型）用中文。"""
    return "en" if model_family(name) == "qwen3" else "zh"


def strip_thinking(text: str) -> str:
    """去掉模型自己寫出來的思考區塊（沒有照聊天範本走、或思考沒關掉時）；沒有思考區塊就原樣回傳。"""
    return _STRAY_THINK_RE.sub("", _THINK_BLOCK_RE.sub("", text or ""))


def model_tag(name: str) -> str:
    """快取鍵用的短名稱：qwen2.5-1.5b-instruct-gguf-q4_k_m → 1.5b-q4_k_m；qwen3-1.7b-gguf-q8_0 → qwen3-1.7b-q8_0。"""
    return name.removeprefix("qwen2.5-").replace("-instruct-gguf-", "-").replace("-gguf-", "-")


def display_name(name: str) -> str:
    """表格與設定畫面用的短名稱：3B Q4_K_M、Qwen3 1.7B Q8_0（Qwen2.5 是預設系列，名稱不加系列）。"""
    if not name.lower().startswith("qwen"):
        return name  # 自訂的名稱照原樣顯示
    tag = model_tag(name)
    if model_family(name) == "qwen3":
        return "Qwen3 " + tag.removeprefix("qwen3-").replace("-", " ").upper()
    return tag.replace("-", " ").upper()
