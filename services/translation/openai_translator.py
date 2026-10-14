"""OpenAI 兼容 Chat Completions 翻译适配器。"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from services.translation.base import TranslationError, Translator, register_translator
from services.translation.http_errors import raise_for_status
from utils.language_utils import to_openai_lang

DEFAULT_GLOSSARY = """\
health = 生命值
mana = 法力
attack = 攻擊力
defense = 防禦力
damage = 傷害
experience = 經驗
level = 等級
quest = 任務
inventory = 背包
settings = 設定
save = 儲存
load = 讀取
"""


def _parse_json_array(content: str) -> list[str] | None:
    content = content.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # 模型偶尔会包一层 markdown 代码块或前后有说明文字
    match = re.search(r"\[[\s\S]*\]", content)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return None


class OpenAITranslator(Translator):
    name = "openai"

    def __init__(self, config_section: dict, cache=None, api_key: str = "") -> None:
        super().__init__(config_section, cache)
        openai_cfg = config_section.get("openai", {}) if config_section else {}
        self.api_key = api_key or str(openai_cfg.get("api_key", ""))
        self.base_url = str(openai_cfg.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        self.model = str(openai_cfg.get("model", "gpt-4o-mini"))
        custom_glossary = str(config_section.get("glossary", "")) if config_section else ""
        self.glossary = DEFAULT_GLOSSARY + ("\n" + custom_glossary if custom_glossary else "")

    def model_label(self) -> str:
        return f"{self.name}/{self.model}"

    def _translate_batch(self, texts: list[str], source_language: str | None, target_language: str) -> list[str]:
        if not self.api_key:
            raise TranslationError(
                "缺少 OpenAI API Key：請設定 OPENAI_API_KEY 環境變數或在設定中填寫"
            )

        target_desc = to_openai_lang(target_language)
        source_desc = to_openai_lang(source_language) if source_language and source_language != "auto" else "自動檢測"
        numbered = "\n".join(f"{i + 1}\t{text}" for i, text in enumerate(texts))

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是資深遊戲本地化與字幕翻譯專家。把使用者按行號給出的文本翻譯成 "
                        f"{target_desc}（源語言：{source_desc}）。\n"
                        "要求：\n"
                        "1. 譯文口語自然、貼近母語表達，不要逐字直譯，不要翻譯腔；\n"
                        "2. 長度儘量與原文接近，適合覆蓋顯示；\n"
                        "3. 專有名詞、人名、地名、產品名保持原文；\n"
                        "4. 術語表（原文 = 譯文）必須嚴格遵守：\n"
                        f"{self.glossary}\n"
                        "5. 只輸出 JSON 陣列，元素順序和數量與輸入行完全一致，每元素一行譯文；\n"
                        "6. 保留數字、網址、郵箱、佔位符、變數名和特殊符號；\n"
                        "7. 不要輸出任何解釋、引號包裹或程式碼塊標記。"
                    ),
                },
                {"role": "user", "content": numbered},
            ],
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=float(self.config.get("timeout_seconds", 30)),
            )
        except requests.RequestException as exc:
            raise TranslationError("OpenAI 請求失敗（超時或網路錯誤）") from exc

        response.encoding = "utf-8"
        if response.status_code != 200:
            raise_for_status(response, "OpenAI")

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise TranslationError("OpenAI 響應格式異常") from exc

        parsed = _parse_json_array(content)
        if parsed is None:
            raise TranslationError("OpenAI 返回內容不是可解析的 JSON 陣列")
        if len(parsed) != len(texts):
            # 数量对不上说明模型漏行/错位，抛错让上层逐条重试，避免译文错位
            raise TranslationError(f"OpenAI 返回 {len(parsed)} 條，期望 {len(texts)} 條，已逐條重試")
        return parsed


register_translator(OpenAITranslator)
