"""提示詞調整實驗用的各種版本（``scripts/tune_prompts.py`` 用）。

一個版本 ＝ 一個 ``Spec``：系統提示詞、示範、每句使用者輪的格式。``install`` 把它裝進 ``offline_translator`` 模組
（換掉 ``static_prompt`` / ``history_prompt`` / ``user_prompt``），不必改正式程式碼就能比較。
``v4`` 是調整前（``PROMPT_VERSION = 4``）的原文，**逐字凍結**在這裡，之後正式程式碼改了也能重現基準。
``production`` 不裝任何東西，量的是目前正式程式碼。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

SKIP = "[SKIP]"
NAMES = {"ja": "Japanese", "en": "English"}


@dataclass(frozen=True)
class Spec:
    system: str  # 一般字幕的系統提示詞（不含略過擬聲詞的規則）
    system_lyrics: str  # 歌詞模式的系統提示詞
    skip_rule: str  # 開著「略過擬聲詞」時接在系統提示詞後面
    shots: dict[str, list[tuple[str, str]]]
    shots_lyrics: dict[str, list[tuple[str, str]]]
    user_fmt: str = "Translate this {name} text:\n{text}"  # 每句使用者輪（示範、前文、待翻譯句都用同一個格式）
    names: dict[str, str] = field(default_factory=lambda: dict(NAMES))  # 使用者輪格式裡 {name} 填的語言名稱
    notes: dict[str, str] = field(default_factory=dict)  # 這個版本改了什麼（報表用）

    def static_prompt(self, language: str, lyrics: bool = False, skip_sounds: bool = True, style: str = "") -> str:
        system = self.system_lyrics if lyrics else self.system
        if skip_sounds:
            system += self.skip_rule
        parts = [f"<|im_start|>system\n{system}<|im_end|>\n"]
        for source, target in (self.shots_lyrics if lyrics else self.shots)[language]:
            if target == SKIP and not skip_sounds:
                continue
            parts.append(f"<|im_start|>user\n{self._user(source, language)}<|im_end|>\n")
            parts.append(f"<|im_start|>assistant\n{target}<|im_end|>\n")
        return "".join(parts)

    def _user(self, text: str, language: str) -> str:
        return self.user_fmt.format(name=self.names[language], text=text)

    def history_prompt(self, entries, style: str = "") -> str:
        parts = []
        for entry in entries:
            parts.append(f"<|im_start|>user\n{self._user(entry.source, entry.language)}<|im_end|>\n")
            parts.append(f"<|im_start|>assistant\n{entry.translation}<|im_end|>\n")
        return "".join(parts)

    def user_prompt(self, text: str, language: str, answer_prefix: str = "", style: str = "") -> str:
        return f"<|im_start|>user\n{self._user(text, language)}<|im_end|>\n<|im_start|>assistant\n{answer_prefix}"


def install(module, spec: Spec) -> None:
    """把 ``spec`` 裝進 ``offline_translator`` 模組（翻譯器每次翻譯都現場讀這三個函式）。
    正式程式碼依模型系列選的提示詞風格（``style``）會被忽略：實驗版本對所有模型都用同一組。"""
    module.static_prompt = spec.static_prompt
    module.history_prompt = spec.history_prompt
    module.user_prompt = spec.user_prompt


# ----------------------------------------------------------------------------- v4：調整前（逐字凍結）
_V4_NO_SUBJECT = (
    " Do not add a subject or pronoun (such as 我, 你, 他, 我們) that the original text does not have: "
    "if the original leaves out the subject, leave it out of the translation too."
)
_V4_SKIP = (
    " If the text is only a sound effect, onomatopoeia, laughter, sigh, scream or other meaningless "
    "vocalization (for example ドン, ガシャーン, ワンワン, ははは, Boom, Woof, Haha), do not turn it into Chinese "
    "sound words such as 砰 or 汪汪; output exactly [SKIP] instead."
)
V4 = Spec(
    system=(
        "You are a professional translator. Translate the user's text into Traditional Chinese "
        "as used in Taiwan. Output only the translation, with no explanations." + _V4_NO_SUBJECT
    ),
    system_lyrics=(
        "You are a professional translator of song lyrics. The user's text is one line from a song. "
        "Translate it into natural, poetic Traditional Chinese as used in Taiwan, keeping the meaning, "
        "emotion and imagery instead of translating word by word. "
        "Write in modern vernacular Chinese (白話文), the way people speak and write today; "
        "do not use classical or literary Chinese (文言文) and avoid archaic words such as 之, 乃, 吾, 汝, 爾, 兮, 矣, 焉. "
        "The line may be a fragment. "
        "Output only the translation, with no explanations." + _V4_NO_SUBJECT
    ),
    skip_rule=_V4_SKIP,
    shots={
        "ja": [
            ("今日はいい天気ですね。", "今天天氣真好呢。"),
            ("ファイルを保存しますか？", "要儲存檔案嗎？"),
            ("HPが50回復した！", "恢復了 50 點 HP！"),
            ("疲れたから早く帰りたい", "累了，想早點回家"),
            ("私は学生です。", "我是學生。"),
            ("ドーン！ガシャーン！", SKIP),
            ("おかえり", "歡迎回來"),
            ("ワンワン", SKIP),
            ("好きだよ", "喜歡喔"),
            ("ふふっ", SKIP),
        ],
        "en": [
            ("Press any key to continue", "按任意鍵繼續"),
            ("You found a rusty key.", "你找到了一把生鏽的鑰匙。"),
            ("Want some coffee?", "要喝杯咖啡嗎？"),
            ("I am a student.", "我是學生。"),
            ("Save changes before closing?", "關閉前要儲存變更嗎？"),
            ("Boom! Crash!", SKIP),
            ("See you soon", "待會見"),
            ("Woof woof", SKIP),
            ("Haha", SKIP),
        ],
    },
    shots_lyrics={
        "ja": [
            ("君の声が聞こえる", "聽得見你的聲音"),
            ("星空の下で踊ろう", "在星空下共舞吧"),
            ("僕は君を忘れない", "我不會忘記你"),
            ("明日になれば忘れてしまうの？", "到了明天，就會忘記了嗎？"),
            ("ラララ…", SKIP),
            ("ふふふっ", SKIP),
        ],
        "en": [
            ("I will always love you", "我會永遠愛著你"),
            ("Hold me close and never let go", "緊緊抱住我，永遠不要放手"),
            ("We are dancing in the moonlight", "我們在月光下翩翩起舞"),
            ("Dancing in the moonlight", "在月光下起舞"),
            ("Na na na na", SKIP),
        ],
    },
    notes={"v4": "調整前（PROMPT_VERSION 4）"},
)

# ----------------------------------------------------------------------------- s1：只改系統提示詞
# 基準的問題（dev 批並排閱讀）：中文裡用日文頓號「、」；「不要加主詞」害問句缺「你」不通順；英文單字留在譯文裡；
# 歌詞把否定譯反（傘もささずに → 撐著傘）。s1 只重寫兩個系統提示詞，示範與其他不動。
S1 = replace(
    V4,
    system=(
        "You are a professional subtitle translator. Translate the user's text into natural, fluent Traditional Chinese "
        "as spoken in Taiwan. Keep the full meaning of the original: do not leave anything out and do not add anything. "
        "Translate every word into Chinese; keep only names, brand names and abbreviations as they are. "
        "Keep the tone of the speaker (polite stays polite, casual stays casual). "
        "Use Chinese punctuation such as \uff0c\u3002\uff1f\uff01 and never the Japanese comma \u3001. "
        "Chinese often needs a subject where Japanese or English drops it: when the speaker is clearly talking to or about "
        "someone (for example a question or a request to the listener), add \u4f60, \u6211 or \u4ed6 so that the sentence reads naturally, "
        "but never guess a subject that the text does not make clear. "
        "Output only the translation, with no explanations."
    ),
    system_lyrics=(
        "You are a professional translator of song lyrics. The user's text is one line from a song and may be a fragment. "
        "Translate it into natural, poetic Traditional Chinese as used in Taiwan, capturing the meaning, emotion and imagery "
        "instead of translating word by word, in plain modern spoken wording (\u767d\u8a71\u6587; no classical Chinese such as "
        "\u4e4b, \u4e43, \u543e, \u6c5d, \u516e, \u77e3, \u7109). "
        "Stay faithful: keep every image and idea of the line, keep negations and \"without\" (\u2026\u305a\u306b, \u2026\u306a\u3044\u3067, never, don't) exactly, "
        "and do not invent anything that is not in the line. Keep it short, like a sung line. "
        "Use Chinese punctuation such as \uff0c\u3002\uff1f\uff01 and never the Japanese comma \u3001. "
        "Lyrics often leave out the subject: leave it out too unless the line would be unclear without it. "
        "Output only the translation, with no explanations."
    ),
    notes={"s1": "只重寫兩個系統提示詞：完整保留語意、不加不漏、英文單字也要譯、保持語氣、中文標點（不用頓號）、需要時補主詞（問句、請求）；歌詞版加「忠於原句、保留否定、不要編造」"},
)

# ----------------------------------------------------------------------------- s2：維持基準的系統提示詞，擴充示範
# s1 證明只加規則幾乎沒用（chrF 持平，還多出幻覺與英文殘留）；小模型看示範比看規則有效。s2 改示範：
# 涵蓋問句補「你」、敬語、否定（…ずに、方言的ん）、語氣詞、人名與敬稱、單位、英文的它們／長官／慣用語、更多種擬聲詞；
# 標點對齊原文（原文有句號才有句號、日文頓號改逗號）。系統提示詞只加一句標點說明。示範的句子都不在題庫裡。
_S2_PUNCT = " Use Chinese punctuation such as \uff0c and never the Japanese comma \u3001."
S2_SHOTS = {
    "ja": [
        ("今日はいい天気ですね。", "今天天氣真好呢。"),
        ("ファイルを保存しますか？", "要儲存檔案嗎？"),
        ("まだ起きてたの？", "你還沒睡嗎？"),
        ("HPが50回復した！", "恢復了 50 點 HP！"),
        ("疲れたから早く帰りたい", "累了，想早點回家"),
        ("ドーン！ガシャーン！", SKIP),
        ("少し待ってください", "請稍等一下"),
        ("何も言わずに行ってしまった", "什麼都沒說就走了"),
        ("おかえり", "歡迎回來"),
        ("田中さんはもう来ていますか", "田中先生已經到了嗎"),
        ("ワンワン", SKIP),
        ("誰にも渡さん", "絕不交給任何人"),
        ("今日は楽しかったね", "今天好開心喔"),
        ("ちょっと待って、まだ準備できてない", "等一下，我還沒準備好"),
        ("好きだよ", "喜歡喔"),
        ("ゴゴゴゴ…", SKIP),
        ("ありがとうございます。助かりました", "謝謝您，幫了大忙"),
        ("それ、本当にやるの？", "那個，你真的要做嗎？"),
        ("残り時間は3分です", "剩餘時間 3 分鐘"),
        ("ふふっ", SKIP),
        ("私は学生です。", "我是學生。"),
    ],
    "en": [
        ("Press any key to continue", "按任意鍵繼續"),
        ("Where did you put it?", "你把它放在哪裡了？"),
        ("You found a rusty key.", "你找到了一把生鏽的鑰匙。"),
        ("Yes, sir. Right away.", "是，長官。馬上去。"),
        ("Boom! Crash!", SKIP),
        ("Your tickets are ready. They're at the front desk.", "你的票準備好了，放在櫃檯。"),
        ("Probably not, but I'll try.", "大概不行，但我會試試看。"),
        ("Want some coffee?", "要喝杯咖啡嗎？"),
        ("Ugh...", SKIP),
        ("Don't worry about it.", "別擔心。"),
        ("That was close!", "好險！"),
        ("Save changes before closing?", "關閉前要儲存變更嗎？"),
        ("Woof woof", SKIP),
        ("See you soon", "待會見"),
        ("I am a student.", "我是學生。"),
        ("Haha", SKIP),
    ],
}
S2_SHOTS_LYRICS = {
    "ja": [
        ("君の声が聞こえる", "聽得見你的聲音"),
        ("星空の下で踊ろう", "在星空下共舞吧"),
        ("夢の続きを探してる", "尋找夢的後續"),
        ("僕は君を忘れない", "我不會忘記你"),
        ("ラララ…", SKIP),
        ("泣かないで 笑ってよ", "別哭了 笑一個吧"),
        ("名前も知らない花のように", "像一朵不知名的花"),
        ("振り向かずに歩き出した", "頭也不回地邁步走去"),
        ("明日になれば忘れてしまうの？", "到了明天，就會忘記了嗎？"),
        ("ウォーウォー", SKIP),
        ("ねえ、聞こえてる？", "欸，聽得見嗎？"),
        ("ふふふっ", SKIP),
    ],
    "en": [
        ("I will always love you", "我會永遠愛著你"),
        ("Hold me close and never let go", "緊緊抱住我，永遠不要放手"),
        ("Under the neon lights tonight", "今夜在霓虹燈下"),
        ("We were wild and free", "我們狂放又自由"),
        ("Na na na na", SKIP),
        ("Don't let me go", "別放開我"),
        ("Dancing in the moonlight", "在月光下起舞"),
        ("Oh yeah", SKIP),
    ],
}
S2 = replace(
    V4,
    system=V4.system + _S2_PUNCT,
    system_lyrics=V4.system_lyrics + _S2_PUNCT,
    shots=S2_SHOTS,
    shots_lyrics=S2_SHOTS_LYRICS,
    notes={"s2": "示範擴充（日文一般 10→21、英文 9→16、日文歌詞 6→12、英文歌詞 5→8）；標點對齊原文；系統提示詞只加一句「用中文標點、不用日文頓號」"},
)

# ----------------------------------------------------------------------------- s3～s5：在 s2 上一次只改一件事
# s2 的 chrF 與基準沒有差別（雜訊範圍）。讀真實歌詞（紀錄檔）的基準譯文後，剩下的提示詞層級問題：
# 「不要加主詞」讓台詞少「你」、歌詞被「寫得有詩意」而加油添醋、遇到亂碼般的句子 Qwen3 會道歉並要求更多上下文。
_S3_SYSTEM = (
    "You are a professional translator. Translate the user's text into Traditional Chinese "
    "as used in Taiwan. Output only the translation, with no explanations." + _S2_PUNCT
)
S3 = replace(
    S2,
    system=_S3_SYSTEM,
    notes={"s3": "s2 拿掉一般提示詞裡的「原文沒有主詞，譯文也不要加」（歌詞提示詞不動）"},
)
_S4_LYRICS = (
    "You are a professional translator of song lyrics. The user's text is one line from a song and may be a fragment. "
    "Translate it faithfully into natural Traditional Chinese as used in Taiwan, in plain modern spoken wording "
    "(\u767d\u8a71\u6587; no classical Chinese such as \u4e4b, \u4e43, \u543e, \u6c5d, \u516e, \u77e3, \u7109). "
    "Keep the meaning, emotion and imagery of the line, but do not embellish: do not add images, words or subjects that are not in the line, "
    "and keep negations (\u2026\u305a\u306b, \u2026\u306a\u3044\u3067, don't, never) exactly. Keep it short, like a sung line. "
    "Output only the translation, with no explanations." + V4.system_lyrics[V4.system_lyrics.index(" Do not add a subject"):] + _S2_PUNCT
)
S4 = replace(
    S3,
    system_lyrics=_S4_LYRICS,
    notes={"s4": "s3 的歌詞提示詞改成「忠實翻譯、不加油添醋、保留否定」（不再要求「有詩意」），一般提示詞不動"},
)
_S5_UNCLEAR = " If the text is garbled or does not make sense, translate it as literally as you can; never apologize, ask for more context, or explain."
S5 = replace(
    S4,
    system=S4.system + _S5_UNCLEAR,
    system_lyrics=S4.system_lyrics + _S5_UNCLEAR,
    notes={"s5": "s4 的兩個系統提示詞都加「看不懂（亂碼）就照字面盡量翻，不要道歉、不要要求更多上下文、不要解釋」"},
)

# ----------------------------------------------------------------------------- s6、s7：提示詞改用中文
# s1～s5 的結果顯示英文提示詞的措辭幾乎沒有影響。Qwen 是中文為主的模型，試試指示用中文寫（輸出也是中文，語言一致）。
# s6 只把兩個系統提示詞（含略過擬聲詞的規則）換成中文，示範與使用者輪格式不動（和 v4 只差系統提示詞）；
# s7 再把每句使用者輪的格式也換成中文（「翻譯這句日文：」），示範同步。
_ZH_SKIP = (
    "如果文字只有音效、擬聲詞、笑聲、嘆息、尖叫這類沒有意義的聲音（例如 ドン、ガシャーン、ワンワン、ははは、Boom、Woof、Haha），"
    "不要翻成「砰」「汪汪」之類的中文擬聲詞，直接輸出 [SKIP]。"
)
_ZH_NO_SUBJECT = "原文沒有主詞時，譯文也不要加主詞（例如不要自己補「我」「你」「他」）。"
S6 = replace(
    V4,
    system="你是專業的翻譯。請把使用者給的文字翻譯成自然流暢的台灣繁體中文，完整保留原意，只輸出譯文，不要任何說明。" + _ZH_NO_SUBJECT,
    system_lyrics=(
        "你是專業的歌詞翻譯。使用者給的是一句歌詞（可能只是片段）。請翻成自然、有詩意的台灣繁體中文，保留原意、情感與意象，"
        "不要逐字直譯；用現代白話文，不要用文言文，避免「之、乃、吾、汝、爾、兮、矣、焉」這類古字。只輸出譯文，不要任何說明。" + _ZH_NO_SUBJECT
    ),
    skip_rule=" " + _ZH_SKIP,
    notes={"s6": "兩個系統提示詞（含略過擬聲詞規則）改成中文，其餘同 v4"},
)
S7 = replace(
    S6,
    user_fmt="翻譯這句{name}：\n{text}",
    names={"ja": "日文", "en": "英文"},
    notes={"s7": "s6 再把每句使用者輪的格式改成中文「翻譯這句日文：」（示範、前文同步）"},
)

# ----------------------------------------------------------------------------- s8：s7（全中文）＋ s2 的擴充示範
# 盲測（3B、dev、只評兩邊譯文不同的句子）：s2 35:26（p=0.31）、s5 37:44、s7 50:35（p=0.13，通順度 +0.41）。s7 方向最明確，
# s2 的示範也沒有變差，兩個合起來。
S8 = replace(
    S7,
    shots=S2_SHOTS,
    shots_lyrics=S2_SHOTS_LYRICS,
    notes={"s8": "s7（中文系統提示詞與使用者輪）＋ s2 擴充後的示範"},
)

# ----------------------------------------------------------------------------- 之後的版本寫在下面
VARIANTS: dict[str, Spec | None] = {
    "production": None,  # 不裝任何東西：量目前正式程式碼的提示詞
    "v4": V4,
    "s1": S1,
    "s2": S2,
    "s3": S3,
    "s4": S4,
    "s5": S5,
    "s6": S6,
    "s7": S7,
    "s8": S8,
}


def get(name: str) -> Spec | None:
    if name not in VARIANTS:
        raise SystemExit(f"沒有這個版本：{name}（可用：{', '.join(VARIANTS)}）")
    return VARIANTS[name]


__all__ = ["VARIANTS", "Spec", "get", "install", "replace"]
