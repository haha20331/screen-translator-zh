"""提示詞調整用的題庫：自己寫的台詞與歌詞（附參考譯文），以及從程式的輸出紀錄取出的真實歌詞。

- ``SCENES``：本檔自己寫的場景（原文與參考譯文都是原創，可以放進版控）。一個場景＝一段連續的字幕，
  依序翻譯，這樣「上下文兩句」才會有前文可用。參考譯文是人工寫的台灣用語，不是唯一答案，只用來算相似度。
  參考譯文是 ``[SKIP]`` 的句子是擬聲詞／笑聲／吶喊，應該略過不翻。
- ``split``：``dev`` 是調提示詞時反覆看的那一批；``holdout`` 調整期間不看輸出，最後才拿來驗證有沒有調過頭。
- ``log_scenes()``：從 ``%LOCALAPPDATA%\\ScreenTranslator\\output\\*.txt`` 取連續的幾段（真實 OCR 結果，含雜訊與重複，
  沒有參考譯文）。內容多半有版權，**只在本機用，不寫進提交的文件**。
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

SKIP = "[SKIP]"


def scene(id_: str, kind: str, lang: str, split: str, title: str, lines: list[tuple[str, str]]) -> dict:
    return {"id": id_, "kind": kind, "lang": lang, "split": split, "title": title, "lines": lines}


SCENES: list[dict] = [
    # ------------------------------------------------------------------ 日文台詞（dev）
    scene("ja-d1", "dialogue", "ja", "dev", "放學後的教室（口語、省略主詞）", [
        ("あれ、まだ帰ってなかったの？", "咦，你還沒回去嗎？"),
        ("うん、ちょっと先生に呼ばれててさ", "嗯，剛才被老師叫去了一下"),
        ("また宿題忘れたんでしょ", "你又忘記寫作業了吧"),
        ("違うよ！今回はちゃんとやったってば", "才不是！這次我有好好寫啦"),
        ("ほんとかなあ…", "真的嗎……"),
        ("ほんとだって。見てよ、ほら", "真的啦。你看，喏"),
        ("あはは、ごめんごめん。疑って悪かった", "哈哈，抱歉抱歉。是我不該懷疑你"),
    ]),
    scene("ja-d2", "dialogue", "ja", "dev", "戰鬥（硬派口氣、擬聲詞）", [
        ("ここから先は通さん！", "休想再往前一步！"),
        ("どいてください。私たちは先を急いでいるんです", "請讓開。我們趕時間"),
        ("ならば力ずくで通ってみろ", "那就用實力闖過去吧"),
        ("ガキン！", SKIP),
        ("くっ…なんて硬い鎧なの", "可惡……這是什麼堅硬的鎧甲"),
        ("回復します！持ちこたえて！", "我來治療！撐住！"),
        ("ああ、まだ戦える", "嗯，我還能戰"),
    ]),
    scene("ja-d3", "dialogue", "ja", "dev", "職場（敬語）", [
        ("お疲れ様です。例の資料、確認していただけましたか", "辛苦了。上次那份資料，您確認過了嗎"),
        ("ああ、見たよ。数字が少し合っていないようだけど", "嗯，看過了。數字好像有點對不上"),
        ("申し訳ありません。すぐに修正してお送りします", "非常抱歉。我馬上修正後寄給您"),
        ("明日の会議までに間に合うかな", "趕得上明天的會議嗎"),
        ("はい、今夜中に仕上げます", "可以，今晚之前完成"),
        ("無理しなくていいからね", "不用勉強自己喔"),
    ]),
    scene("ja-d4", "dialogue", "ja", "dev", "家裡（關西腔）", [
        ("なんでやねん、そんなん聞いてへんで", "為什麼啊，這種事我可沒聽說過"),
        ("言うたやんか、昨日の晩に", "我說過啦，昨天晚上"),
        ("ほんまに？覚えてへんわ", "真的嗎？我不記得耶"),
        ("もうええわ。ご飯できたから、はよ食べ", "算了啦。飯做好了，快點吃"),
        ("ありがとう、お母さん", "謝謝，媽媽"),
        ("ええから、冷めんうちにな", "不用客氣，趁還沒涼趕快吃"),
    ]),
    scene("ja-d5", "dialogue", "ja", "dev", "車站廣播", [
        ("まもなく、2番線に快速電車が参ります", "快速電車即將進入2號月台"),
        ("危ないですから、黄色い線の内側までお下がりください", "為了安全，請退到黃線內側等候"),
        ("ピンポーン", SKIP),
        ("ご乗車ありがとうございました", "感謝您的搭乘"),
        ("次は、新宿、新宿です", "下一站，新宿，新宿"),
        ("お忘れ物のないようご注意ください", "請注意不要遺落隨身物品"),
    ]),
    scene("ja-d9", "dialogue", "ja", "dev", "天氣預報與遊戲介面", [
        ("今日の東京は午後から雨になるでしょう", "今天東京下午起會下雨"),
        ("最高気温は18度の予想です", "預測最高氣溫為18度"),
        ("傘をお持ちください", "請攜帶雨傘"),
        ("セーブしますか？", "要存檔嗎？"),
        ("データをダウンロード中です", "正在下載資料"),
        ("はい　いいえ", "是　否"),
    ]),
    # ------------------------------------------------------------------ 日文台詞（holdout）
    scene("ja-d6", "dialogue", "ja", "holdout", "告白", [
        ("あの…ちょっといいかな", "那個……可以打擾一下嗎"),
        ("どうしたの？急に改まって", "怎麼了？突然這麼嚴肅"),
        ("ずっと言えなかったことがあるんだ", "有件事，我一直沒能說出口"),
        ("実は…君のことが好きなんだ", "其實……我喜歡你"),
        ("えっ…本当に？", "咦……真的嗎？"),
        ("うそじゃないよ。返事は急がないから", "不是騙你的。不用急著回覆我"),
        ("…私も、同じ気持ちだよ", "……我也是一樣的心情"),
    ]),
    scene("ja-d7", "dialogue", "ja", "holdout", "餐廳", [
        ("いらっしゃいませ。何名様ですか", "歡迎光臨。請問幾位？"),
        ("二人です。禁煙席でお願いします", "兩位。麻煩給我禁菸座位"),
        ("かしこまりました。こちらへどうぞ", "好的。請往這邊走"),
        ("このおすすめランチ、まだありますか", "這個推薦午餐，還有嗎"),
        ("申し訳ございません、本日分は終了いたしました", "非常抱歉，今天的份已經賣完了"),
        ("そっか、じゃあこっちのカレーで", "這樣啊，那我點這個咖哩"),
    ]),
    scene("ja-d8", "dialogue", "ja", "holdout", "奇幻（專有名詞、反派口氣）", [
        ("ようやく目覚めたか、勇者アレクよ", "終於醒了嗎，勇者亞歷克"),
        ("お前は…魔王ゼノン！なぜここに", "你是……魔王澤農！為什麼會在這裡"),
        ("ふふ、お前の仲間は既に倒れたぞ", "呵呵，你的夥伴已經倒下了"),
        ("そんな…みんなが", "怎麼會……大家"),
        ("絶望するがいい", "盡情絕望吧"),
        ("いや、まだ終わっていない。この剣にかけて", "不，還沒結束。我以這把劍起誓"),
    ]),
    scene("ja-d10", "dialogue", "ja", "holdout", "追逐（短句、驚呼）", [
        ("まさか、そんなはずは…", "怎麼可能，不可能的……"),
        ("くそっ！逃げられた！", "可惡！被他逃掉了！"),
        ("落ち着いて。まだ遠くへは行ってないはず", "冷靜點。他應該還沒走遠"),
        ("うわあああ！", SKIP),
        ("あっちだ！追え！", "在那邊！追！"),
        ("待ってよ、置いていかないで", "等等我，不要丟下我"),
    ]),
    # ------------------------------------------------------------------ 英文台詞
    scene("en-d1", "dialogue", "en", "dev", "家裡的日常", [
        ("Hey, did you see my keys anywhere?", "嘿，你有看到我的鑰匙嗎？"),
        ("They're on the kitchen table, where you left them.", "在廚房桌上，你放的地方。"),
        ("Oh, thank goodness. I'm already late.", "哦，謝天謝地。我已經遲到了。"),
        ("Are you coming home for dinner tonight?", "你今晚會回家吃晚餐嗎？"),
        ("Probably not. Don't wait up for me.", "大概不會。不用等我。"),
        ("Okay, be careful out there.", "好，路上小心。"),
    ]),
    scene("en-d2", "dialogue", "en", "dev", "太空艦橋（擬聲詞）", [
        ("Captain, we're picking up a strange signal.", "艦長，我們偵測到一個奇怪的訊號。"),
        ("Where is it coming from?", "它是從哪裡傳來的？"),
        ("Deep inside the asteroid belt, sir.", "從小行星帶深處，長官。"),
        ("Boom!", SKIP),
        ("Shields are down to forty percent!", "護盾剩下百分之四十！"),
        ("Get us out of here, now!", "立刻帶我們離開這裡！"),
    ]),
    scene("en-d3", "dialogue", "en", "dev", "面試（正式）", [
        ("Thank you for coming in today.", "感謝你今天前來。"),
        ("Could you tell us a little about your previous experience?", "可以請你簡單介紹一下你過去的經驗嗎？"),
        ("Sure. I spent five years leading a small design team.", "當然。我帶領過一個小型設計團隊五年。"),
        ("What made you decide to leave?", "是什麼讓你決定離開？"),
        ("I wanted a bigger challenge, to be honest.", "老實說，我想要更大的挑戰。"),
    ]),
    scene("en-d4", "dialogue", "en", "holdout", "電影（情緒）", [
        ("I never meant to hurt you.", "我從來沒有想傷害你。"),
        ("Then why didn't you tell me the truth?", "那你為什麼不告訴我真相？"),
        ("Because I was afraid you'd leave.", "因為我怕你會離開。"),
        ("Sigh...", SKIP),
        ("I'm not going anywhere. Not this time.", "我哪裡都不會去。這次不會。"),
    ]),
    scene("en-d5", "dialogue", "en", "holdout", "遊戲任務與提示", [
        ("New quest available: Find the lost cat.", "有新任務：尋找走失的貓。"),
        ("Talk to the old woman near the fountain.", "去和噴水池旁的老婆婆談談。"),
        ("You have received 200 gold.", "你獲得了 200 金幣。"),
        ("Your inventory is full.", "你的背包已滿。"),
        ("Press E to open the door.", "按 E 開門。"),
        ("Haha, nice try!", "哈哈，想得美！"),
    ]),
    # ------------------------------------------------------------------ 日文歌詞（dev）
    scene("ja-l1", "lyrics", "ja", "dev", "夏末的離別", [
        ("夏の終わりの風が吹いて", "夏末的風吹起"),
        ("君の背中が遠くなる", "你的背影漸行漸遠"),
        ("言えなかった言葉だけが", "只有沒能說出口的話"),
        ("胸の奥で光ってる", "還在心底閃閃發亮"),
        ("さよならは言わないよ", "我不會說再見"),
    ]),
    scene("ja-l2", "lyrics", "ja", "dev", "失眠的夜（意象）", [
        ("眠れない夜は窓を開けて", "睡不著的夜晚就打開窗"),
        ("星のかけらを数えている", "數著星星的碎片"),
        ("どこかで君も見てるかな", "你是否也在某處看著呢"),
        ("同じ空の下", "在同一片天空下"),
        ("ラララ…", SKIP),
    ]),
    scene("ja-l3", "lyrics", "ja", "dev", "加油打氣（重複句型）", [
        ("走れ走れ どこまでも", "跑吧跑吧 無論到哪裡"),
        ("転んだって立ち上がれ", "就算跌倒也要站起來"),
        ("涙のあとには虹がかかる", "淚水之後會有彩虹出現"),
        ("信じてごらん 未来を", "試著去相信吧 未來"),
        ("ハイ！ハイ！", SKIP),
    ]),
    scene("ja-l4", "lyrics", "ja", "dev", "易碎的戀情（比喻）", [
        ("触れたら壊れそうな恋だった", "那是一段彷彿一碰就碎的戀情"),
        ("雨に濡れた街角で", "在被雨淋濕的街角"),
        ("傘もささずに笑ってた", "你連傘也沒撐，笑著"),
        ("あの日の約束はまだ", "那一天的約定仍然"),
        ("心の中で生きている", "活在心裡"),
    ]),
    scene("ja-l5", "lyrics", "ja", "dev", "追夢", [
        ("弱い自分に負けないで", "不要輸給軟弱的自己"),
        ("この手で夢をつかみたい", "想用這雙手抓住夢想"),
        ("たとえ誰かに笑われても", "就算被誰嘲笑"),
        ("僕は僕のまま歩いていく", "我也要以我自己的樣子走下去"),
    ]),
    # ------------------------------------------------------------------ 日文歌詞（holdout）
    scene("ja-l6", "lyrics", "ja", "holdout", "春天與櫻花", [
        ("桜舞う坂道で", "在櫻花紛飛的坡道上"),
        ("初めて君に出会った", "第一次遇見了你"),
        ("春風がそっと背中を押して", "春風輕輕推著我的背"),
        ("新しい季節へ", "走向新的季節"),
        ("ありがとうをここに残して", "把謝謝留在這裡"),
    ]),
    scene("ja-l7", "lyrics", "ja", "holdout", "孤獨（抽象）", [
        ("声にならない叫びが", "無法化為聲音的吶喊"),
        ("夜の底でこだまする", "在夜的深處迴盪"),
        ("それでも朝は来るのだろう", "即便如此，早晨仍會到來吧"),
        ("光を探して", "尋找光"),
    ]),
    scene("ja-l8", "lyrics", "ja", "holdout", "友情（輕快）", [
        ("ねえ聞いて 今日のこと", "欸 聽我說 今天的事"),
        ("くだらない話ばっかりで", "盡是些無聊的話題"),
        ("笑い転げた帰り道", "笑到不行的回家路上"),
        ("ずっと友達でいようね", "我們要永遠當朋友喔"),
        ("イエーイ！", SKIP),
    ]),
    # ------------------------------------------------------------------ 英文歌詞
    scene("en-l1", "lyrics", "en", "dev", "夜路", [
        ("Walking down this empty road tonight", "今晚走在這條空蕩蕩的路上"),
        ("Every shadow whispers your name", "每一道影子都在低語著你的名字"),
        ("I would give it all to hold you again", "我願付出一切，只為再擁抱你一次"),
        ("Oh oh oh", SKIP),
    ]),
    scene("en-l2", "lyrics", "en", "dev", "青春", [
        ("We were young and we were reckless", "我們年輕又魯莽"),
        ("Burning bridges, chasing the sun", "燒毀橋樑，追逐太陽"),
        ("Now the stars are all that's left", "如今只剩星星留下"),
        ("But I'd do it all again", "但我還是會再來一次"),
    ]),
    scene("en-l3", "lyrics", "en", "holdout", "風暴", [
        ("Hold on, the storm will pass", "撐下去，風暴終會過去"),
        ("Light is breaking through the clouds", "陽光正穿透雲層"),
        ("You are stronger than you know", "你比你知道的還要堅強"),
        ("La la la", SKIP),
    ]),
]


# ----------------------------------------------------------------------------- 從紀錄檔取真實歌詞
_KANA = re.compile(r"[\u3040-\u30ff]")  # 平假名、片假名
_GAP_SECONDS = 45  # 兩句字幕間隔超過這麼久，算不同一段（不同的歌或影片）

# (第幾段, 從該段第幾句開始, 取幾句, 分到哪一批)；段的編號依時間順序、只算含假名的句子。
# 挑的是不同首歌、不同 OCR 狀況（乾淨、重複、斷字、亂碼）的連續片段。
LOG_PICKS: list[tuple[int, int, int, str]] = [
    (2, 0, 20, "dev"),
    (5, 0, 20, "dev"),
    (14, 0, 20, "dev"),
    (6, 0, 20, "dev"),
    (3, 36, 20, "holdout"),
    (11, 0, 20, "holdout"),
    (12, 0, 20, "holdout"),
    (9, 0, 20, "holdout"),
]


def _read_rows(log_dir: Path) -> list[tuple[datetime.datetime, str]]:
    rows: list[tuple[datetime.datetime, str]] = []
    for path in sorted(log_dir.glob("*.txt")):
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 6 or not _KANA.search(parts[2]):
                continue
            try:
                ended = datetime.datetime.strptime(parts[1][:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            rows.append((ended, parts[2].strip()))
    return rows


def log_scenes(log_dir: str | Path) -> list[dict]:
    """依 ``LOG_PICKS`` 從輸出紀錄取連續片段；連續重複的同一句只留一句（程式本來就不會對同一句再翻一次）。"""
    segments: list[list[str]] = []
    previous = None
    for ended, text in _read_rows(Path(log_dir)):
        if previous is None or (ended - previous).total_seconds() > _GAP_SECONDS:
            segments.append([])
        previous = ended
        if not segments[-1] or segments[-1][-1] != text:
            segments[-1].append(text)
    scenes = []
    for number, start, count, split in LOG_PICKS:
        if number > len(segments):
            continue
        lines = [(text, "") for text in segments[number - 1][start : start + count]]
        scenes.append(scene(f"log-{number}", "lyrics", "ja", f"log-{split}", f"紀錄檔第 {number} 段", lines))
    return scenes
