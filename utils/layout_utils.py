"""OCR 文本行合并与布局计算。"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from app.models import TextRegion
from services.ocr.base import OCRLine
from utils.language_utils import is_romanized_japanese, needs_translation, same_script_class, script_class


def filter_by_confidence(lines: list[OCRLine], min_confidence: float) -> list[OCRLine]:
    return [line for line in lines if line.confidence >= min_confidence]


def group_into_rows(lines: list[OCRLine], y_tolerance_ratio: float = 0.5) -> list[list[OCRLine]]:
    """按纵向重叠把文本行分组，返回按阅读顺序排列的行组。"""
    ordered = sorted(lines, key=lambda line: (line.box[1], line.box[0]))
    rows: list[list[OCRLine]] = []
    for line in ordered:
        _, top, _, height = line.box
        bottom = top + height
        placed = False
        for row in rows:
            row_top = min(item.box[1] for item in row)
            row_bottom = max(item.box[1] + item.box[3] for item in row)
            row_height = max(item.box[3] for item in row)
            overlap = min(row_bottom, bottom) - max(row_top, top)
            # 纵向重叠比例达到 (1-容差) 才视为同一行，避免把相邻两行并到一起
            required = (1.0 - y_tolerance_ratio) * min(height, row_height)
            if overlap >= required:
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])
    for row in rows:
        row.sort(key=lambda item: item.box[0])
    return rows


CHAR_WIDTH_PER_HEIGHT = 0.6  # 估計一個字的寬度 = 行高 × 0.6
MIN_CHAR_WIDTH = 4.0


def x_gap_limit(line_height: float, x_gap_ratio: float) -> float:
    """同一行內兩個文字塊的空隙小於這個像素數就會接成一塊（設定的「橫向合併閾值」換算成像素）。"""
    return x_gap_ratio * max(MIN_CHAR_WIDTH, line_height * CHAR_WIDTH_PER_HEIGHT)


def merge_row_into_region(
    row: list[OCRLine], x_gap_ratio: float = 0.8
) -> tuple[tuple[int, int, int, int], str, float]:
    """同一行内的相邻文本合并成一个文本块。"""
    if not row:
        return (0, 0, 0, 0), "", 1.0
    merged = [row[0]]
    for line in row[1:]:
        prev = merged[-1]
        prev_x, prev_top, prev_w, prev_h = prev.box
        cur_x, cur_top, cur_w, cur_h = line.box
        prev_right = prev_x + prev_w
        gap = cur_x - prev_right
        if gap < x_gap_limit(max(prev_h, cur_h), x_gap_ratio):
            merged.append(line)
        else:
            yield _region_from_lines(merged)
            merged = [line]
    yield _region_from_lines(merged)


def _region_from_lines(lines: list[OCRLine]) -> tuple[tuple[int, int, int, int], str, float]:
    left = min(line.box[0] for line in lines)
    top = min(line.box[1] for line in lines)
    right = max(line.box[0] + line.box[2] for line in lines)
    bottom = max(line.box[1] + line.box[3] for line in lines)
    text = " ".join(line.text.strip() for line in lines if line.text.strip())
    confidence = min(line.confidence for line in lines)
    return (left, top, right - left, bottom - top), text, confidence


def merge_lines(
    lines: list[OCRLine],
    min_confidence: float = 0.6,
    y_tolerance_ratio: float = 0.5,
    x_gap_ratio: float = 0.8,
) -> list[tuple[tuple[int, int, int, int], str, float]]:
    """过滤 -> 分行 -> 合并，返回 (box, text, confidence) 列表。"""
    kept = filter_by_confidence(lines, min_confidence)
    result: list[tuple[tuple[int, int, int, int], str, float]] = []
    for row in group_into_rows(kept, y_tolerance_ratio):
        for region in merge_row_into_region(row, x_gap_ratio):
            result.append(region)
    return result


def ocr_lines_to_regions(
    lines: list[OCRLine],
    offset_x: int,
    offset_y: int,
    min_confidence: float = 0.6,
    y_tolerance_ratio: float = 0.5,
    x_gap_ratio: float = 0.8,
    screen_index: int = 0,
) -> list[TextRegion]:
    regions: list[TextRegion] = []
    for box, text, confidence in merge_lines(lines, min_confidence, y_tolerance_ratio, x_gap_ratio):
        if not text:
            continue
        x, y, w, h = box
        regions.append(
            TextRegion(
                text=text,
                x=offset_x + x,
                y=offset_y + y,
                width=w,
                height=h,
                source_line_height=float(h),
                confidence=confidence,
                screen_index=screen_index,
            )
        )
    return regions


def vertical_regions(
    lines: list[OCRLine],
    *,
    image_width: int,
    image_height: int,
    offset_x: int,
    offset_y: int,
    min_confidence: float,
    y_tolerance_ratio: float,
    x_gap_ratio: float,
    target_language: str,
    max_width_ratio: float,
    merge_centered: bool,
    han_is_source: bool,
) -> list[TextRegion]:
    """直排文字（由上往下讀、欄由右往左）的版面：把每一欄當成「一行」，沿用橫排的分行與合併。

    座標轉成「順時針轉 90 度」的空間：原圖的上→下變成左→右（一欄變成一行），右→左變成上→下
    （右邊的欄排在前面）。橫排的規則（同一行內相鄰的接起來、靠左對齊的相鄰行併成一段）在這個空間裡
    剛好就是「同一欄內相鄰的接起來、頂端對齊的相鄰欄併成一段」。算完再轉回原圖座標。

    ``image_width``／``image_height``：擷取影像的尺寸（OCR 座標是影像內的座標）；
    ``offset_x``／``offset_y``：影像左上角在螢幕上的位置。
    """
    turned = [
        replace(line, box=(line.box[1], image_width - (line.box[0] + line.box[2]), line.box[3], line.box[2]))
        for line in lines
    ]
    regions = ocr_lines_to_regions(
        turned, 0, 0,
        min_confidence=min_confidence,
        y_tolerance_ratio=y_tolerance_ratio,
        x_gap_ratio=x_gap_ratio,
    )
    regions = merge_wrapped_labels(
        regions,
        target_language=target_language,
        capture_width=image_height,  # 轉過之後的「寬」就是原圖的高
        max_width_ratio=max_width_ratio,
        merge_centered=merge_centered,
        han_is_source=han_is_source,
    )
    for region in regions:
        turned_x, turned_y, turned_width, turned_height = region.x, region.y, region.width, region.height
        region.x = offset_x + image_width - turned_y - turned_height
        region.y = offset_y + turned_x
        region.width, region.height = turned_height, turned_width
        region.vertical = True
    return regions


def merge_wrapped_labels(
    regions: list[TextRegion],
    target_language: str,
    capture_width: int,
    max_width_ratio: float = 0.36,
    merge_centered: bool = False,
    han_is_source: bool = False,
) -> list[TextRegion]:
    """合并被 OCR 拆开的标签和段落，同时保留原始单行字号参考。

    只处理单列宽度内且左边缘对齐的相邻行，避免把表格不同列乱并。
    基准名称、文件名等不翻译内容也不会被吸进译文框。

    ``max_width_ratio``：單行寬度上限佔擷取寬度的比例（上游預設 0.36，只適合窄標籤；
    字幕與對白通常很寬，可調到 1.0）。``merge_centered``：置中對齊（中心點接近）的相鄰行
    也視為同一段，用於置中字幕。``han_is_source``：翻譯器把純漢字當來源語言。
    """
    if len(regions) < 2:
        return regions

    remaining = sorted(regions, key=lambda item: (item.y, item.x))
    max_label_width = max(180, int(round(max(1, capture_width) * max(0.05, max_width_ratio))))
    merged: list[TextRegion] = []
    while remaining:
        current = remaining.pop(0)
        source_height = current.source_line_height or float(current.height)

        while remaining:
            match_index: int | None = None
            for candidate_index, candidate in enumerate(remaining):
                candidate_height = candidate.source_line_height or float(candidate.height)
                gap_tolerance = max(
                    8, int(round(max(source_height, candidate_height) * 0.8))
                )
                gap = candidate.y - current.bottom
                if gap > gap_tolerance:
                    break
                # 大字體、藝術字的 OCR 框常彼此重疊；輕微重疊仍視為相鄰行
                overlap_tolerance = int(round(max(source_height, candidate_height) * 0.5))
                x_tolerance = max(
                    5, int(round(max(source_height, candidate_height) * 0.35))
                )
                left_aligned = abs(candidate.x - current.x) <= x_tolerance
                centered = merge_centered and (
                    abs(candidate.center[0] - current.center[0]) <= x_tolerance
                )
                adjacent = (left_aligned or centered) and gap >= -overlap_tolerance
                # 日文／羅馬拼音／中文譯文疊成幾行的字幕不能併成一段，也不能跳過夾在中間的那一行去併下面的
                same_script = same_script_class(current.text, candidate.text)
                if adjacent and not same_script:
                    break
                if (
                    current.width <= max_label_width
                    and candidate.width <= max_label_width
                    and adjacent
                    and needs_translation(current.text, target_language, han_is_source)
                    and needs_translation(candidate.text, target_language, han_is_source)
                    and same_script
                ):
                    match_index = candidate_index
                    break
            if match_index is None:
                break

            candidate = remaining.pop(match_index)
            candidate_height = candidate.source_line_height or float(candidate.height)
            right = max(current.right, candidate.right)
            bottom = max(current.bottom, candidate.bottom)
            current.text = f"{current.text.rstrip()} {candidate.text.lstrip()}"
            current.width = right - current.x
            current.height = bottom - current.y
            current.source_line_height = max(source_height, candidate_height)
            current.source_line_count += max(1, candidate.source_line_count)
            current.confidence = min(current.confidence, candidate.confidence)
            source_height = current.source_line_height

        merged.append(current)

    return sorted(merged, key=lambda item: (item.y, item.x))


def _stacked(first: TextRegion, second: TextRegion, max_gap_lines: float) -> bool:
    """兩個文字塊是不是上下疊放：垂直距離不超過 ``max_gap_lines`` 行，且置中或靠左對齊。"""
    line = max(first.source_line_height or first.height, second.source_line_height or second.height, 1.0)
    gap = max(second.y - first.bottom, first.y - second.bottom)  # 負值 = 垂直方向重疊（同一列，不是疊起來）
    if not -0.3 * line <= gap <= max_gap_lines * line:
        return False
    centered = abs(first.center[0] - second.center[0]) <= 0.2 * max(first.width, second.width)
    left_aligned = abs(first.x - second.x) <= max(8.0, 0.5 * line)
    return centered or left_aligned


def companion_translation_texts(regions: list[TextRegion], max_gap_lines: float = 3.0) -> set[str]:
    """字幕檔自己附的中文譯文：疊在日文字幕上下、純漢字的那幾行。

    日文漢字與中文無法區分，所以純漢字文字預設當日文來源翻譯；但雙語字幕常在日文下方
    （或上方）疊一行中文，把它再當日文翻一遍只會得到亂改過的句子。符合下列條件的純漢字行
    視為附帶的中文譯文，呼叫端略過不翻（它本來就看得懂）：

    - 疊在含假名的字幕旁（``max_gap_lines`` 行以內，中間可能夾著羅馬拼音），置中或靠左對齊；並且
    - 看得出是中文（``chinese.looks_chinese``），或同一組字幕裡有一行羅馬拼音（日文／羅馬拼音／
      中文三行是歌詞字幕的典型排法）。

    只有日文漢字的標語（「20年連続」）沒有這些證據，仍然照日文翻。
    """
    from utils import chinese

    kana_regions = [region for region in regions if script_class(region.text) == "ja"]
    if not kana_regions:
        return set()
    romaji_regions = [region for region in regions if is_romanized_japanese(region.text)]
    companions: set[str] = set()
    for region in regions:
        if script_class(region.text) != "han":
            continue
        for kana in kana_regions:
            if not _stacked(region, kana, max_gap_lines):
                continue
            trilingual = any(
                _stacked(romaji, kana, max_gap_lines) or _stacked(romaji, region, max_gap_lines)
                for romaji in romaji_regions
            )
            if trilingual or chinese.looks_chinese(region.text):
                companions.add(region.text)
                break
    return companions


def clamp_region(region: TextRegion, bounds: tuple[int, int, int, int]) -> TextRegion:
    """把区域限制在屏幕物理边界内，越界则平移 / 收缩。"""
    left, top, right, bottom = bounds
    x = max(left, min(region.x, right - 1))
    y = max(top, min(region.y, bottom - 1))
    x2 = max(x + 1, min(region.right, right))
    y2 = max(y + 1, min(region.bottom, bottom))
    region.x, region.y = x, y
    region.width, region.height = x2 - x, y2 - y
    return region


# ----------------------------------------------------------------------------- 譯文面板（框選區域下方）
PLACEMENT_BELOW = "below"
PLACEMENT_IN_PLACE = "in_place"


def uses_translation_panel(placement: str | None, capture_mode: str | None) -> bool:
    """譯文是否放在框選區域下方（原文不被遮住）。

    只有框選區域（``region``）有「框」可以放在下方；全螢幕、視窗擷取的文字塊遍布整個畫面，
    仍然逐塊蓋在原文位置上。
    """
    return (placement or PLACEMENT_BELOW) == PLACEMENT_BELOW and capture_mode == "region"


def translation_lines(regions: Iterable[TextRegion]) -> list[str]:
    """譯文面板要顯示的文字，依閱讀順序（由上到下，同一列由左到右；直排文字是由右到左）。

    略過空白，以及沒翻成功（譯文與原文相同）的文字塊：原文本來就在畫面上，重複顯示只是雜訊。
    同一列的多個文字塊接成一行。
    """
    translated = [
        region
        for region in regions
        if (region.translated_text or "").strip()
        and region.translated_text.strip() != region.text.strip()
    ]
    if any(region.vertical for region in translated):
        # 直排：欄由右往左讀，每一欄（段）各佔一行
        ordered = sorted(translated, key=lambda item: (-(item.x + item.width), item.y))
        return [region.translated_text.strip() for region in ordered]
    rows: list[list[TextRegion]] = []
    for region in sorted(translated, key=lambda item: (item.y, item.x)):
        middle = region.y + region.height / 2
        row = rows[-1] if rows else None
        # 垂直中心落在目前這一列的範圍內，就視為同一列
        if row is not None and min(r.y for r in row) <= middle <= max(r.bottom for r in row):
            row.append(region)
        else:
            rows.append([region])
    return ["  ".join(r.translated_text.strip() for r in sorted(row, key=lambda item: item.x)) for row in rows]
