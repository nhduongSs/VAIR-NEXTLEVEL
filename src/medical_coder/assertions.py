"""Phát hiện assertion bằng luật, có phạm vi giới hạn.

Pipeline đang emit assertion rỗng cho mọi concept. Đó từng là lựa chọn đúng khi
mù: gán sai một assertion là mất trọn Jaccard của concept đó, còn đoán rỗng đúng
với ground truth rỗng thì được 1.0. Nhưng nhãn giả cho thấy 24-27% concept có
assertion, nên đang bỏ trắng một phần đáng kể của thành phần chiếm 30% điểm.

Ngưỡng hoà vốn: emit đúng được +1, emit nhầm mất -1 (vì đang được 1.0 miễn phí),
nên có lời khi độ chính xác vượt `1/(1+e)` với `e` là độ đúng của tập nhãn —
khoảng 50-56%. Thấp, nhưng chỉ đạt được nếu luật có phạm vi chặt.

Ba loại có bản chất khác hẳn nhau, đo trên chính nhãn giả:

* `isNegated` — cục bộ và học được: "không" xuất hiện trước 70% mention bị phủ
  định. Dùng cue trong một cửa sổ ngắn, chặn khi gặp ranh giới mệnh đề.
* `isHistorical` — chỉ dùng cue TRONG CÂU. Suy theo tiêu đề mục ("mọi mention
  dưới mục Tiền sử đều là tiền sử") gán cho 27.6% concept với precision 57-62%,
  còn cue cục bộ gán 5.1% với precision 84%. Quan trọng hơn con số precision:
  bản theo mục được +0.83 trên bộ nhãn `sol` nhưng chỉ +0.39 trên `terra` —
  chênh 2.1 lần, mà `sol` chính là bộ gán assertion rộng nhất. Đó là dấu hiệu
  đang khớp với thiên lệch của một bộ nhãn chứ không phải cải thiện thật. Bản
  cue cục bộ cho +0.372 và +0.397, gần như trùng nhau, nên lợi ích không phụ
  thuộc vào việc tin bộ nào.
* `isFamily` — chỉ 70 lượt trên hai bộ nhãn, cue yếu và lẫn với ngữ cảnh bệnh
  nhân. Bật lên gần như chắc chắn emit thừa, nên mặc định TẮT.
"""
from __future__ import annotations

import re

from .models import AssertionType, EntityType

# Chỉ ba loại này mới mang assertion theo đặc tả.
ASSERTABLE = frozenset(
    {EntityType.SYMPTOM, EntityType.DIAGNOSIS, EntityType.MEDICATION}
)

# Dấu hiệu phủ định. Xếp trước các cụm dài để khớp cụm trước từ đơn.
NEGATION_CUES = (
    "chưa ghi nhận", "không ghi nhận", "chưa phát hiện", "không phát hiện",
    "không thấy", "chưa thấy", "phủ nhận", "loại trừ", "không có", "chưa có",
    "không bị", "chưa bị", "âm tính", "không", "chưa",
)

# Ranh giới mệnh đề: qua các dấu này thì phạm vi phủ định coi như đã đóng.
_CLAUSE_BREAK = re.compile(r"[.;:\n]|\b(?:nhưng|song|tuy nhiên|còn|mà)\b")

# Tiêu đề mục — chỉ dùng khi bật `use_sections`, mặc định TẮT. Xem docstring.
HISTORY_SECTIONS = (
    "tiền sử", "bệnh sử", "trước khi nhập viện", "trước nhập viện",
    "thuốc đang dùng trước", "quá trình bệnh lý",
)

# Cue tiền sử ngay trong câu.
HISTORY_CUES = ("tiền sử", "đã từng", "trước đây", "trước đó", "đã được chẩn đoán")

FAMILY_CUES = (
    "bố", "mẹ", "cha", "anh trai", "chị gái", "ông", "bà", "gia đình",
    "người thân", "bố mẹ", "con trai", "con gái",
)


def _window_before(document: str, start: int, width: int) -> str:
    """Văn bản ngay trước mention, không vượt quá đầu dòng."""
    line_start = document.rfind("\n", 0, start) + 1
    return document[max(line_start, start - width) : start].lower()


def _has_cue_in_scope(window: str, cues: tuple[str, ...]) -> bool:
    """True khi có cue mà giữa nó và mention không có ranh giới mệnh đề.

    Kiểm tra ranh giới là điều tách luật này khỏi 'tìm thấy từ không ở đâu đó
    phía trước'. Không có nó thì 'không sốt, có ho' sẽ gán nhầm phủ định cho ho.
    """
    for cue in cues:
        position = window.rfind(cue)
        if position == -1:
            continue
        if not _CLAUSE_BREAK.search(window[position + len(cue) :]):
            return True
    return False


def _section_heading(document: str, start: int, lookback: int = 400) -> str:
    """Các dòng ngay trên mention, để nhận ra mục đang đứng trong."""
    begin = max(0, start - lookback)
    return document[begin : document.rfind("\n", 0, start) + 1].lower()


def detect(
    document: str,
    start: int,
    end: int,
    entity_type: EntityType,
    *,
    negation_window: int = 40,
    history_window: int = 60,
    use_sections: bool = False,
    enable_family: bool = False,
) -> list[AssertionType]:
    """Assertion cho một mention. Trả rỗng cho loại không mang assertion."""
    if entity_type not in ASSERTABLE:
        return []

    found: list[AssertionType] = []

    if _has_cue_in_scope(_window_before(document, start, negation_window), NEGATION_CUES):
        found.append(AssertionType.NEGATED)

    historical = _has_cue_in_scope(
        _window_before(document, start, history_window), HISTORY_CUES
    )
    if use_sections and not historical:
        heading = _section_heading(document, start)
        # Chỉ xét vài dòng cuối: tiêu đề mục ở xa không còn chi phối.
        recent = "\n".join(heading.splitlines()[-4:])
        historical = any(section in recent for section in HISTORY_SECTIONS)
    if historical:
        found.append(AssertionType.HISTORICAL)

    if enable_family and _has_cue_in_scope(
        _window_before(document, start, 50), FAMILY_CUES
    ):
        found.append(AssertionType.FAMILY)

    return found
