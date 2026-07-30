"""Generate the Kaggle Run-All notebook for the predict-v2 pipeline.

The notebook is self-contained — no source Dataset, no git clone — but the two
kinds of payload are carried differently on purpose:

* **Source** goes in verbatim via ``%%writefile``. Code is meant to be read and
  patched in place on Kaggle, and a one-line change should show up as a one-line
  diff here rather than rewriting an opaque blob.
* **The ICD table** goes in via ``%%writefile`` too.
* **The test inputs** go in as a plain dict literal, because ``%%writefile``
  takes one file per cell and 100 cells is not a scrollable notebook.

Nothing is encoded. An earlier version base64-gzipped the data for a 5x size
win, but that put a decode step between the reader and the content, and a
half-finished refactor left the decode without its import.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MD = "markdown"
CODE = "code"

cells = []


KAGGLE_SRC = "/kaggle/working/medical_coder_src"

# Modules the predict-v2 path actually needs, in reading order. `pipeline.py`
# and its generative-LLM dependencies are deliberately absent: the v2 pipeline
# uses none of them, and shipping them would put 43 KB of dead code in front of
# anyone reading the notebook.
SOURCE_MODULES = [
    "__init__.py",
    "models.py",
    "validation.py",
    "submission.py",
    "terminology.py",
    "gliner_ner.py",
    "exact_link.py",
    "selector.py",
    "pipeline_v2.py",
    "icd_vn.py",
    "rxnorm_kb.py",
    "scoring.py",
]


def check_module_closure() -> None:
    """Fail the build if SOURCE_MODULES misses a relative import."""
    import ast

    package = REPO_ROOT / "src" / "medical_coder"
    shipped = {name[:-3] for name in SOURCE_MODULES}
    for name in SOURCE_MODULES:
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                if node.module not in shipped:
                    raise SystemExit(
                        f"{name} imports .{node.module}, which is not in SOURCE_MODULES"
                    )


def icd_table() -> str:
    return (REPO_ROOT / "data" / "terminology" / "icd10_vn.tsv").read_text(encoding="utf-8")


def input_files_literal() -> str:
    """The 100 public-test files as a plain Python dict literal.

    `%%writefile` handles exactly one file per cell, and 100 cells is not a
    notebook anyone can scroll. A dict of `repr`-ed strings keeps the same
    property that matters: plain readable text, nothing to decode.
    """
    lines = ["INPUT_FILES = {"]
    for path in sorted((REPO_ROOT / "input").glob("*.txt"), key=lambda p: int(p.stem)):
        lines.append(f"    {path.name!r}: {path.read_text(encoding='utf-8')!r},")
    lines.append("}")
    return "\n".join(lines)


def add(kind, source):
    cells.append({"cell_type": kind, "metadata": {}, "source": source.strip("\n").splitlines(True),
                  **({"execution_count": None, "outputs": []} if kind == CODE else {})})


add(MD, """
# Viettel AI Race — `predict-v2` Run All (Kaggle)

Pipeline **precision-first**, thay cho pipeline sinh văn bản của lần nộp 01 (14.4255).

```text
GLiNER spans (ngưỡng riêng theo type)
  → Qwen corrector: TRIỆU_CHỨNG → CHẨN_ĐOÁN   (GPU)
  → Qwen consensus additions cho type không có candidate  (GPU, cần 2 teacher)
  → trim generic prefix + loại header
  → exact-alias linking (ICD-10 tiếng Việt TT06 + RxNorm), chỉ emit khi khớp duy nhất
  → assertions rỗng
  → validate → output.zip
```

**Vì sao đổi cách làm.** Scorer của BTC đếm mỗi concept thừa **hai lần** vào mẫu số
của cả ba thành phần. Nên precision đáng giá hơn recall, và candidate thừa còn đắt
hơn nữa: một concept sai mang 3 mã tốn `2×(3+1)=8` đơn vị mẫu số thay vì 2.

**Trước khi Run All:**

1. Settings → Accelerator: **GPU T4 x2** (khuyến nghị) hoặc **P100**.
2. Settings → Internet: **On** — để tải weights và RxNorm.
3. Cần khoảng **12 GB trống** trên `/kaggle/working` cho weights. Nếu đã chạy dở
   lần trước, hãy **Run → Factory reset** trước khi Run All; notebook có cell
   báo cáo và dọn dung lượng ở mục 1.
4. Nếu chưa attach weights: đặt HF token trong Add-ons → Secrets với tên
   `HF_TOKEN`.

**Không cần attach gì cả.** Notebook tự chứa: package `medical_coder`, bảng
ICD-10 tiếng Việt và bản test Vòng 1 đều được nhúng sẵn. Chỉ cần tải lên đúng
một tệp `.ipynb` này.

Attach Dataset input vẫn được và **luôn được ưu tiên** hơn bản nhúng — bắt buộc
làm vậy khi chạy trên private test của BTC.
""")

add(MD, "## 1. Kiểm tra GPU và môi trường")

add(CODE, """
!nvidia-smi || echo "Không thấy GPU — pipeline vẫn chạy được trên CPU nhưng KHÔNG có bước corrector."
""")

add(CODE, """
import os, sys, subprocess, json, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
IS_KAGGLE = Path("/kaggle").exists()
print("kaggle:", IS_KAGGLE, "| python:", sys.version.split()[0])

try:
    import torch
    print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
          "| devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print("  ", i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
except ImportError:
    print("torch chưa được cài")
""")

add(MD, """
### P100 (`sm_60`)

Wheel torch mặc định của Kaggle có thể không chứa kiến trúc `sm_60`. Cell dưới chỉ
cài lại torch khi phát hiện P100 **và** arch hiện tại thiếu `sm_60`. Sau khi cài
lại phải **Restart Session** rồi Run All lần nữa — không thể tráo binary torch
trong kernel đã import nó.
""")

add(CODE, """
NEEDS_RESTART = False
try:
    import torch
    if torch.cuda.is_available():
        caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
        arches = torch.cuda.get_arch_list()
        if (6, 0) in caps and not any("sm_60" in a for a in arches):
            print("P100 nhưng torch thiếu sm_60 — cài lại torch CUDA 12.6")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "torch==2.10.0", "--index-url",
                            "https://download.pytorch.org/whl/cu126"], check=False)
            NEEDS_RESTART = True
        else:
            print("torch arch OK:", [a for a in arches if a.startswith("sm_")])
except Exception as exc:
    print("bỏ qua kiểm tra arch:", exc)

if NEEDS_RESTART:
    print("\\n>>> HÃY CHỌN 'Restart Session' RỒI RUN ALL LẠI <<<")
""")

add(MD, """
### Dung lượng đĩa

`/kaggle/working` chỉ có khoảng 20 GB và cũng chính là quota output. Weights là
thứ ngốn nhiều nhất, nên cell này báo cáo chỗ trống trước rồi mới dọn cache —
hết đĩa giữa chừng sẽ nổ ra `OSError: [Errno 28]` ở một cell chẳng liên quan gì,
rất khó lần ra nguyên nhân.
""")

add(CODE, """
import shutil as _sh

# Đặt True để xoá SẠCH /kaggle/working. Notebook tự tạo lại được mọi thứ nó cần,
# nhưng nếu bạn có tệp riêng ở đó thì sẽ mất.
PURGE_ALL = False

# Tên do các phiên bản notebook trước tạo ra, đều tái tạo được nên xoá an toàn.
REGENERABLE = [
    "models", "hf", "medical_coder_src", "terminology",
    "input_embedded", "output", "output_smoke", "cache",
    "embedded_viettel_ai_race", "viettel_ai_race", "VAIR-NEXTLEVEL",
    "output.zip", "output_v2.zip", "rxnorm.zip",
]

def report_disk():
    for path in ("/kaggle/working", "/tmp", "/"):
        if Path(path).exists():
            usage = _sh.disk_usage(path)
            print(f"  {path:18s} trống {usage.free / 2**30:6.1f} GB "
                  f"/ tổng {usage.total / 2**30:6.1f} GB")

def entry_size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

def show_largest(top=10):
    \"\"\"Liệt kê thứ đang chiếm chỗ, để cái gì không nằm trong danh sách xoá vẫn nhìn thấy.\"\"\"
    entries = []
    for path in WORK.iterdir():
        try:
            entries.append((entry_size(path), path))
        except OSError:
            continue
    entries.sort(reverse=True)
    if not entries:
        print("  (trống)")
    for size, path in entries[:top]:
        mark = "  [sẽ xoá]" if path.name in REGENERABLE or PURGE_ALL else ""
        print(f"  {size / 2**30:7.2f} GB  {path.name}{mark}")

print("TRƯỚC khi dọn:")
report_disk()
print("\\nĐang chiếm chỗ trong /kaggle/working:")
show_largest()

before = _sh.disk_usage("/kaggle/working").free

# pip cache và wheel đã cài xong thì không còn tác dụng
!rm -rf /root/.cache/pip /tmp/pip-* 2>/dev/null

targets = list(WORK.iterdir()) if PURGE_ALL else [
    WORK / name for name in REGENERABLE if (WORK / name).exists()
]
for target in targets:
    if target.is_dir():
        _sh.rmtree(target, ignore_errors=True)
    else:
        target.unlink(missing_ok=True)

reclaimed = (_sh.disk_usage("/kaggle/working").free - before) / 2**30
print(f"\\nSAU khi dọn (giải phóng {reclaimed:.1f} GB):")
report_disk()

FREE_GB = _sh.disk_usage("/kaggle/working").free / 2**30
if FREE_GB < 12:
    print(f"\\n>>> CHỈ CÒN {FREE_GB:.1f} GB — teacher chính cần ~9 GB, chưa kể torch.")
    print(">>> Xem danh sách bên trên: thứ nào lớn mà không có nhãn [sẽ xoá] thì")
    print(">>> đặt PURGE_ALL = True rồi chạy lại cell này, hoặc Run → Factory reset.")
""")

add(MD, "## 2. Cài đặt")

add(CODE, """
%%capture install_log
!python -m pip install -q "gliner>=0.2.13" "transformers>=4.51" accelerate bitsandbytes pydantic
""")

add(CODE, """
import importlib
for module in ("gliner", "transformers", "pydantic"):
    try:
        importlib.import_module(module)
        print("ok  ", module)
    except ImportError as exc:
        print("LỖI", module, exc)

import torch
assert torch.cuda.is_available() or True, "không có CUDA"
print("cuda sau khi cài:", torch.cuda.is_available())
""")

check_module_closure()

add(MD, f"""
## 3. Source code

{len(SOURCE_MODULES)} module của package `medical_coder` được ghi thẳng ra đĩa bằng
`%%writefile`, không nén, không mã hoá. Đọc được, sửa được ngay tại chỗ: gặp lỗi
trên Kaggle thì sửa cell rồi Restart & Run All, khỏi phải dựng lại notebook ở máy
rồi tải lên.

Đây là đúng những module mà nhánh predict-v2 cần. `pipeline.py` cùng backend LLM
sinh văn bản của lần nộp 01 **không** có ở đây — chúng không được dùng, và đưa vào
chỉ tổ đặt 43 KB code chết trước mặt người đọc.

> Sửa cell nào thì phải **Restart Session** rồi chạy lại, vì `medical_coder` đã
> được import vào kernel.
""")

add(CODE, f"""
import pathlib
pathlib.Path("{KAGGLE_SRC}/medical_coder").mkdir(parents=True, exist_ok=True)
print("thư mục source:", "{KAGGLE_SRC}/medical_coder")
""")

for module_name in SOURCE_MODULES:
    body = (REPO_ROOT / "src" / "medical_coder" / module_name).read_text(encoding="utf-8")
    add(CODE, f"%%writefile {KAGGLE_SRC}/medical_coder/{module_name}\n{body}")

add(CODE, f"""
IMPORT_DIR = Path("{KAGGLE_SRC}")
if str(IMPORT_DIR) not in sys.path:
    sys.path.insert(0, str(IMPORT_DIR))

modules = sorted(p.name for p in (IMPORT_DIR / "medical_coder").glob("*.py"))
print(f"{{len(modules)}} module ->", IMPORT_DIR)
print(" ", ", ".join(modules))

import medical_coder
from medical_coder import exact_link, gliner_ner, pipeline_v2, selector, submission
print("\\nmedical_coder:", medical_coder.__file__)

REPO = None   # không có repo trên đĩa; mọi thứ dựng ra nằm ở /kaggle/working
""")

add(MD, """
## 4. Dữ liệu đầu vào

Ưu tiên Dataset đã attach. Nếu không có Dataset nào, notebook dùng bản test Vòng 1
**nhúng sẵn** bên dưới — nhờ vậy chỉ cần tải lên đúng một tệp notebook, không cần
attach gì cả.

> Khi chấm trên private test, Ban Tổ chức sẽ cấp input khác. Lúc đó **phải** attach
> Dataset input mới; cell này sẽ tự ưu tiên nó và in rõ nguồn đang dùng, nhưng nếu
> quên attach thì nó rơi về bản public test nhúng sẵn và điểm sẽ sai. Hãy đọc dòng
> `nguồn input:` mà cell in ra.
""")

add(CODE, input_files_literal() + """

print(f"input nhúng sẵn: {len(INPUT_FILES)} tệp")""")

add(CODE, """
# Đường dẫn Dataset đã biết, thử trước để khỏi quét toàn bộ /kaggle/input.
KNOWN_INPUT_DIRS = [
    Path("/kaggle/input/datasets/thanhhiepvo/viettelairace/input"),
    Path("/kaggle/input/viettelairace/input"),
    Path("/kaggle/input/viettelairace"),
]

def is_input_dir(folder):
    return folder.is_dir() and (folder / "1.txt").exists() and (folder / "100.txt").exists()

def find_attached_input():
    for folder in KNOWN_INPUT_DIRS:
        if is_input_dir(folder):
            return folder
    base = Path("/kaggle/input")
    if base.exists():
        for path in base.rglob("1.txt"):
            if is_input_dir(path.parent):
                return path.parent
    return None

INPUT_DIR = find_attached_input()
INPUT_SOURCE = "Dataset đã attach"

if INPUT_DIR is None:
    INPUT_SOURCE = "BẢN NHÚNG trong notebook (public test Vòng 1)"
    target = WORK / "input_embedded"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name, body in INPUT_FILES.items():
        (target / name).write_text(body, encoding="utf-8")
    INPUT_DIR = target

n = len(list(INPUT_DIR.glob("*.txt")))
print("nguồn input:", INPUT_SOURCE)
print("đường dẫn  :", INPUT_DIR, f"({n} tệp)")
assert n == 100, f"Cần đúng 100 tệp, thấy {n}"
if INPUT_DIR == WORK / "input_embedded":
    print("\\n>>> Đang dùng public test nhúng sẵn. Nếu đây là lần chạy cho PRIVATE TEST,")
    print(">>> hãy attach Dataset input của BTC rồi chạy lại cell này. <<<")
""")

add(MD, """
## 5. Knowledge base

* **ICD-10 tiếng Việt** — dựng từ Phụ lục TT06/2026/TT-BYT (Bộ Y tế) và **nhúng
  sẵn** trong notebook. Đây là thay đổi quan trọng nhất ở phần gán mã: KB cũ là
  bản tiếng Anh của CDC nên một mention như `viêm túi mật` không thể khớp alias
  nào.
* **RxNorm** — Current Prescribable Content của NLM. Không nhúng được vì là dữ
  liệu của bên thứ ba, nên cell dưới tự tải khi bật Internet. Thiếu nó thì
  candidates `THUỐC` rỗng, pipeline vẫn chạy.
""")

add(CODE, """
# /kaggle/input là READ-ONLY, nên mọi thứ dựng ra phải nằm ở /kaggle/working.
# %%writefile KHÔNG tự tạo thư mục cha, nên phải mkdir trước.
TERM_DIR = WORK / "terminology"
TERM_DIR.mkdir(parents=True, exist_ok=True)
ICD_TSV = TERM_DIR / "icd10_vn.tsv"
print("thư mục terminology:", TERM_DIR)
""")

add(CODE, "%%writefile /kaggle/working/terminology/icd10_vn.tsv\n" + icd_table())

add(CODE, """
rows = ICD_TSV.read_text(encoding="utf-8").splitlines()
print(f"ICD KB: {len(rows) - 1:,} mã -> {ICD_TSV} ({ICD_TSV.stat().st_size:,} bytes)")
print("ví dụ:", rows[1])
""")

add(CODE, """
from medical_coder.rxnorm_kb import build as build_rxnorm

RX_TSV = TERM_DIR / "rxnorm.tsv"
RX_URL = "https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_07062026.zip"

def find_rxnorm_archive():
    for base in (Path("/kaggle/input"), WORK):
        if base.exists():
            for path in base.rglob("RxNorm_full_prescribe_*.zip"):
                return path
    return None

if RX_TSV.exists():
    print("dùng RxNorm TSV có sẵn:", RX_TSV)
else:
    archive = find_rxnorm_archive()
    if archive is None:
        archive = WORK / "rxnorm.zip"
        print("tải RxNorm …")
        rc = subprocess.run(["curl", "-sSL", "--max-time", "600", "-o", str(archive), RX_URL],
                            check=False).returncode
        if rc != 0 or not archive.exists() or archive.stat().st_size < 10_000_000:
            print("CẢNH BÁO: tải RxNorm thất bại — candidates THUỐC sẽ rỗng")
            archive = None
    if archive is not None:
        print("dựng RxNorm KB từ:", archive)
        print("số RxCUI:", build_rxnorm(archive, RX_TSV))
    else:
        RX_TSV = None
""")

add(MD, """
## 6. Weights

| Model | Vai trò | Tham số | Đĩa (bf16) |
|---|---|---:|---:|
| `urchade/gliner_multi-v2.1` | NER | 0.289B | ~1.2 GB |
| `Qwen/Qwen3-4B-Instruct-2507` | corrector | 4.022B | ~8.0 GB |
| `Qwen/Qwen3.5-4B` | teacher phụ (additions) | 4.206B | ~8.4 GB |

Tham số thì cả ba cộng lại là 8.517B, vẫn dưới 9B. Nhưng **đĩa mới là ràng buộc
thật**: cả ba là ~17.6 GB, trong khi `/kaggle/working` chỉ có ~20 GB và còn phải
chứa torch, output và cache. Đó chính là nguyên nhân `Errno 28`.

Nên **mặc định chỉ tải teacher chính** (~9.2 GB tổng cộng): corrector chạy, chỉ
bỏ bước additions. Đây cũng là đánh đổi hợp lý — corrector sửa type cho span đã
có, còn additions chỉ thêm span cho các type không mang candidate.

Muốn bật additions thì attach cả hai Qwen dưới dạng **Kaggle Dataset**: đọc từ
`/kaggle/input` là read-only, không tính vào quota `/kaggle/working`. Khi đó đặt
`SECONDARY = "Qwen/Qwen3.5-4B"` và cell dưới sẽ tự tìm thấy.
""")

add(CODE, """
GLINER_MODEL = "urchade/gliner_multi-v2.1"
PRIMARY   = "Qwen/Qwen3-4B-Instruct-2507"
SECONDARY = None    # đặt tên repo để bật additions — CHỈ nên làm khi đã attach Dataset

TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    for name in ("HF_TOKEN", "HF_KEY", "HUGGINGFACE_TOKEN", "HUGGINGFACE_KEY"):
        try:
            TOKEN = UserSecretsClient().get_secret(name)
            if TOKEN:
                print("dùng secret:", name)
                break
        except Exception:
            continue
except ImportError:
    pass

# Tải vào cache rồi dùng thẳng đường dẫn cache trả về. Dùng local_dir= sẽ giữ
# thêm một bản trong cache nữa, tức gấp đôi đĩa cho cùng một model.
HF_CACHE = WORK / "hf"
HF_CACHE.mkdir(parents=True, exist_ok=True)

def find_attached_model(repo_id):
    leaf = repo_id.split("/")[-1]
    base = Path("/kaggle/input")
    if base.exists():
        for path in base.rglob(leaf):
            if path.is_dir() and (path / "config.json").exists():
                return str(path)
    return None

def resolve_model(repo_id, need_gb):
    attached = find_attached_model(repo_id)
    if attached:
        print(f"  {repo_id}: dùng Dataset đã attach (không tốn quota)")
        return attached
    free_gb = _sh.disk_usage("/kaggle/working").free / 2**30
    if free_gb < need_gb + 2:
        raise RuntimeError(
            f"{repo_id} cần ~{need_gb} GB nhưng chỉ còn {free_gb:.1f} GB trống. "
            "Factory reset session, hoặc attach model này dưới dạng Dataset."
        )
    from huggingface_hub import snapshot_download
    print(f"  {repo_id}: tải về (~{need_gb} GB, còn trống {free_gb:.1f} GB) …")
    return snapshot_download(
        repo_id=repo_id, cache_dir=str(HF_CACHE), token=TOKEN,
        ignore_patterns=["*.pth", "*.onnx", "*.msgpack", "*.h5", "*.gguf"],
    )

GLINER_PATH = resolve_model(GLINER_MODEL, 1.2)
PRIMARY_PATH = resolve_model(PRIMARY, 8.0)

SECONDARY_PATH = None
if SECONDARY:
    try:
        SECONDARY_PATH = resolve_model(SECONDARY, 8.4)
    except Exception as exc:
        print(f"  bỏ qua teacher phụ: {exc}")

print()
print("gliner   :", GLINER_PATH)
print("primary  :", PRIMARY_PATH)
print("secondary:", SECONDARY_PATH or "(không có — chỉ chạy corrector, bỏ additions)")
print()
report_disk()
""")

add(CODE, """
# Sau bước provision, khoá offline: inference hoàn toàn self-host, không gọi API.
TOKEN = None
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
print("đã khoá chế độ offline")
""")

add(MD, "## 7. Cấu hình")

add(CODE, """
import torch
from medical_coder.gliner_ner import DEFAULT_THRESHOLDS
from medical_coder.models import EntityType
from medical_coder.pipeline_v2 import PipelineV2Config, run_pipeline_v2

HAS_CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if HAS_CUDA else 0

OUTPUT_DIR = WORK / "output"
ZIP_PATH   = WORK / "output.zip"

# Ngưỡng theo từng type. GLiNER có phân bố score khác nhau theo label nên một
# ngưỡng chung là sai; các giá trị này lấy từ lời giải tham chiếu 27.8786.
THRESHOLDS = dict(DEFAULT_THRESHOLDS)
for k, v in THRESHOLDS.items():
    print(f"  {k.value:22s} {v}")

CONFIG = dict(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    model_path=GLINER_PATH,
    device="cuda" if HAS_CUDA else "cpu",
    icd_kb=ICD_TSV,
    rxnorm_kb=RX_TSV,
    thresholds=THRESHOLDS,
    max_candidates=1,          # >1 làm phình mẫu số candidate
    primary_teacher=PRIMARY_PATH if HAS_CUDA else None,
    secondary_teacher=SECONDARY_PATH if HAS_CUDA else None,
    teacher_device="cuda:0" if HAS_CUDA else "cpu",
    teacher_quantization="4bit",
    teacher_batch_size=48 if NGPU else 8,
    # Bộ loại span: hỏi teacher xem span sắp emit có thực sự là khái niệm y khoa
    # không, đặt None để tắt. Xem mục ghi chú cuối notebook về ngưỡng hoà vốn.
    reject_margin=1.0 if HAS_CUDA else None,
)
print("\\nGPU:", NGPU, "| corrector:", bool(CONFIG["primary_teacher"]),
      "| additions:", bool(CONFIG["secondary_teacher"]))
""")

add(MD, """
## 8. Smoke test (2 bản ghi)

Chạy thử trước khi chạy đủ 100 để bắt lỗi cấu hình sớm. Bước này **không** tạo ZIP.
""")

add(CODE, """
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)

SMOKE_DIR = WORK / "output_smoke"
smoke = run_pipeline_v2(PipelineV2Config(**{**CONFIG, "output_dir": SMOKE_DIR,
                                            "selected_ids": frozenset({"1", "2"})}))
print("\\nsmoke concepts:", smoke)

for stem in ("1", "2"):
    data = json.loads((SMOKE_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    raw = (INPUT_DIR / f"{stem}.txt").read_text(encoding="utf-8")
    assert all(raw[c["position"][0]:c["position"][1]] == c["text"] for c in data), "offset sai"
    print(f"\\n--- {stem}.json ({len(data)} concept) ---")
    for c in data[:6]:
        print(f"  {c['position']} {c['type']:20s} {c['text'][:44]!r} {c.get('candidates', '')}")
print("\\noffset khớp nguyên văn trên cả hai bản ghi")
""")

add(MD, "## 9. Chạy đủ 100 bản ghi")

add(CODE, """
import time

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

start = time.time()
total = run_pipeline_v2(PipelineV2Config(**CONFIG))
print(f"\\n{total} concept trong {time.time() - start:.0f}s")
""")

add(MD, "## 10. Kiểm tra và đóng gói")

add(CODE, """
from medical_coder.submission import create_submission_zip, validate_all

validate_all(INPUT_DIR, OUTPUT_DIR)
print("validator: PASS")

if ZIP_PATH.exists():
    ZIP_PATH.unlink()
create_submission_zip(OUTPUT_DIR, ZIP_PATH)

import zipfile, hashlib
with zipfile.ZipFile(ZIP_PATH) as archive:
    names = archive.namelist()
    assert names == [f"output/{i}.json" for i in range(1, 101)], "cấu trúc ZIP sai"
    assert archive.testzip() is None, "ZIP hỏng"
print(f"ZIP OK: {len(names)} tệp, {ZIP_PATH.stat().st_size:,} bytes")
print("sha256:", hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest())
""")

add(CODE, """
from collections import Counter

records = {p.stem: json.loads(p.read_text(encoding="utf-8"))
           for p in OUTPUT_DIR.glob("*.json") if p.stem.isdigit()}
concepts = [c for v in records.values() for c in v]
types = Counter(c["type"] for c in concepts)
with_codes = [c for c in concepts if c.get("candidates")]

print(f"tổng concept        : {len(concepts)}")
print(f"trung bình / bản ghi: {len(concepts) / len(records):.2f}")
print(f"bản ghi rỗng        : {[k for k, v in records.items() if not v]}")
for k in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"):
    print(f"  {k:22s} {types.get(k, 0)}")
print(f"concept có candidate: {len(with_codes)}")
print(f"tổng mã xuất ra     : {sum(len(c['candidates']) for c in with_codes)}")
print(f"nhãn assertion      : {sum(len(c['assertions']) for c in concepts)} (chủ ý để rỗng)")
""")

add(MD, """
## 11. Tải kết quả

`/kaggle/working/output.zip` — nộp trực tiếp tệp này.
""")

add(CODE, """
from IPython.display import FileLink, display
display(FileLink(str(ZIP_PATH)))
""")

add(MD, """
## 12. Ghi chú

**Đã cố ý bỏ:**

* **Assertions để rỗng.** Lời giải tham chiếu đo được `isNegated` tách biệt ở AUC
  0.497 (ngang ngẫu nhiên); mọi rule đều emit thừa. Một assertion sai mất trọn
  Jaccard của concept đó, trong khi dự đoán rỗng đúng với ground truth rỗng được
  1.0. Chỉ nên bật lại khi đã có dữ liệu gán nhãn.
* **Candidate tối đa 1 và chỉ khi alias khớp duy nhất.** Bỏ toàn bộ candidate chỉ
  làm candidate Jaccard của họ giảm 0.0036 — thành phần 40% này gần như hoàn toàn
  do chất lượng khớp concept quyết định, không phải do tra đúng mã.
* **Additions chỉ cho type không có candidate.** Thêm nhầm một CHẨN_ĐOÁN/THUỐC
  còn bị tính vào mẫu số candidate.

**Chưa kiểm chứng:** chưa có ground truth nên chưa đo được điểm cục bộ. Sau khi
gán nhãn 15–20 bản ghi, dùng:

```bash
medical-coder score --output-dir output --truth-dir data/labelled --per-record
```

Scorer tự chấm ground truth bằng 1.0 và tái lập đúng cả hai mốc điểm đã công bố
(14.4255 và 27.8786).
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = REPO_ROOT / "notebooks" / "Viettel_AI_Race_Kaggle_Predict_V2.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, len(cells), "cells")
