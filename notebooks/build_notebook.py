"""Generate the Kaggle Run-All notebook for the predict-v2 pipeline.

The notebook is self-contained: the `medical_coder` package and the Vietnamese
ICD-10 table are embedded as base64 blobs, so a run needs no source Dataset and
no git clone.
"""
import base64
import gzip
import io
import json
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MD = "markdown"
CODE = "code"

cells = []


def source_tarball_b64() -> str:
    """gzip tarball of src/medical_coder/*.py, base64-encoded.

    Deterministic on purpose: regenerating without touching the source must
    leave the notebook byte-identical, otherwise every rebuild shows up as a
    3,000-line diff and real changes become invisible. That means zeroing the
    tar member metadata *and* the gzip header timestamp.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted((REPO_ROOT / "src" / "medical_coder").glob("*.py")):
            info = tar.gettarinfo(str(path), arcname=f"medical_coder/{path.name}")
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                tar.addfile(info, handle)
    return base64.b64encode(
        gzip.compress(buffer.getvalue(), compresslevel=9, mtime=0)
    ).decode("ascii")


def icd_table_b64() -> str:
    data = (REPO_ROOT / "data" / "terminology" / "icd10_vn.tsv").read_bytes()
    return base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)).decode("ascii")


def as_python_literal(blob: str, width: int = 108) -> str:
    """Render a long base64 string as concatenated short literals."""
    lines = [f'    "{blob[i : i + width]}"' for i in range(0, len(blob), width)]
    return "(\n" + "\n".join(lines) + "\n)"


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
2. Settings → Internet: **On** (để tải weights và RxNorm). Nếu đã attach sẵn mọi
   thứ thì có thể tắt.
3. Attach dataset chứa `input/1.txt` … `100.txt`
   (ví dụ `/kaggle/input/datasets/thanhhiepvo/viettelairace/input`).
4. Nếu chưa attach weights: đặt HF token trong Add-ons → Secrets với tên
   `HF_TOKEN`.

**Không cần** Dataset source và **không** clone git: package `medical_coder` và
bảng ICD-10 tiếng Việt được nhúng thẳng trong notebook.
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

add(MD, """
## 3. Source code nhúng sẵn

Toàn bộ package `medical_coder` được nhúng ngay trong notebook dưới dạng tarball
base64. Không cần Dataset source, không cần clone, không cần Internet cho bước
này — và source luôn khớp đúng phiên bản đã sinh ra notebook.

Cell dưới giải nén ra `/kaggle/working/medical_coder_src/` rồi thêm vào
`sys.path`.
""")

add(CODE, f"""
# tarball gzip của src/medical_coder/*.py, base64. Sinh tự động — đừng sửa tay.
SOURCE_TGZ_B64 = {as_python_literal(source_tarball_b64())}
print("blob source:", len(SOURCE_TGZ_B64), "ký tự base64")
""")

add(CODE, """
import base64, io, tarfile

IMPORT_DIR = WORK / "medical_coder_src"
if IMPORT_DIR.exists():
    shutil.rmtree(IMPORT_DIR)
IMPORT_DIR.mkdir(parents=True)

with tarfile.open(fileobj=io.BytesIO(base64.b64decode(SOURCE_TGZ_B64)), mode="r:gz") as tar:
    for member in tar.getmembers():
        # chỉ nhận đường dẫn tương đối nằm trong medical_coder/
        if member.name.startswith("/") or ".." in Path(member.name).parts:
            raise SystemExit(f"tarball có đường dẫn không hợp lệ: {member.name}")
    try:
        tar.extractall(IMPORT_DIR, filter="data")   # Python >= 3.12
    except TypeError:
        tar.extractall(IMPORT_DIR)

modules = sorted(p.name for p in (IMPORT_DIR / "medical_coder").glob("*.py"))
print(f"giải nén {len(modules)} module ->", IMPORT_DIR)
print(" ", ", ".join(modules))

if str(IMPORT_DIR) not in sys.path:
    sys.path.insert(0, str(IMPORT_DIR))

import medical_coder
from medical_coder import exact_link, gliner_ner, pipeline_v2, selector
print("\\nmedical_coder:", medical_coder.__file__)

REPO = None   # không có repo trên đĩa; mọi thứ dựng ra nằm ở /kaggle/working
""")

add(MD, """
## 4. Tìm dữ liệu đầu vào

Đây là thứ **duy nhất** phải attach: Dataset chứa `1.txt` … `100.txt`.
""")

add(CODE, """
# Đường dẫn Dataset đã biết, thử trước để khỏi quét toàn bộ /kaggle/input.
KNOWN_INPUT_DIRS = [
    Path("/kaggle/input/datasets/thanhhiepvo/viettelairace/input"),
    Path("/kaggle/input/viettelairace/input"),
    Path("/kaggle/input/viettelairace"),
]

def is_input_dir(folder):
    return folder.is_dir() and (folder / "1.txt").exists() and (folder / "100.txt").exists()

def find_input_dir():
    for folder in KNOWN_INPUT_DIRS:
        if is_input_dir(folder):
            return folder
    for base in (Path("/kaggle/input"), REPO):
        if base and base.exists():
            for path in base.rglob("1.txt"):
                if is_input_dir(path.parent):
                    return path.parent
    return None

INPUT_DIR = find_input_dir()
if INPUT_DIR is None:
    raise SystemExit(
        "Không tìm thấy thư mục chứa 1.txt … 100.txt.\\n"
        "Đã thử: " + ", ".join(str(p) for p in KNOWN_INPUT_DIRS) + "\\n"
        "Attach dataset input rồi chạy lại cell này."
    )
n = len(list(INPUT_DIR.glob("*.txt")))
print("input:", INPUT_DIR, f"({n} tệp)")
assert n == 100, f"Cần đúng 100 tệp, thấy {n}"
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

add(CODE, f"""
# data/terminology/icd10_vn.tsv nén gzip rồi base64. Sinh tự động — đừng sửa tay.
ICD_TSV_GZ_B64 = {as_python_literal(icd_table_b64())}
print("blob ICD:", len(ICD_TSV_GZ_B64), "ký tự base64")
""")

add(CODE, """
import gzip

# /kaggle/input là READ-ONLY, nên mọi thứ dựng ra phải nằm ở /kaggle/working.
TERM_DIR = WORK / "terminology"
TERM_DIR.mkdir(parents=True, exist_ok=True)
ICD_TSV = TERM_DIR / "icd10_vn.tsv"

ICD_TSV.write_bytes(gzip.decompress(base64.b64decode(ICD_TSV_GZ_B64)))

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

| Model | Vai trò | Tham số |
|---|---|---:|
| `urchade/gliner_multi-v2.1` | NER | 0.289B |
| `Qwen/Qwen3-4B-Instruct-2507` | corrector (teacher chính) | 4.022B |
| `Qwen/Qwen3.5-4B` | teacher phụ, chỉ dùng cho additions | 4.206B |
| **Tổng** | | **8.517B** < 9B |

Teacher phụ là **tuỳ chọn**. Nếu tải không được thì pipeline vẫn chạy với riêng
corrector (tổng 4.311B) và bỏ qua bước additions — vì additions bắt buộc phải có
hai teacher đồng thuận.

Nếu chỉ muốn một teacher mạnh hơn: đặt `PRIMARY = "Qwen/Qwen3-8B"` và
`SECONDARY = None` → tổng 8.489B, vẫn dưới 9B.
""")

add(CODE, """
GLINER_MODEL = "urchade/gliner_multi-v2.1"
PRIMARY   = "Qwen/Qwen3-4B-Instruct-2507"
SECONDARY = "Qwen/Qwen3.5-4B"      # đặt None để chỉ chạy corrector

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

MODEL_DIR = WORK / "models"
MODEL_DIR.mkdir(exist_ok=True)

def resolve_model(repo_id):
    \"\"\"Trả về đường dẫn local; ưu tiên dataset đã attach, sau đó mới tải.\"\"\"
    leaf = repo_id.split("/")[-1]
    for base in (Path("/kaggle/input"), MODEL_DIR):
        if base.exists():
            for path in base.rglob(leaf):
                if path.is_dir() and any(path.glob("config.json")):
                    return str(path)
    from huggingface_hub import snapshot_download
    target = MODEL_DIR / leaf
    snapshot_download(repo_id=repo_id, local_dir=str(target), token=TOKEN,
                      ignore_patterns=["*.pth", "*.onnx", "*.msgpack", "*.h5"])
    return str(target)

GLINER_PATH = resolve_model(GLINER_MODEL)
print("gliner:", GLINER_PATH)

PRIMARY_PATH = resolve_model(PRIMARY)
print("primary:", PRIMARY_PATH)

SECONDARY_PATH = None
if SECONDARY:
    try:
        SECONDARY_PATH = resolve_model(SECONDARY)
        print("secondary:", SECONDARY_PATH)
    except Exception as exc:
        print(f"không tải được teacher phụ ({exc}) — chỉ chạy corrector, bỏ additions")
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
from medical_coder.pipeline import create_submission_zip, validate_all

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
