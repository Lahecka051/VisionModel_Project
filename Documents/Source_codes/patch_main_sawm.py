"""
SAWM 적용: main.py 자동 패치 스크립트

사용법:
  python patch_main_sawm.py

대상: C:\gitnconda\Swin-Transformer\rf-detr\src\rfdetr\main.py
동작: 4곳에 set_cost_nwd, nwd_C 파라미터를 삽입한다.
      원본은 main.py.bak으로 백업된다.
"""

import shutil
from pathlib import Path

MAIN_PY = Path(r"C:\gitnconda\Swin-Transformer\rf-detr\src\rfdetr\main.py")
BACKUP = MAIN_PY.with_suffix(".py.bak")


def patch():
    if not MAIN_PY.exists():
        print(f"[ERROR] {MAIN_PY} not found")
        return

    # 백업
    shutil.copy2(MAIN_PY, BACKUP)
    print(f"[BACKUP] {BACKUP}")

    text = MAIN_PY.read_text(encoding="utf-8")
    count = 0

    # ── 패치 1: 허용 파라미터 목록 (line ~792) ──
    anchor = '            "set_cost_giou",\n'
    insert = '            "set_cost_nwd",\n            "nwd_C",\n'
    if anchor in text and '"set_cost_nwd"' not in text:
        text = text.replace(anchor, anchor + insert, 1)
        count += 1
        print("[PATCH 1] allowed params list: set_cost_nwd, nwd_C added")

    # ── 패치 2: argparse (line ~946) ──
    anchor2 = '    parser.add_argument("--set_cost_giou", default=2, type=float, help="giou box coefficient in the matching cost")\n'
    insert2 = (
        '    parser.add_argument("--set_cost_nwd", default=0, type=float, help="NWD coefficient in the matching cost (SAWM)")\n'
        '    parser.add_argument("--nwd_C", default=0.5, type=float, help="NWD normalization constant")\n'
    )
    if anchor2 in text and '"--set_cost_nwd"' not in text:
        text = text.replace(anchor2, anchor2 + insert2, 1)
        count += 1
        print("[PATCH 2] argparse: --set_cost_nwd, --nwd_C added")

    # ── 패치 3: 함수 기본값 (line ~1127) ──
    anchor3 = "    set_cost_giou=2,\n"
    insert3 = "    set_cost_nwd=0,\n    nwd_C=0.5,\n"
    if anchor3 in text and "set_cost_nwd=0," not in text:
        text = text.replace(anchor3, anchor3 + insert3, 1)
        count += 1
        print("[PATCH 3] function defaults: set_cost_nwd=0, nwd_C=0.5 added")

    # ── 패치 4: args 전달 (line ~1239) ──
    anchor4 = "        set_cost_giou=set_cost_giou,\n"
    insert4 = "        set_cost_nwd=set_cost_nwd,\n        nwd_C=nwd_C,\n"
    if anchor4 in text and "set_cost_nwd=set_cost_nwd," not in text:
        text = text.replace(anchor4, anchor4 + insert4, 1)
        count += 1
        print("[PATCH 4] args forwarding: set_cost_nwd, nwd_C added")

    if count == 4:
        MAIN_PY.write_text(text, encoding="utf-8")
        print(f"\n[SUCCESS] {count}/4 patches applied to {MAIN_PY}")
    elif count > 0:
        MAIN_PY.write_text(text, encoding="utf-8")
        print(f"\n[WARNING] {count}/4 patches applied. Check manually.")
    else:
        print("\n[SKIP] All patches already applied or anchors not found.")


if __name__ == "__main__":
    patch()
