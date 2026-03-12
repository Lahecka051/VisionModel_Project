"""
SAQG integration patch for lwdetr.py and transformer.py

Usage:
    python patch_saqg.py

Applies:
1. transformer.py: refpoint_embed batch-expand에 dim 체크 추가
2. lwdetr.py: SAQG import, __init__ 등록, forward 호출
"""

import os

BASE = os.path.join("rf-detr", "src", "rfdetr", "models")


def patch_transformer():
    """transformer.py: refpoint_embed expand에 dim 체크 추가.

    원본:
        refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)

    변경:
        # [v2.0.0] SAQG: already [B,N,4] if SAQG active
        if refpoint_embed.dim() == 2:
            refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)
    """
    path = os.path.join(BASE, "transformer.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old = "            refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)"
    new = (
        "            # [v2.0.0] SAQG: skip expand if already [B,N,4]\n"
        "            if refpoint_embed.dim() == 2:\n"
        "                refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)"
    )

    if old not in content:
        if "# [v2.0.0] SAQG" in content:
            print("[transformer.py] already patched, skip")
            return
        print("[transformer.py] ERROR: target string not found")
        return

    content = content.replace(old, new, 1)  # 1회만 교체
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[transformer.py] OK: dim check added")


def patch_lwdetr():
    """lwdetr.py:
    1. SAQG import 추가
    2. LWDETR.__init__에 self.saqg 등록
    3. LWDETR.forward에서 SAQG 호출
    """
    path = os.path.join(BASE, "lwdetr.py")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # --- 1. Import 추가 (SSFA import 아래 또는 파일 상단 import 영역) ---
    import_line = "from rfdetr.models.saqg import SAQG\n"
    if import_line.strip() in "".join(lines):
        print("[lwdetr.py] SAQG import already exists, skip import")
    else:
        # rfdetr.models.ssfa import 뒤에 추가하거나, backbone import 뒤에
        insert_idx = None
        for i, l in enumerate(lines):
            if "from rfdetr.models" in l and "import" in l:
                insert_idx = i + 1  # 마지막 models import 뒤
        if insert_idx is None:
            # fallback: 첫 import 블록 끝
            for i, l in enumerate(lines):
                if l.startswith("import ") or l.startswith("from "):
                    insert_idx = i + 1
        lines.insert(insert_idx, "# [v2.0.0] VIVID-Det SAQG\n")
        lines.insert(insert_idx + 1, import_line)
        print(f"[lwdetr.py] OK: import added at line {insert_idx + 1}")

    # --- 2. __init__에 self.saqg 추가 ---
    # self.backbone = backbone 다음에 추가
    if "self.saqg" in "".join(lines):
        print("[lwdetr.py] self.saqg already exists, skip __init__")
    else:
        for i, l in enumerate(lines):
            if "self.backbone = backbone" in l:
                indent = "        "
                insert = [
                    f"{indent}# [v2.0.0] VIVID-Det SAQG\n",
                    f"{indent}self.saqg = SAQG() if getattr(backbone[0], 'ssfa_enabled', False) else None\n",
                ]
                lines[i + 1:i + 1] = insert
                print(f"[lwdetr.py] OK: self.saqg added after line {i + 1}")
                break

    # --- 3. forward에서 SAQG 호출 추가 ---
    # transformer 호출 직전, refpoint_embed_weight 준비 후
    # 학습/추론 분기 이후에 삽입
    if "self.saqg(" in "".join(lines):
        print("[lwdetr.py] SAQG forward call already exists, skip")
    else:
        # "hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(" 앞에 삽입
        for i, l in enumerate(lines):
            if "hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(" in l:
                indent = "            " if l.startswith("            ") else "        "
                insert = [
                    f"\n",
                    f"{indent}# [v2.0.0] VIVID-Det SAQG: SOPM-based query rearrangement\n",
                    f"{indent}if self.saqg is not None:\n",
                    f"{indent}    _sopm = self.backbone[0].sopm  # cached from SSFA\n",
                    f"{indent}    _bs = srcs[0].shape[0]\n",
                    f"{indent}    refpoint_embed_weight = self.saqg(refpoint_embed_weight, _sopm, _bs)\n",
                    f"\n",
                ]
                lines[i:i] = insert
                print(f"[lwdetr.py] OK: SAQG forward call added before line {i + 1}")
                break

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    patch_transformer()
    patch_lwdetr()
    print("\nDone. Verify with:")
    print('  python -c "from rfdetr import RFDETRMedium; m = RFDETRMedium(ssfa_enabled=True); print(type(m.model.model.saqg).__name__)"')
