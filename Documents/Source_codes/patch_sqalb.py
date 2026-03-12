"""
SQALB integration patch for lwdetr.py

Usage:
    python patch_sqalb.py

Applies:
1. SQALB import 추가
2. SetCriterion.__init__에 self.sqalb 등록
3. loss_boxes에 SQALB 분기 추가 (원본 L1 유지, SQALB 활성 시 NWD/WIoU)
4. build_criterion_and_postprocessors에서 criterion.sqalb 연결

NOTE: 실제 코드는 "return criterion, postprocess" (s 없음)
"""

import os

PATH = os.path.join("rf-detr", "src", "rfdetr", "models", "lwdetr.py")


def patch():
    with open(PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    content = "".join(lines)

    # --- 1. Import ---
    import_line = "from rfdetr.models.sqalb import SQALB\n"
    if "from rfdetr.models.sqalb" in content:
        print("[import] already exists, skip")
    else:
        for i, l in enumerate(lines):
            if "from rfdetr.models.saqg" in l:
                lines.insert(i + 1, "# [v2.0.0] VIVID-Det SQALB\n")
                lines.insert(i + 2, import_line)
                print(f"[import] OK: inserted at line {i + 2}")
                break

    # --- 2. SetCriterion.__init__: self.sqalb 등록 ---
    content = "".join(lines)
    if "self.sqalb" in content:
        print("[__init__] already exists, skip")
    else:
        for i, l in enumerate(lines):
            if "self.use_position_supervised_loss = use_position_supervised_loss" in l:
                indent = "        "
                insert = [
                    f"{indent}# [v2.0.0] VIVID-Det SQALB\n",
                    f"{indent}self.sqalb = None  # set externally after build\n",
                ]
                lines[i + 1:i + 1] = insert
                print(f"[__init__] OK: self.sqalb added after line {i + 1}")
                break

    # --- 3. loss_boxes: SQALB 분기 추가 ---
    content = "".join(lines)
    if "# [v2.0.0] SQALB" in content:
        print("[loss_boxes] already patched, skip")
    else:
        old_block = (
            '        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")\n'
            '\n'
            '        losses = {}\n'
            '        losses["loss_bbox"] = loss_bbox.sum() / num_boxes'
        )
        new_block = (
            '        # [v2.0.0] SQALB: adaptive NWD/WIoU balancing\n'
            '        if self.sqalb is not None and self.training:\n'
            '            # Get confidence from pred_logits if available\n'
            '            if "pred_logits" in outputs:\n'
            '                _logits = outputs["pred_logits"][idx]\n'
            '                _conf = _logits.sigmoid().max(dim=-1).values\n'
            '            else:\n'
            '                _conf = torch.zeros(src_boxes.shape[0], device=src_boxes.device)\n'
            '            # IoU quality\n'
            '            with torch.no_grad():\n'
            '                _iou = torch.diag(box_ops.generalized_box_iou(\n'
            '                    box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),\n'
            '                    box_ops.box_cxcywh_to_xyxy(target_boxes),\n'
            '                )).clamp(0, 1)\n'
            '            # layer_idx from aux loss counter (default 0)\n'
            '            _layer_idx = getattr(self, "_current_layer_idx", 0)\n'
            '            loss_sqalb, _alpha_mean = self.sqalb(\n'
            '                src_boxes, target_boxes, _conf, _iou, _layer_idx\n'
            '            )\n'
            '            losses = {}\n'
            '            losses["loss_bbox"] = loss_sqalb.sum() / num_boxes\n'
            '        else:\n'
            '            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")\n'
            '            losses = {}\n'
            '            losses["loss_bbox"] = loss_bbox.sum() / num_boxes'
        )

        content = "".join(lines)
        if old_block in content:
            content = content.replace(old_block, new_block, 1)
            lines = content.splitlines(True)
            print("[loss_boxes] OK: SQALB branch added")
        else:
            print("[loss_boxes] WARNING: exact match not found, trying line-by-line")
            for i, l in enumerate(lines):
                if 'loss_bbox = F.l1_loss(src_boxes, target_boxes' in l:
                    indent = "        "
                    insert = [
                        f"{indent}# [v2.0.0] SQALB: adaptive NWD/WIoU balancing\n",
                        f"{indent}if self.sqalb is not None and self.training:\n",
                        f"{indent}    if 'pred_logits' in outputs:\n",
                        f"{indent}        _logits = outputs['pred_logits'][idx]\n",
                        f"{indent}        _conf = _logits.sigmoid().max(dim=-1).values\n",
                        f"{indent}    else:\n",
                        f"{indent}        _conf = torch.zeros(src_boxes.shape[0], device=src_boxes.device)\n",
                        f"{indent}    with torch.no_grad():\n",
                        f"{indent}        _iou = torch.diag(box_ops.generalized_box_iou(\n",
                        f"{indent}            box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),\n",
                        f"{indent}            box_ops.box_cxcywh_to_xyxy(target_boxes),\n",
                        f"{indent}        )).clamp(0, 1)\n",
                        f"{indent}    _layer_idx = getattr(self, '_current_layer_idx', 0)\n",
                        f"{indent}    loss_sqalb, _alpha_mean = self.sqalb(\n",
                        f"{indent}        src_boxes, target_boxes, _conf, _iou, _layer_idx\n",
                        f"{indent}    )\n",
                        f"{indent}    losses = {{}}\n",
                        f"{indent}    losses['loss_bbox'] = loss_sqalb.sum() / num_boxes\n",
                        f"{indent}else:\n",
                        f"    ",
                    ]
                    lines[i:i] = insert
                    print(f"[loss_boxes] OK: fallback insert at line {i}")
                    break

    # --- 4. build_criterion에서 sqalb 연결 ---
    # NOTE: 실제 반환문은 "return criterion, postprocess" (s 없음)
    content = "".join(lines)
    if "criterion.sqalb" in content:
        print("[build] already patched, skip")
    else:
        for i, l in enumerate(lines):
            if "return criterion, postprocess" in l and "def " not in l:
                indent = "    "
                insert = [
                    f"\n",
                    f"{indent}# [v2.0.0] VIVID-Det SQALB\n",
                    f"{indent}if getattr(args, 'ssfa_enabled', False):\n",
                    f'{indent}    criterion.sqalb = SQALB(num_decoder_layers=args.dec_layers)\n',
                    f"\n",
                ]
                lines[i:i] = insert
                print(f"[build] OK: criterion.sqalb wiring added before line {i + 1}")
                break

    with open(PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    patch()
    print("\nDone. Verify with:")
    print('  python -c "from rfdetr.models.lwdetr import build_criterion_and_postprocessors; '
          'from rfdetr import RFDETRMedium; m = RFDETRMedium(ssfa_enabled=True); '
          'c, _ = build_criterion_and_postprocessors(m.model.args); '
          'print(type(c.sqalb).__name__)"')