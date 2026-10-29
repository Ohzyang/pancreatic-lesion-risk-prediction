# -*- coding: utf-8 -*-
"""
按参考“可编辑 PDF”的同一逻辑拼接原生 PDF。
修改点：将原有的 a-j 改为 a, b, c, d, e，每一行代表一个实验（包含左右两张图），
只在每行最左侧标记一个代表该实验的字母。
"""

from pathlib import Path
import os
import fitz  # PyMuPDF


# ============================================================
# 1. 文件路径：一个实验（一行）对应一个字母，包含两张图
# ============================================================
BASE_DIR = Path(r".\\subfigure")

pdf_files = {
    "a": [
        BASE_DIR / "Prediction_horizons_AUC.pdf",
        BASE_DIR / "Prediction_horizons_RR.pdf"
    ],
    "b": [
        BASE_DIR / "Model_types_AUC.pdf",
        BASE_DIR / "Model_types_RR.pdf"
    ],
    "c": [
        BASE_DIR / "Feature_tiers_AUC.pdf",
        BASE_DIR / "Feature_tiers_RR.pdf"
    ],
    "d": [
        BASE_DIR / "Minimum_sequence_length_AUC.pdf",
        BASE_DIR / "Minimum_sequence_length_RR.pdf"
    ],
    "e": [
        BASE_DIR / "Missing_tolerance_AUC.pdf",
        BASE_DIR / "Missing_tolerance_RR.pdf"
    ]
}

row_titles = [
    "Prediction horizons",
    "Model types",
    "Feature tiers",
    "Minimum sequence length",
    "Missingness tolerance"
]


# ============================================================
# 2. 页面与版式参数：原始尺寸拼接，不缩放
# ============================================================

margin_left = 35
margin_right = 5
margin_y = 5

# 减少两列之间的空白
gap_x = 0

# 减少各行之间的空白
gap_y = 0

# 保持原始 PDF 尺寸
w_auc = 349.7
w_rr = 565.3
row_height = 348.5

row_count = len(pdf_files)

total_width = (
    margin_left
    + w_auc
    + gap_x
    + w_rr
    + margin_right
)

total_height = (
    margin_y * 2
    + row_count * row_height
    + (row_count - 1) * gap_y
)

output_path = Path(r".\figure\fig2.pdf")


# ============================================================
# 3. 检查源 PDF 是否真正包含矢量内容
# ============================================================
def inspect_source_pdf(path: Path) -> None:
    doc = fitz.open(str(path))
    try:
        if doc.page_count < 1:
            raise ValueError(f"PDF 没有页面：{path}")

        page = doc[0]
        image_count = len(page.get_images(full=True))
        drawing_count = len(page.get_drawings())
        word_count = len(page.get_text("words"))

        print(
            f"检查 {path.name}: "
            f"images={image_count}, "
            f"drawings={drawing_count}, "
            f"words={word_count}"
        )

        if image_count > 0:
            print("  [警告] 该 PDF 含图片对象；图片部分不能作为矢量路径编辑。")

        if drawing_count == 0:
            print("  [警告] 没有检测到矢量绘图路径，源 PDF 可能已被栅格化。")
    finally:
        doc.close()


# ============================================================
# 4. 主程序
# ============================================================
def main() -> None:
    # 提取标签 a, b, c, d, e
    labels = list(pdf_files.keys())

    # 一次性检查所有文件
    missing = []
    for paths in pdf_files.values():
        for path in paths:
            if not path.exists():
                missing.append(path)

    if missing:
        print("以下原生 PDF 不存在：")
        for path in missing:
            print(f"  - {path}")

        print(
            "\n请先从生成每张图的 Matplotlib 代码中直接导出 PDF，"
            "不要把 SVG 再转换成 PDF。"
        )
        raise FileNotFoundError("缺少输入 PDF。")

    # 检查源 PDF 的对象类型
    print("开始检查源 PDF：")
    for paths in pdf_files.values():
        for path in paths:
            inspect_source_pdf(path)

    out_pdf = fitz.open()
    page = out_pdf.new_page(
        width=total_width,
        height=total_height,
    )

    try:
        # 逐行遍历实验 (a, b, c, d, e)
        for row, label in enumerate(labels):
            paths = pdf_files[label]
            
            # 遍历该行的左右两张图 (col=0: AUC, col=1: RR)
            for col, file_path in enumerate(paths):
                is_left = (col == 0)
                src_doc = fitz.open(str(file_path))

                try:
                    # 计算横纵坐标
                    x0 = margin_left if is_left else margin_left + w_auc + gap_x
                    y0 = margin_y + row * (row_height + gap_y)
                    w_current = w_auc if is_left else w_rr

                    rect = fitz.Rect(
                        x0,
                        y0,
                        x0 + w_current,
                        y0 + row_height,
                    )
                    src_page = src_doc[0]

                    src_w = src_page.rect.width
                    src_h = src_page.rect.height

                    scale = min(
                        w_current / src_w,
                        row_height / src_h
                    )

                    print(
                        f"{file_path.name}: "
                        f"原始尺寸={src_w:.1f}x{src_h:.1f} pt, "
                        f"目标尺寸={w_current}x{row_height} pt, "
                        f"实际缩放={scale:.3f}, "
                        f"16pt字体合成后约={16 * scale:.1f}pt"
                    )
                    # 嵌入源 PDF 页面
                    page.show_pdf_page(
                        rect,
                        src_doc,
                        pno=0,
                        keep_proportion=True,
                        overlay=True,
                    )

                finally:
                    src_doc.close()

                # ======= 以下文字标记，只在每行【左侧图】绘制一次 =======
                if is_left:
                    # 1. 插入该行的字母标号 (a, b, c, d, e)
                    page.insert_text(
                        fitz.Point(x0 - 5, y0 + 18),
                        label,
                        fontname="hebo",
                        fontsize=28,
                        color=(0, 0, 0),
                        overlay=True,
                    )

                    # 2. 插入竖排实验标题
                    title = row_titles[row]
                    title_fontsize = 24
                    title_x = x0 - 15
                    estimated_text_width = len(title) * title_fontsize * 0.5
                    title_y = y0 + (row_height + estimated_text_width) / 2 - 15

                    page.insert_text(
                        fitz.Point(title_x, title_y),
                        title,
                        fontsize=title_fontsize,
                        fontname="helv",
                        color=(0, 0, 0),
                        rotate=90,
                        overlay=True,
                    )

                print(f"成功拼接图 {label} ({'左' if is_left else '右'}): {file_path.name}")

        out_pdf.save(str(output_path))

    finally:
        out_pdf.close()

    # 最终文件检查
    check_doc = fitz.open(str(output_path))
    try:
        check_page = check_doc[0]
        print("\n最终 PDF 检查：")
        print(
            f"images={len(check_page.get_images(full=True))}, "
            f"drawings={len(check_page.get_drawings())}, "
            f"words={len(check_page.get_text('words'))}, "
            f"xobjects={len(check_page.get_xobjects())}"
        )
    finally:
        check_doc.close()

    print(f"\n拼接完成：{output_path}")


if __name__ == "__main__":
    main()