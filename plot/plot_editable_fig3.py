
# -*- coding: utf-8 -*-
"""
Figure 2-2 原生矢量 PDF 拼接

特点：
1. 所有 PDF 按 1:1 原始尺寸拼接，不缩放
2. 自动读取 PDF 的实际宽度和高度
3. 减少图与图之间的空白
4. 左侧竖排标题默认 24 pt
5. 保留 a、b、c 三组实验
6. 不栅格化，保留原始 PDF 的文字和矢量内容

布局：
a:
    Exclusion Interval        -> exclude.pdf
    Data exclusion result     -> AUC + RR

b:
    Independent Hold-out Datasets -> AUC + RR

c:
    Broader Outcome Definitions   -> AUC + RR
"""

from pathlib import Path
import os
import fitz


# ============================================================
# 1. 文件路径
# ============================================================

# 脚本放在 plot_1 文件夹中即可自动定位
BASE_DIR = Path(r".\\subfigure")

exclude_pdf = BASE_DIR / "exclude.pdf"

pdf_files = {
    "a": [
        BASE_DIR / "Data_exclusion_result_AUC.pdf",
        BASE_DIR / "Data_exclusion_result_RR.pdf"
    ],
    "b": [
        BASE_DIR / "Independent_hold-out_datasets_AUC.pdf",
        BASE_DIR / "Independent_hold-out_datasets_RR.pdf"
    ],
    "c": [
        BASE_DIR / "Broader_outcome_definitions_AUC.pdf",
        BASE_DIR / "Broader_outcome_definitions_RR.pdf"
    ]
}

row_titles = {
    "b": "Independent hold-out datasets",
    "c": "Broader outcome definitions"
}

output_path = Path(r".\figure\fig3.pdf")


# ============================================================
# 2. 页面与版式参数
# ============================================================

# 给左侧字母和竖排标题留空间
margin_left = 60
margin_right = 5
margin_y = 5

# 左右 AUC、RR 之间的距离
gap_x = 0

# a 内部两部分之间的距离
gap_a_inner = 0

# a、b、c 实验之间的距离
gap_y = 0

# 统一缩放比例：1.0 = 不缩放
SCALE = 1.0

# 左侧标题字号
TITLE_FONTSIZE = 24

# 字母 a、b、c 的字号
LABEL_FONTSIZE = 28

# 可选：裁剪源 PDF 内部白边
# 顺序：（左、上、右、下），单位 pt
# 默认不裁剪，避免误切坐标或图例
CROP_AUC = (0, 0, 0, 0)
CROP_RR = (0, 0, 0, 0)
CROP_EXCLUDE = (0, 0, 0, 0)


# ============================================================
# 3. 检查 PDF 并读取真实尺寸
# ============================================================

def get_pdf_info(path, crop=(0, 0, 0, 0)):

    with fitz.open(str(path)) as doc:

        if doc.page_count < 1:
            raise ValueError(f"PDF没有页面：{path}")

        page = doc[0]
        rect = page.rect

        left, top, right, bottom = crop

        clip = fitz.Rect(
            rect.x0 + left,
            rect.y0 + top,
            rect.x1 - right,
            rect.y1 - bottom
        )

        if (
            clip.is_empty
            or clip.x0 < rect.x0
            or clip.y0 < rect.y0
            or clip.x1 > rect.x1
            or clip.y1 > rect.y1
        ):
            raise ValueError(f"裁剪区域无效：{path.name}")

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
            print("  [提示] PDF包含图片对象。")

        if drawing_count == 0:
            print("  [提示] 未检测到矢量绘图路径。")

        return {
            "clip": clip,
            "width": clip.width * SCALE,
            "height": clip.height * SCALE
        }


# ============================================================
# 4. 原始尺寸插入 PDF
# ============================================================

def insert_native_pdf(page, path, info, x, y):

    width = info["width"]
    height = info["height"]

    rect = fitz.Rect(
        x,
        y,
        x + width,
        y + height
    )

    with fitz.open(str(path)) as src_doc:

        page.show_pdf_page(
            rect,
            src_doc,
            pno=0,
            clip=info["clip"],
            keep_proportion=True,
            overlay=True
        )

    print(
        f"成功拼接：{path.name} | "
        f"缩放比例={SCALE:.3f}"
    )


# ============================================================
# 5. 插入左侧竖排标题
# ============================================================

def insert_vertical_title(page, title, y0, height):

    fontsize = TITLE_FONTSIZE

    # 根据真实文字长度判断是否超出行高
    text_width = fitz.get_text_length(
        title,
        fontname="helv",
        fontsize=fontsize
    )

    # 如果太长，仅适当缩小标题
    available_height = height - 16

    if text_width > available_height:

        fontsize *= available_height / text_width

        text_width = fitz.get_text_length(
            title,
            fontname="helv",
            fontsize=fontsize
        )

    # 左侧标题位置
    title_x = margin_left - 17

    # rotate=90 时，文字从基线向上延伸
    # 根据真实文字长度垂直居中
    title_y = y0 + (height + text_width) / 2

    page.insert_text(
        fitz.Point(title_x, title_y),
        title,
        fontsize=fontsize,
        fontname="helv",
        color=(0, 0, 0),
        rotate=90,
        overlay=True
    )


# ============================================================
# 6. 插入 a、b、c 字母
# ============================================================

def insert_label(page, label, y0):

    page.insert_text(
        fitz.Point(
            5,
            y0 + 20
        ),
        label,
        fontname="hebo",
        fontsize=LABEL_FONTSIZE,
        color=(0, 0, 0),
        overlay=True
    )


# ============================================================
# 7. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 7.1 检查所有文件是否存在
    # --------------------------------------------------------

    all_files = [exclude_pdf]

    for paths in pdf_files.values():
        all_files.extend(paths)

    missing = [
        path for path in all_files
        if not path.exists()
    ]

    if missing:

        print("以下PDF不存在：")

        for path in missing:
            print(path)

        raise FileNotFoundError("缺少输入PDF。")

    # --------------------------------------------------------
    # 7.2 获取所有 PDF 真实尺寸
    # --------------------------------------------------------

    print("\n开始检查源PDF：")

    exclude_info = get_pdf_info(
        exclude_pdf,
        CROP_EXCLUDE
    )

        # 单独放大 exclude.pdf
    EXCLUDE_SCALE = 1.25  # 放大20%

    exclude_info["width"] *= EXCLUDE_SCALE
    exclude_info["height"] *= EXCLUDE_SCALE

    source_info = {}

    for label, paths in pdf_files.items():

        for col, path in enumerate(paths):

            crop = CROP_AUC if col == 0 else CROP_RR

            source_info[(label, col)] = get_pdf_info(
                path,
                crop
            )

    # --------------------------------------------------------
    # 7.3 自动计算 AUC、RR 两列宽度
    # --------------------------------------------------------

    w_auc = max(
        source_info[(label, 0)]["width"]
        for label in pdf_files
    )

    w_rr = max(
        source_info[(label, 1)]["width"]
        for label in pdf_files
    )

    row_height = max(
        info["height"]
        for info in source_info.values()
    )

    pair_width = w_auc + gap_x + w_rr

    exclude_width = exclude_info["width"]
    exclude_height = exclude_info["height"]

    # 整张图的内容宽度
    content_width = max(
        pair_width,
        exclude_width
    )

    # AUC + RR 组合居中
    pair_x = (
        margin_left
        + (content_width - pair_width) / 2
    )

    # exclude.pdf 居中
    exclude_x = (
        margin_left
        + (content_width - exclude_width) / 2
    )

    # --------------------------------------------------------
    # 7.4 自动计算各区域纵坐标
    # --------------------------------------------------------

    exclude_y = margin_y

    a_row_y = (
        exclude_y
        + exclude_height
        + gap_a_inner
    )

    b_y = (
        a_row_y
        + row_height
        + gap_y
    )

    c_y = (
        b_y
        + row_height
        + gap_y
    )

    total_width = (
        margin_left
        + content_width
        + margin_right
    )

    total_height = (
        c_y
        + row_height
        + margin_y
    )

    print("\n========== 版式信息 ==========")

    print(f"AUC列宽度：{w_auc:.2f} pt")
    print(f"RR列宽度：{w_rr:.2f} pt")
    print(f"普通行高：{row_height:.2f} pt")

    print(
        f"exclude.pdf："
        f"{exclude_width:.2f} x "
        f"{exclude_height:.2f} pt"
    )

    print(
        f"最终PDF尺寸："
        f"{total_width:.2f} x "
        f"{total_height:.2f} pt"
    )

    # --------------------------------------------------------
    # 7.5 定义 AUC + RR 插入函数
    # --------------------------------------------------------

    def insert_auc_rr_row(page, label, y0):

        paths = pdf_files[label]

        for col, path in enumerate(paths):

            info = source_info[(label, col)]

            if col == 0:

                column_x = pair_x
                column_width = w_auc

            else:

                column_x = (
                    pair_x
                    + w_auc
                    + gap_x
                )

                column_width = w_rr

            # 横向和纵向居中
            x = (
                column_x
                + (column_width - info["width"]) / 2
            )

            y = (
                y0
                + (row_height - info["height"]) / 2
            )

            insert_native_pdf(
                page,
                path,
                info,
                x,
                y
            )

    # --------------------------------------------------------
    # 7.6 创建输出PDF
    # --------------------------------------------------------

    out_pdf = fitz.open()

    try:

        page = out_pdf.new_page(
            width=total_width,
            height=total_height
        )

        # ====================================================
        # a：Exclusion Interval
        # ====================================================

        insert_native_pdf(
            page,
            exclude_pdf,
            exclude_info,
            exclude_x,
            exclude_y
        )

        # a 只标记一次
        insert_label(
            page,
            "a",
            exclude_y
        )

        insert_vertical_title(
            page,
            "Exclusion interval",
            exclude_y,
            exclude_height
        )

        # ====================================================
        # a：Data exclusion result (AUC + RR)
        # ====================================================

        insert_auc_rr_row(
            page,
            "a",
            a_row_y
        )

        insert_vertical_title(
            page,
            "Data exclusion result",
            a_row_y,
            row_height
        )

        # ====================================================
        # b：Independent Hold-out Datasets
        # ====================================================

        insert_auc_rr_row(
            page,
            "b",
            b_y
        )

        insert_label(
            page,
            "b",
            b_y
        )

        insert_vertical_title(
            page,
            row_titles["b"],
            b_y,
            row_height
        )

        # ====================================================
        # c：Broader Outcome Definitions
        # ====================================================

        insert_auc_rr_row(
            page,
            "c",
            c_y
        )

        insert_label(
            page,
            "c",
            c_y
        )

        insert_vertical_title(
            page,
            row_titles["c"],
            c_y,
            row_height
        )

        out_pdf.save(
            str(output_path),
            garbage=4,
            deflate=True
        )

    finally:

        out_pdf.close()

    # --------------------------------------------------------
    # 7.7 检查最终PDF
    # --------------------------------------------------------

    with fitz.open(str(output_path)) as check_doc:

        check_page = check_doc[0]

        print("\n========== 最终PDF检查 ==========")

        print(
            f"images="
            f"{len(check_page.get_images(full=True))}"
        )

        print(
            f"drawings="
            f"{len(check_page.get_drawings())}"
        )

        print(
            f"words="
            f"{len(check_page.get_text('words'))}"
        )

        print(
            f"xobjects="
            f"{len(check_page.get_xobjects())}"
        )

    print(f"\n拼接完成：{output_path}")


if __name__ == "__main__":
    main()