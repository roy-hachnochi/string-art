from collections import defaultdict
from typing import Optional, Dict
import io
import math
import textwrap

from PIL import Image
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader

from string_art.configs import CanvasConfig, Shape
from string_art.utils import rgb2hex
from string_art.globals import COLOR_TYPE, COLOR_LINES_TYPE


TEXT_COLOR = Color(58 / 255, 58 / 255, 58 / 255)
FONT = "Helvetica"
FONT_SIZE = 12
GROUP_SIZE = 100
GROUPS_PER_PAGE = 4
PAGE_WIDTH, PAGE_HEIGHT = A4
TEMPLATE_PATH = 'examples/String Art Instructions Template.pdf'


def save_pdf_instructions(
        lines: COLOR_LINES_TYPE,
        pdf_path: str,
        cfg: CanvasConfig,
        image_path: Optional[str]
):
    """
    Create and save a PDF with instructions on how to create a string-art using the stored lines.
    :param lines: COLOR_LINES_TYPE object specifying the nail order of lines to reconstruct the image.
    :param pdf_path: path to save PDF to.
    :param cfg: Canvas config for canvas information.
    :param image_path: optional path to preview image to add to PDF.
    """

    reader = PdfReader(TEMPLATE_PATH)
    writer = PdfWriter()

    n_lines_per_color = _n_lines_per_color(lines)
    _intro_pages(reader, writer)
    _preview_page(reader, writer, image_path)
    _setup_page(reader, writer, cfg, n_lines_per_color)
    _instruction_pages(writer, lines, n_lines_per_color)

    with open(pdf_path, "wb") as f:
        writer.write(f)
    print(f"✅ PDF saved to {pdf_path}")


# --- Pages 1-2: Intro and generic instructions ---
def _intro_pages(reader, writer):
    # Intro
    writer.add_page(reader.pages[0])

    # General instructions
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    c.setFont(FONT, 10)
    c.drawRightString(PAGE_WIDTH / 2, 1.5 * cm, "1")
    c.save()
    packet.seek(0)
    instruction_page = reader.pages[1]
    instruction_page.merge_page(PdfReader(packet).pages[0])
    writer.add_page(instruction_page)


# --- Page 3: Preview page ---
def _preview_page(reader, writer, image_path):
    add_preview_page = True
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    c.setFont(FONT, FONT_SIZE)
    c.setFillColor(TEXT_COLOR)
    try:
        with Image.open(image_path) as im:
            img_w, img_h = im.size
            aspect = img_h / img_w

            # calculate image size on page
            image_width = PAGE_WIDTH - 4 * cm
            image_height = image_width * aspect
            if image_height + 9 * cm > PAGE_HEIGHT:
                image_height = PAGE_HEIGHT - 9 * cm
                image_width = image_height / aspect

            # resize image to max size
            im.thumbnail((image_width, image_height), Image.LANCZOS)
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            buffer.seek(0)
            image_reader = ImageReader(buffer)
        x = (PAGE_WIDTH - image_width) / 2
        y = PAGE_HEIGHT - 4.5 * cm - image_height
        c.drawImage(image_reader, x=x, y=y, width=image_width, preserveAspectRatio=True)
    except Exception:
        add_preview_page = False
    c.setFont(FONT, 10)
    c.drawRightString(PAGE_WIDTH / 2, 1.5 * cm, f"{len(writer.pages)}")
    c.showPage()
    c.save()

    if add_preview_page:
        packet.seek(0)
        preview_overlay = PdfReader(packet).pages[0]
        base_preview = reader.pages[2]
        base_preview.merge_page(preview_overlay)
        writer.add_page(base_preview)


 # --- Page 4: Setup page ---
def _setup_page(reader, writer, cfg, n_lines_per_color):
    canvas_page_index = 3 if cfg.shape == Shape.ELLIPSE else 4
    base_canvas_page = reader.pages[canvas_page_index]

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    c.setFont(FONT, FONT_SIZE)
    c.setFillColor(TEXT_COLOR)

    # canvas info
    y = 25 * cm
    c.drawString(3 * cm, y, f"Number of nails: {cfg.nails}")
    y -= 0.8 * cm
    h, w = cfg.size
    h, w = h / 10, w / 10
    shape = 'Circular' if cfg.shape == Shape.ELLIPSE else 'Rectangular'
    c.drawString(3 * cm, y, f"Canvas size ({shape}): {h} cm × {w} cm")
    y -= 0.8 * cm
    if cfg.shape == Shape.ELLIPSE:
        c.drawString(3 * cm, y, f"Distance between nails: {360 / cfg.nails:.2f} °")
    else:
        perimeter = 2 * (h + w)
        n_nails_h = int(h * cfg.nails / perimeter)
        n_nails_w = int(cfg.nails - n_nails_h * 2) // 2
        d = h / n_nails_h
        c.drawString(3 * cm, y, f"Distance between nails: {d:.2f} cm")

    # thread info
    y -= 0.8 * cm
    r = max(h, w) / 2
    c.drawString(3 * cm, y, "Number of lines per color:")
    for idx, (color, n_lines) in enumerate(n_lines_per_color.items()):
        y -= 0.8 * cm
        _color_text(3.5 * cm, y, f"Color #{idx + 1}", color, c)
        c.setFillColor(TEXT_COLOR)
        c.drawString(5.5 * cm, y, f"{rgb2hex(color)}: {n_lines} (~{_string_length(n_lines, r)} meters)")

    # setup scheme
    c.setFont(FONT, FONT_SIZE)
    c.setFillColor(TEXT_COLOR)
    if cfg.shape == Shape.ELLIPSE:
        b, u, l, r = 2.5 * cm, 15.7 * cm, 2.3 * cm, 16.8 * cm
        c.drawString(r, (b + u) / 2, f"Nail #1")
        c.drawString((l + r) / 2, b, f"Nail #{cfg.nails // 4 + 1}")
        c.drawString(l, (b + u) / 2, f"Nail #{2 * cfg.nails // 4 + 1}")
        c.drawString((l + r) / 2, u, f"Nail #{3 * cfg.nails // 4 + 1}")
    else:
        b, u, l, r = 3.3 * cm, 15.7 * cm, 4.3 * cm, 14.7 * cm
        c.drawString(r, u, f"Nail #1")
        c.drawString(r, b, f"Nail #{n_nails_h + 1}")
        c.drawString(l, b, f"Nail #{n_nails_h + n_nails_w + 1}")
        c.drawString(l, u, f"Nail #{2 * n_nails_h + n_nails_w + 1}")

    c.setFont(FONT, 10)
    c.drawRightString(PAGE_WIDTH / 2, 1.5 * cm, f"{len(writer.pages)}")
    c.showPage()
    c.save()
    packet.seek(0)
    canvas_overlay = PdfReader(packet).pages[0]
    base_canvas_page.merge_page(canvas_overlay)
    writer.add_page(base_canvas_page)

def _n_lines_per_color(lines: COLOR_LINES_TYPE) -> Dict[COLOR_TYPE, int]:
    n_lines = defaultdict(int)
    for color, path in lines:
        n_lines[color] += len(path)
    return n_lines

def _string_length(n_lines: int, r: float) -> int:
    length = 4 * n_lines * r / (math.pi * 100)  # meters
    return 100 * math.ceil(length / 100)  # make divisible by 100

def _color_text(box_x, box_y, text, color, canvas):
    r, g, b = color
    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    # Box dimensions
    box_w = 1.8 * cm
    box_h = 0.6 * cm

    if color == (255, 255, 255):  # white
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setFillColorRGB(1, 1, 1)
        canvas.rect(box_x, box_y - 4, box_w, box_h, fill=True, stroke=True)
        canvas.setFillColorRGB(0, 0, 0)
    else:
        canvas.setStrokeColorRGB(r / 255, g / 255, b / 255)
        canvas.setFillColorRGB(r / 255, g / 255, b / 255)
        canvas.rect(box_x, box_y - 4, box_w, box_h, fill=True, stroke=True)
        fill_color = (0, 0, 0) if brightness > 60 else (1, 1, 1)
        canvas.setFillColorRGB(*fill_color)  # readable text

    # Draw text
    canvas.setFont(FONT, FONT_SIZE)
    canvas.drawString(box_x + 2, box_y, text)


# --- Pages 5+: Instruction pages ---
def _instruction_pages(writer, lines, n_lines_per_color):
    n_nails = sum(n for n in n_lines_per_color.values())
    color_idx = {color: i + 1 for i, color in enumerate(n_lines_per_color)}

    # Create groups
    nail_count = 0
    instruction_groups, group = [], []
    for color, path in lines:
        group.append((color, []))
        for i, nail in enumerate(path):
            group[-1][1].append(nail)
            nail_count += 1
            if nail_count % GROUP_SIZE == 0:
                instruction_groups.append({"start": nail_count - GROUP_SIZE + 1, "end": nail_count, "group": group})
                if i < len(path) - 1:
                    group = [(color, [])]
    if nail_count % GROUP_SIZE != 0:  # final incomplete group
        group_size = nail_count % GROUP_SIZE
        instruction_groups.append({"start": nail_count - group_size + 1, "end": nail_count, "group": group})

    # Create pages
    normal_style = ParagraphStyle(
        name='Normal',
        fontName=FONT,
        fontSize=FONT_SIZE,
        textColor=TEXT_COLOR,
        firstLineIndent=1.9*cm,
        leading=FONT_SIZE + 4,
    )

    for g_idx in range(0, len(instruction_groups), GROUPS_PER_PAGE):
        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=A4)
        c.setFont(FONT, FONT_SIZE)
        c.setFillColor(TEXT_COLOR)

        y = 27 * cm
        for group in instruction_groups[g_idx:min(g_idx + GROUPS_PER_PAGE, len(instruction_groups))]:
            c.setLineWidth(1)
            c.rect(2 * cm, y, 7, 7)
            c.setFont(FONT + "-Bold", 12)
            header_text = f"Lines {group['start']}-{group['end']} out of {n_nails}:"
            c.drawString(2 * cm + 20, y, header_text)
            y -= 0.4 * cm

            # Draw nail path
            for color, nails in group['group']:
                _color_text(2 * cm, y - 0.4 * cm, f"Color #{color_idx[color]}", color, c)
                c.setStrokeColor(TEXT_COLOR)
                c.setFillColor(TEXT_COLOR)
                text = f": " + " – ".join(str(n + 1) for n in nails)
                paragraph = Paragraph(text, normal_style)
                max_width = PAGE_WIDTH - 4 * cm
                max_height = 0.6 * cm
                width, height = paragraph.wrap(max_width, max_height)
                paragraph.drawOn(c, 2 * cm, y - height)
                y -= height + 0.2 * cm
            y -= 0.8 * cm

        c.setFont(FONT, 10)
        c.drawRightString(PAGE_WIDTH / 2, 1.5 * cm, f"{len(writer.pages)}")
        c.showPage()
        c.save()

        packet.seek(0)
        new_page = PdfReader(packet).pages[0]
        writer.add_page(new_page)

def _wrapped_line(canvas, text, length, x_pos, y_pos, y_offset):
    if len(text) > length:
        wraps = textwrap.wrap(text, length)
        for x in range(len(wraps)):
            canvas.drawCenteredString(x_pos, y_pos, wraps[x])
            y_pos -= y_offset
        y_pos += y_offset  # add back offset after last wrapped line
    else:
        canvas.drawCenteredString(x_pos, y_pos, text)
    return y_pos