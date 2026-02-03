#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pikepdf", "pymupdf"]
# ///
"""
PDF Accessibility Tool

A simple tool to make PDFs accessible for screen readers and
pass Adobe Acrobat's accessibility checker (PDF/UA compliance).

Usage:
  uv run pdf_access.py                    # File picker
  uv run pdf_access.py document.pdf       # Process PDF
  uv run pdf_access.py alt_text.md        # Apply alt text from markdown
  uv run pdf_access.py --check file.pdf   # Check accessibility status
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import fitz
import pikepdf

# ============================================================================
# Constants
# ============================================================================

KNOWN_TAGS = ["Document", "H1", "H2", "H3", "P", "Figure", "Table", "L", "LI"]
CATEGORIES = [
    "Document Settings",
    "Fonts",
    "Document Structure",
    "Headings",
    "Lists",
    "Tables",
    "Links",
    "Images & Alt Text",
    "Navigation",
    "Metadata",
    "Advanced (PDF/UA)",
]


# ============================================================================
# pikepdf helpers (type checking for PDF objects)
# ============================================================================


def is_mcid_value(val) -> bool:
    """Check if value is an MCID (integer, not a dict)."""
    try:
        int(val)
        return not hasattr(val, "get")
    except (TypeError, ValueError):
        return False


def is_pdf_array(val) -> bool:
    """Check if value is a pikepdf Array or Python list (not dict/str)."""
    # Check for pikepdf.Array first
    if isinstance(val, pikepdf.Array):
        return True
    # Also accept Python lists (for testing with mocks)
    if isinstance(val, list):
        return True
    # For other iterables, check it's not a dict-like or string
    if isinstance(val, (str, bytes, dict)):
        return False
    if isinstance(val, pikepdf.Dictionary):
        return False
    # For mock objects in tests, check if iterable but not dict-like
    if hasattr(val, "__iter__") and not hasattr(val, "keys"):
        return True
    return False


def is_struct_elem(elem) -> bool:
    """Check if element is a structure element (has /S tag)."""
    if not hasattr(elem, "get"):
        return False
    try:
        return elem.get("/S") is not None
    except (TypeError, ValueError):
        return False


def _count_structure_tags(struct_tree) -> Dict[str, Any]:
    """
    Count all structure tags and analyze figure alt text in a structure tree.
    Returns dict with 'tag_counts', 'figures_with_alt', 'figures_with_placeholder', 'figures_without_alt'.
    """
    result = {
        "tag_counts": {},
        "figures_with_alt": 0,
        "figures_with_placeholder": 0,
        "figures_without_alt": 0,
    }
    tag_counts = result["tag_counts"]

    def count_elem(elem):
        if not hasattr(elem, "get"):
            return

        try:
            tag = str(elem.get("/S", "")).replace("/", "")
        except (TypeError, ValueError):
            return

        if tag:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

            if tag == "Figure":
                # Check both /Alt and /ActualText (Pages may use either)
                alt = elem.get("/Alt")
                actual_text = elem.get("/ActualText")
                description = alt or actual_text

                if description:
                    desc_str = str(description)
                    if "alt text needed" in desc_str.lower() or desc_str == "":
                        result["figures_with_placeholder"] += 1
                    else:
                        result["figures_with_alt"] += 1
                else:
                    result["figures_without_alt"] += 1

        try:
            kids = elem.get("/K")
        except (TypeError, ValueError):
            return

        if kids is None:
            return

        if is_pdf_array(kids):
            try:
                for kid in kids:
                    count_elem(kid)
            except (TypeError, ValueError):
                pass
        else:
            count_elem(kids)

    # Start from the root /K element
    doc_elem = struct_tree.get("/K")
    if doc_elem is not None:
        if is_struct_elem(doc_elem):
            count_elem(doc_elem)
        elif is_pdf_array(doc_elem):
            try:
                for elem in doc_elem:
                    count_elem(elem)
            except (TypeError, ValueError):
                pass

    return result


def _check_for_mcids(struct_tree) -> bool:
    """
    Check if a structure tree contains any MCIDs (content linked to structure).
    Returns True if MCIDs are found.
    """
    found_mcid = False

    def check_elem(elem):
        nonlocal found_mcid
        if found_mcid or not hasattr(elem, "get"):
            return

        try:
            kids = elem.get("/K")
        except (TypeError, ValueError):
            return

        if kids is None:
            return

        # /K can be: integer (MCID), dict with /MCID, or array of these
        if is_mcid_value(kids):
            found_mcid = True
            return

        if is_pdf_array(kids):
            try:
                for kid in kids:
                    if is_mcid_value(kid):
                        found_mcid = True
                        return
                    if hasattr(kid, "get"):
                        try:
                            if (
                                kid.get("/MCID") is not None
                                or kid.get("/Type") == "/MCR"
                            ):
                                found_mcid = True
                                return
                        except (TypeError, ValueError):
                            pass
                        check_elem(kid)
                        if found_mcid:
                            return
            except (TypeError, ValueError):
                pass
            return

        # /K is a dictionary with /MCID or a struct elem
        if hasattr(kids, "get"):
            if kids.get("/MCID") is not None or kids.get("/Type") == "/MCR":
                found_mcid = True
                return
            check_elem(kids)

    # Start from the root /K element
    doc_elem = struct_tree.get("/K")
    if doc_elem is None:
        return False

    if is_struct_elem(doc_elem):
        check_elem(doc_elem)
    elif is_pdf_array(doc_elem):
        try:
            for elem in doc_elem:
                check_elem(elem)
                if found_mcid:
                    break
        except (TypeError, ValueError):
            pass

    return found_mcid


def _count_empty_marked_content(pdf) -> int:
    """
    Count marked content blocks (BDC/EMC with MCID) that contain no actual content.

    PDF/UA requires that all marked content blocks contain real content (text, images, etc.).
    Empty blocks (e.g., font-setup-only BT/ET) are a compliance violation.

    Returns the count of empty marked content blocks.
    """
    import re

    empty_count = 0

    for page in pdf.pages:
        content = page.get("/Contents")
        if not content:
            continue

        try:
            if isinstance(content, pikepdf.Array):
                # Array of content streams
                raw = b"".join(c.read_bytes() for c in content)
            else:
                raw = content.read_bytes()
        except Exception:
            continue

        text = raw.decode("latin-1", errors="replace")

        # Find all BDC...EMC blocks with MCIDs
        # Pattern: /TagName << /MCID n >> BDC ... EMC
        pattern = r"/\w+\s*<<\s*/MCID\s+\d+\s*>>\s*BDC(.*?)EMC"

        for match in re.finditer(pattern, text, re.DOTALL):
            content_block = match.group(1)

            # Check if block contains actual content:
            # - Text: Tj or TJ operators
            # - Image/XObject: Do operator
            has_text = bool(re.search(r"Tj|TJ", content_block))
            has_image = bool(re.search(r"/[^\s]+\s+Do", content_block))

            if not has_text and not has_image:
                empty_count += 1

    return empty_count


def _count_untagged_graphics(pdf) -> Dict[str, Any]:
    """
    Count graphics operators (path painting) that are outside any marked content.

    PDF/UA requires ALL content to be either:
    1. Inside tagged marked content (BDC/EMC with structure), OR
    2. Marked as Artifact (decorative)

    Graphics operators outside both are a compliance violation.

    Returns dict with:
        - total: total untagged graphics count
        - by_page: dict mapping page number to count
        - operators: dict mapping operator type to count
    """
    import re

    # Path painting operators that draw visible content
    # f, F = fill (nonzero winding)
    # f*, F* = fill (even-odd)
    # s, S = stroke (s closes path first)
    # b, B = fill + stroke (nonzero), b closes path first
    # b*, B* = fill + stroke (even-odd), b* closes path first
    # Note: n = end path without painting (no-op, doesn't need marking)
    graphics_operators = {"f", "F", "f*", "F*", "s", "S", "b", "B", "b*", "B*"}

    result = {
        "total": 0,
        "by_page": {},
        "operators": {},
    }

    for page_num, page in enumerate(pdf.pages):
        content = page.get("/Contents")
        if not content:
            continue

        try:
            if isinstance(content, pikepdf.Array):
                raw = b"".join(c.read_bytes() for c in content)
            else:
                raw = content.read_bytes()
        except Exception:
            continue

        text = raw.decode("latin-1", errors="replace")

        # Split into marked regions and unmarked regions
        # Find all BDC...EMC and BMC...EMC blocks
        # Pattern captures both structured (BDC) and simple (BMC) marked content
        marked_regions = []

        # Pattern for tagged content: /Tag << ... >> BDC ... EMC
        for match in re.finditer(r"/\w+\s*<<[^>]*>>\s*BDC.*?EMC", text, re.DOTALL):
            marked_regions.append((match.start(), match.end()))

        # Pattern for artifact content: /Artifact BMC ... EMC or /Artifact << ... >> BDC ... EMC
        for match in re.finditer(r"/Artifact\s+BMC.*?EMC", text, re.DOTALL):
            marked_regions.append((match.start(), match.end()))
        for match in re.finditer(r"/Artifact\s*<<[^>]*>>\s*BDC.*?EMC", text, re.DOTALL):
            marked_regions.append((match.start(), match.end()))

        # Sort marked regions by start position
        marked_regions.sort()

        def is_in_marked_region(pos):
            """Check if position is inside any marked content region."""
            for start, end in marked_regions:
                if start <= pos < end:
                    return True
                if start > pos:  # Regions are sorted, no need to continue
                    break
            return False

        # Find all graphics operators and check if they're in marked content
        page_count = 0
        for match in re.finditer(r"\b([fFsSnbB]\*?)\b", text):
            op = match.group(1)
            if op in graphics_operators:
                if not is_in_marked_region(match.start()):
                    page_count += 1
                    result["operators"][op] = result["operators"].get(op, 0) + 1

        if page_count > 0:
            result["by_page"][page_num + 1] = page_count
            result["total"] += page_count

    return result


def _analyze_structure_deeply(struct_tree, pdf) -> Dict[str, Any]:
    """
    Perform deep analysis of structure tree for PDF/UA compliance.
    Returns detailed information about headings, lists, tables, links, etc.
    """
    analysis = {
        # Heading analysis
        "headings": [],  # List of (level, text_preview) in document order
        "heading_hierarchy_issues": [],  # List of issues like "H3 after H1 (skipped H2)"
        # List analysis
        "lists": {
            "count": 0,
            "well_formed": 0,
            "malformed": [],  # List of issues
        },
        # Table analysis
        "tables": {
            "count": 0,
            "with_headers": 0,
            "without_headers": 0,
            "issues": [],
        },
        # Link analysis
        "links": {
            "count": 0,
            "with_content": 0,  # Have both OBJR and text
            "objr_only": 0,  # Only have OBJR (incomplete)
            "issues": [],
        },
        # Figure analysis (more detailed than _count_structure_tags)
        "figures": {
            "count": 0,
            "with_alt": 0,
            "with_empty_alt": 0,
            "with_placeholder_alt": 0,
            "without_alt": 0,
            "alt_texts": [],  # List of (figure_num, alt_text) for review
        },
        # Role mapping
        "has_rolemap": False,
        "custom_roles": [],  # Roles not in standard set
        # Document element
        "has_document_tag": False,
    }

    # Standard PDF structure types
    STANDARD_ROLES = {
        "Document",
        "Part",
        "Art",
        "Sect",
        "Div",
        "BlockQuote",
        "Caption",
        "TOC",
        "TOCI",
        "Index",
        "NonStruct",
        "Private",
        "H",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "P",
        "L",
        "LI",
        "Lbl",
        "LBody",
        "Table",
        "TR",
        "TH",
        "TD",
        "THead",
        "TBody",
        "TFoot",
        "Span",
        "Quote",
        "Note",
        "Reference",
        "BibEntry",
        "Code",
        "Link",
        "Annot",
        "Ruby",
        "RB",
        "RT",
        "RP",
        "Warichu",
        "WT",
        "WP",
        "Figure",
        "Formula",
        "Form",
    }

    last_heading_level = 0
    figure_num = 0

    def get_tag_name(elem):
        """Get clean tag name from element."""
        if not hasattr(elem, "get"):
            return None
        try:
            s = elem.get("/S")
            if s:
                return str(s).replace("/", "")
        except:
            pass
        return None

    def analyze_elem(elem, parent_tag=None):
        nonlocal last_heading_level, figure_num

        if not hasattr(elem, "get"):
            return

        tag = get_tag_name(elem)
        if not tag:
            return

        # Check for custom roles
        if tag not in STANDARD_ROLES:
            if tag not in analysis["custom_roles"]:
                analysis["custom_roles"].append(tag)

        # Document tag check
        if tag == "Document":
            analysis["has_document_tag"] = True

        # Heading analysis
        if tag in ("H1", "H2", "H3", "H4", "H5", "H6"):
            level = int(tag[1])
            analysis["headings"].append((level, tag))

            # Check hierarchy
            if last_heading_level > 0 and level > last_heading_level + 1:
                issue = f"{tag} after H{last_heading_level} (skipped H{last_heading_level + 1})"
                analysis["heading_hierarchy_issues"].append(issue)

            last_heading_level = level

        elif tag == "H":
            # Generic heading - not ideal but valid
            analysis["headings"].append((0, "H"))

        # List analysis
        elif tag == "L":
            analysis["lists"]["count"] += 1
            # Check if list has proper LI children
            k = elem.get("/K")
            has_li = False
            if k:
                if is_pdf_array(k):
                    for child in k:
                        if get_tag_name(child) == "LI":
                            has_li = True
                            break
                elif get_tag_name(k) == "LI":
                    has_li = True

            if has_li:
                analysis["lists"]["well_formed"] += 1
            else:
                analysis["lists"]["malformed"].append("List without LI children")

        elif tag == "LI":
            # Check LI has Lbl and/or LBody
            k = elem.get("/K")
            has_lbl_or_lbody = False
            if k:
                if is_pdf_array(k):
                    for child in k:
                        child_tag = get_tag_name(child)
                        if child_tag in ("Lbl", "LBody"):
                            has_lbl_or_lbody = True
                            break
                else:
                    child_tag = get_tag_name(k)
                    if child_tag in ("Lbl", "LBody"):
                        has_lbl_or_lbody = True

            # LI can have direct content or Lbl/LBody - both are valid
            # But best practice is Lbl + LBody

        # Table analysis
        elif tag == "Table":
            analysis["tables"]["count"] += 1
            # Check for TH elements
            has_th = False

            def check_for_th(e):
                nonlocal has_th
                if has_th:
                    return
                t = get_tag_name(e)
                if t == "TH":
                    has_th = True
                    return
                k = e.get("/K") if hasattr(e, "get") else None
                if k:
                    if is_pdf_array(k):
                        for child in k:
                            check_for_th(child)
                    elif hasattr(k, "get"):
                        check_for_th(k)

            check_for_th(elem)

            if has_th:
                analysis["tables"]["with_headers"] += 1
            else:
                analysis["tables"]["without_headers"] += 1
                analysis["tables"]["issues"].append("Table without TH header cells")

        # Link analysis
        elif tag == "Link":
            analysis["links"]["count"] += 1
            k = elem.get("/K")
            has_objr = False
            has_content = False

            if k:
                if is_pdf_array(k):
                    for item in k:
                        if is_mcid_value(item):
                            has_content = True
                        elif hasattr(item, "get"):
                            if item.get("/Type") == pikepdf.Name("/OBJR"):
                                has_objr = True
                            elif item.get("/MCID") is not None:
                                has_content = True
                elif hasattr(k, "get"):
                    if k.get("/Type") == pikepdf.Name("/OBJR"):
                        has_objr = True
                    elif k.get("/MCID") is not None:
                        has_content = True
                elif is_mcid_value(k):
                    has_content = True

            if has_objr and has_content:
                analysis["links"]["with_content"] += 1
            elif has_objr:
                analysis["links"]["objr_only"] += 1
                analysis["links"]["issues"].append(
                    "Link with annotation but no text content"
                )

        # Figure analysis
        elif tag == "Figure":
            figure_num += 1
            analysis["figures"]["count"] += 1

            alt = elem.get("/Alt")
            actual_text = elem.get("/ActualText")
            description = alt or actual_text

            if description:
                desc_str = str(description).strip()
                if desc_str == "":
                    analysis["figures"]["with_empty_alt"] += 1
                    analysis["figures"]["alt_texts"].append((figure_num, "[empty]"))
                elif "alt text needed" in desc_str.lower() or desc_str.startswith(
                    "Image "
                ):
                    analysis["figures"]["with_placeholder_alt"] += 1
                    analysis["figures"]["alt_texts"].append(
                        (figure_num, f"[placeholder: {desc_str[:30]}...]")
                    )
                else:
                    analysis["figures"]["with_alt"] += 1
                    preview = desc_str[:50] + "..." if len(desc_str) > 50 else desc_str
                    analysis["figures"]["alt_texts"].append((figure_num, preview))
            else:
                analysis["figures"]["without_alt"] += 1
                analysis["figures"]["alt_texts"].append((figure_num, "[missing]"))

        # Recurse into children
        k = elem.get("/K")
        if k:
            if is_pdf_array(k):
                for child in k:
                    if hasattr(child, "get") and child.get("/S"):
                        analyze_elem(child, tag)
            elif hasattr(k, "get") and k.get("/S"):
                analyze_elem(k, tag)

    # Check RoleMap
    role_map = struct_tree.get("/RoleMap")
    if role_map:
        analysis["has_rolemap"] = True

    # Analyze structure
    doc_elem = struct_tree.get("/K")
    if doc_elem:
        if is_struct_elem(doc_elem):
            analyze_elem(doc_elem)
        elif is_pdf_array(doc_elem):
            for elem in doc_elem:
                analyze_elem(elem)

    return analysis


def _check_bookmarks(pdf) -> Dict[str, Any]:
    """Check document outline (bookmarks) for accessibility."""
    result = {
        "has_bookmarks": False,
        "bookmark_count": 0,
        "issues": [],
    }

    try:
        outlines = pdf.Root.get("/Outlines")
        if outlines:
            result["has_bookmarks"] = True

            # Count bookmarks
            def count_bookmarks(outline_item):
                count = 0
                if outline_item:
                    count = 1
                    # Check for children
                    first = outline_item.get("/First")
                    if first:
                        count += count_bookmarks(first)
                    # Check for siblings
                    next_item = outline_item.get("/Next")
                    if next_item:
                        count += count_bookmarks(next_item)
                return count

            first = outlines.get("/First")
            if first:
                result["bookmark_count"] = count_bookmarks(first)
    except Exception:
        pass

    return result


def _check_metadata_consistency(pdf) -> Dict[str, Any]:
    """Check consistency between Info dict and XMP metadata."""
    result = {
        "info_title": None,
        "xmp_title": None,
        "title_mismatch": False,
        "info_lang": None,
        "root_lang": None,
        "issues": [],
    }

    try:
        # Get Info dict title
        if pdf.trailer.get("/Info"):
            info_title = pdf.trailer.Info.get("/Title")
            if info_title:
                result["info_title"] = str(info_title)

        # Get XMP title
        try:
            with pdf.open_metadata() as meta:
                xmp_title = meta.get("dc:title")
                if xmp_title:
                    result["xmp_title"] = str(xmp_title)
        except Exception:
            pass

        # Check for mismatch
        if result["info_title"] and result["xmp_title"]:
            if result["info_title"] != result["xmp_title"]:
                result["title_mismatch"] = True
                result["issues"].append(
                    "Title mismatch between Info dict and XMP metadata"
                )

        # Get language
        root_lang = pdf.Root.get("/Lang")
        if root_lang:
            result["root_lang"] = str(root_lang)

    except Exception:
        pass

    return result


def select_pdf_file() -> Optional[str]:
    """Open a file picker dialog and return the selected PDF path."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = filedialog.askopenfilename(
            title="Select a PDF to make accessible",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )

        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"Error opening file picker: {e}")
        print("\nYou can also run with a file path argument:")
        print("  uv run pdf_access.py /path/to/your/file.pdf\n")
        return None


def analyze_document(pdf_path: Path | str) -> Dict[str, Any]:
    """
    Analyze PDF content to determine structure using pymupdf.

    Args:
        pdf_path: Path to the PDF file.
    """
    pdf_path = Path(pdf_path) if not isinstance(pdf_path, Path) else pdf_path
    doc = fitz.open(str(pdf_path))

    analysis = {
        "pages": [],
        "font_sizes": set(),
        "images": [],
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_data = {"number": page_num, "elements": [], "tables": [], "links": []}

        # Detect tables and track cell bboxes for proper MCID linking
        table_bboxes = []
        cell_info_map = []  # List of (cell_bbox, table_idx, row_idx, cell_idx, is_header)
        try:
            tables = page.find_tables()
            for table_idx, table in enumerate(tables.tables):
                table_data = {
                    "bbox": table.bbox,
                    "rows": [],
                    "row_count": table.row_count,
                    "col_count": table.col_count,
                }
                table_bboxes.append(table.bbox)

                # Extract table content and cell bboxes
                extracted = table.extract()
                for row_idx, row_obj in enumerate(table.rows):
                    is_header = row_idx == 0
                    row_data = {
                        "cells": [],
                        "is_header": is_header,
                        "cell_bboxes": list(row_obj.cells),  # Store cell bboxes
                    }
                    # Store cell info for lookup
                    for cell_idx, cell_bbox in enumerate(row_obj.cells):
                        cell_info_map.append(
                            (cell_bbox, table_idx, row_idx, cell_idx, is_header)
                        )
                    # Store extracted text
                    if row_idx < len(extracted):
                        for cell in extracted[row_idx]:
                            row_data["cells"].append(cell if cell else "")
                    table_data["rows"].append(row_data)

                page_data["tables"].append(table_data)
        except Exception:
            pass  # Table detection not available or failed

        def find_cell_for_bbox(bbox):
            """Find which table cell a bbox belongs to. Returns (table_idx, row_idx, cell_idx, is_header) or None."""
            for cell_bbox, table_idx, row_idx, cell_idx, is_header in cell_info_map:
                # Check if bbox center is within cell
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                if (
                    cell_bbox[0] <= center_x <= cell_bbox[2]
                    and cell_bbox[1] <= center_y <= cell_bbox[3]
                ):
                    return (table_idx, row_idx, cell_idx, is_header)
            return None

        def is_in_table(bbox):
            """Check if a bbox overlaps with any detected table."""
            for tb in table_bboxes:
                # Check overlap
                if (
                    bbox[0] < tb[2]
                    and bbox[2] > tb[0]
                    and bbox[1] < tb[3]
                    and bbox[3] > tb[1]
                ):
                    return True
            return False

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = round(span["size"], 1)
                        text = span["text"].strip()
                        bbox = span["bbox"]
                        if not text:
                            continue

                        # Check if this text belongs to a table cell
                        cell_info = find_cell_for_bbox(bbox)

                        if cell_info:
                            # This text is in a table cell - include it with cell metadata
                            table_idx, row_idx, cell_idx, is_header = cell_info
                            page_data["elements"].append(
                                {
                                    "type": "table_cell",
                                    "text": text,
                                    "size": size,
                                    "bbox": bbox,
                                    "table_idx": table_idx,
                                    "row_idx": row_idx,
                                    "cell_idx": cell_idx,
                                    "is_header": is_header,
                                    "font": span.get("font", ""),
                                }
                            )
                        elif not is_in_table(bbox):
                            # Regular text outside tables
                            analysis["font_sizes"].add(size)
                            # Extract font flags for bold detection
                            flags = span.get("flags", 0)
                            is_bold = bool(flags & 2**4)  # bit 4 = bold
                            font_name = span.get("font", "").lower()
                            # Also check font name for bold indicator
                            if "bold" in font_name or "black" in font_name:
                                is_bold = True
                            page_data["elements"].append(
                                {
                                    "type": "text",
                                    "text": text,
                                    "size": size,
                                    "bbox": bbox,
                                    "bold": is_bold,
                                    "font": span.get("font", ""),
                                }
                            )
            elif block["type"] == 1:  # image
                bbox = block["bbox"]
                if not is_in_table(bbox):
                    page_data["elements"].append(
                        {
                            "type": "image",
                            "bbox": bbox,
                            "width": block["width"],
                            "height": block["height"],
                        }
                    )
                    analysis["images"].append(
                        {
                            "page": page_num + 1,
                            "width": block["width"],
                            "height": block["height"],
                        }
                    )

        # Collect link annotations
        for link in page.get_links():
            if link.get("kind") == fitz.LINK_URI or link.get("uri"):
                # Link rect is already in PyMuPDF coordinates
                link_rect = link.get("from")
                if link_rect:
                    page_data["links"].append(
                        {
                            "bbox": tuple(link_rect),
                            "uri": link.get("uri", ""),
                        }
                    )

        analysis["pages"].append(page_data)

    doc.close()

    # Determine heading sizes (largest fonts are likely headings)
    sizes = sorted(analysis["font_sizes"], reverse=True)
    analysis["heading_sizes"] = sizes[:3] if len(sizes) >= 3 else sizes

    # Compute most common font size (likely body text)
    size_counts: Dict[float, int] = {}
    for page_data in analysis["pages"]:
        for elem in page_data["elements"]:
            if elem["type"] == "text":
                s = elem["size"]
                size_counts[s] = size_counts.get(s, 0) + 1

    if size_counts:
        analysis["body_size"] = max(size_counts.keys(), key=lambda s: size_counts[s])
    else:
        analysis["body_size"] = 12.0  # Default

    return analysis


def classify_element(
    elem: Dict, heading_sizes: List[float], body_size: float = 0
) -> str:
    """
    Classify a text element as heading or paragraph.

    Uses multiple signals:
    - Font size relative to body text
    - Bold styling
    - Text length (headings are typically short)
    """
    if elem["type"] == "image":
        return "Figure"

    size = elem["size"]
    text = elem.get("text", "")
    is_bold = elem.get("bold", False)

    # If we have heading sizes from analysis, use them
    if heading_sizes and size == heading_sizes[0]:
        return "H1"
    elif len(heading_sizes) > 1 and size == heading_sizes[1]:
        return "H2"
    elif len(heading_sizes) > 2 and size == heading_sizes[2]:
        return "H3"

    # Additional heuristics for edge cases:
    # If text is bold, larger than body, and short - likely a heading
    if body_size > 0 and size > body_size:
        text_len = len(text)
        # Short bold text that's larger than body = likely heading
        if is_bold and text_len < 100:
            size_ratio = size / body_size
            if size_ratio >= 1.5:
                return "H1"
            elif size_ratio >= 1.25:
                return "H2"
            elif size_ratio >= 1.1:
                return "H3"

    return "P"


def find_image_caption(elements: List[Dict], image_index: int) -> Optional[str]:
    """
    Try to find a caption for an image by looking at nearby text.
    """
    if image_index + 1 >= len(elements):
        return None

    next_elem = elements[image_index + 1]
    if next_elem["type"] != "text":
        return None

    text = next_elem["text"].strip()
    caption_starters = [
        "figure",
        "fig.",
        "fig ",
        "image",
        "img.",
        "img ",
        "photo",
        "chart",
        "graph",
        "diagram",
        "table",
        "illustration",
    ]

    text_lower = text.lower()
    for starter in caption_starters:
        if text_lower.startswith(starter):
            return text

    return None


def extract_images(pdf_path: Path | str, output_dir: Path) -> List[Dict[str, Any]]:
    """Extract all images from PDF."""
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    image_list = []
    doc = fitz.open(str(pdf_path))

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_info_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_info_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                image_path = images_dir / image_filename

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                image_list.append(
                    {
                        "page": page_num + 1,
                        "index": img_index + 1,
                        "filename": image_filename,
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to extract image on page {page_num + 1}: {e}")

    doc.close()
    return image_list


def generate_alt_text_markdown(
    pdf_path: Path | str,
    analysis: Dict[str, Any],
    images: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Generate a markdown file for alt text editing."""
    md_path = output_dir / "alt_text.md"
    pdf_name = Path(pdf_path).name

    lines = [
        f"# Alt Text for {pdf_name}",
        "",
        "Edit the alt text below each image. Keep the image reference line unchanged.",
        "Then run: `uv run pdf_access.py alt_text.md` to apply changes.",
        "",
        "---",
        "",
    ]

    # Build a map of which images have captions
    image_num = 0
    for page_data in analysis["pages"]:
        elements = page_data["elements"]
        for i, elem in enumerate(elements):
            if elem["type"] == "image" and image_num < len(images):
                img = images[image_num]
                caption = find_image_caption(elements, i)

                lines.append(f"## Image {image_num + 1} (Page {img['page']})")
                lines.append("")
                lines.append(f"![Image {image_num + 1}](images/{img['filename']})")
                lines.append("")

                if caption:
                    lines.append(f"**Auto-detected caption:** {caption}")
                    lines.append("")
                    lines.append(f"Alt text: {caption}")
                else:
                    lines.append("Alt text: [Describe this image]")

                lines.append("")
                lines.append("---")
                lines.append("")

                image_num += 1

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    # Also generate an HTML version for users without markdown viewers
    html_path = output_dir / "alt_text.html"
    _generate_alt_text_html(pdf_name, analysis, images, output_dir, html_path)

    return md_path


def _generate_alt_text_html(
    pdf_name: str,
    analysis: Dict[str, Any],
    images: List[Dict[str, Any]],
    output_dir: Path,
    html_path: Path,
) -> None:
    """Generate an HTML file showing images for alt text reference."""
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"    <title>Alt Text for {pdf_name}</title>",
        "    <style>",
        "        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }",
        "        h1 { color: #333; }",
        "        .image-card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; background: #fafafa; }",
        "        .image-card img { max-width: 100%; height: auto; border: 1px solid #ccc; }",
        "        .image-card h2 { margin-top: 0; color: #555; }",
        "        .caption { background: #e8f4e8; padding: 10px; border-radius: 4px; margin: 10px 0; }",
        "        .alt-text-field { width: 100%; padding: 10px; font-size: 14px; border: 1px solid #ccc; border-radius: 4px; margin-top: 10px; }",
        "        .instructions { background: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }",
        "        label { font-weight: bold; color: #333; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>Alt Text for {pdf_name}</h1>",
        "    <div class='instructions'>",
        "        <p><strong>Instructions:</strong> Review each image below and write a description in the corresponding field in <code>alt_text.md</code>.</p>",
        "        <p>This HTML file is for <em>reference only</em> - edit the markdown file to apply changes.</p>",
        "    </div>",
    ]

    image_num = 0
    for page_data in analysis["pages"]:
        elements = page_data["elements"]
        for i, elem in enumerate(elements):
            if elem["type"] == "image" and image_num < len(images):
                img = images[image_num]
                caption = find_image_caption(elements, i)

                html_lines.append(f"    <div class='image-card'>")
                html_lines.append(
                    f"        <h2>Image {image_num + 1} (Page {img['page']})</h2>"
                )
                html_lines.append(
                    f"        <img src='images/{img['filename']}' alt='Image {image_num + 1}'>"
                )

                if caption:
                    html_lines.append(
                        f"        <div class='caption'><strong>Auto-detected caption:</strong> {caption}</div>"
                    )

                html_lines.append(
                    f"        <label>Alt text to add in markdown:</label>"
                )
                placeholder = caption if caption else "[Describe this image]"
                html_lines.append(
                    f"        <input type='text' class='alt-text-field' value='{placeholder}' readonly>"
                )
                html_lines.append(f"    </div>")

                image_num += 1

    html_lines.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    with open(html_path, "w") as f:
        f.write("\n".join(html_lines))


def parse_alt_text_markdown(md_path: Path | str) -> Dict[str, Any]:
    """Parse the markdown file to extract alt text entries."""
    with open(md_path, "r") as f:
        content = f.read()

    # Extract PDF filename from title
    title_match = re.search(r"^# Alt Text for (.+)$", content, re.MULTILINE)
    if not title_match:
        raise ValueError("Could not find PDF filename in markdown")

    pdf_name = title_match.group(1)

    # Find all alt text entries
    # Pattern: ## Image N ... Alt text: <text>
    pattern = r"## Image (\d+).*?Alt text:\s*(.+?)(?=\n---|\n## Image|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)

    alt_texts = {}
    for match in matches:
        image_num = int(match[0])
        alt_text = match[1].strip()
        # Clean up the alt text (remove any trailing markdown)
        alt_text = alt_text.split("\n")[0].strip()
        if alt_text and alt_text != "[Describe this image]":
            alt_texts[image_num] = alt_text

    return {
        "pdf_name": pdf_name,
        "alt_texts": alt_texts,
    }


def apply_alt_text_to_pdf(pdf_path: Path | str, alt_texts: Dict[int, str]) -> None:
    """Apply alt text from markdown to the PDF structure."""
    with pikepdf.open(str(pdf_path), allow_overwriting_input=True) as pdf:
        if "/StructTreeRoot" not in pdf.Root:
            raise ValueError("PDF has no structure tree - run accessibility tool first")

        struct_tree = pdf.Root.StructTreeRoot
        doc_elem = struct_tree.K

        figure_num = 0
        for child in doc_elem.K:
            tag = str(child.get("/S", ""))
            if "Figure" in tag:
                figure_num += 1
                if figure_num in alt_texts:
                    child["/Alt"] = pikepdf.String(alt_texts[figure_num])

        pdf.save(str(pdf_path))


def boxes_overlap(box1, box2, tolerance=2.0):
    """Check if two bounding boxes overlap (with small tolerance for rounding)."""
    # box format: (x0, y0, x1, y1)
    return (
        box1[0] < box2[2] + tolerance
        and box1[2] > box2[0] - tolerance
        and box1[1] < box2[3] + tolerance
        and box1[3] > box2[1] - tolerance
    )


def generate_bookmarks_from_structure(pdf) -> int:
    """
    Generate bookmarks/outlines from heading structure elements.

    Walks the structure tree to find H1-H6 elements and creates
    corresponding bookmark entries.

    Returns the number of bookmarks created.
    """
    struct_tree = pdf.Root.get("/StructTreeRoot")
    if not struct_tree:
        return 0

    # Build page objgen to index mapping
    page_objgen_to_idx = {}
    for idx, page in enumerate(pdf.pages):
        page_objgen_to_idx[page.obj.objgen] = idx

    # Collect headings from structure tree
    headings = []  # List of (level, title, page_idx)

    def extract_text_from_elem(elem) -> str:
        """Try to extract text content from a structure element."""
        # Check for ActualText first
        actual_text = elem.get("/ActualText")
        if actual_text:
            return str(actual_text)

        # Check for Alt text
        alt = elem.get("/Alt")
        if alt:
            return str(alt)

        # No direct text available - would need to look at content stream
        return ""

    def find_headings(elem, parent_page_idx=None):
        """Recursively find heading elements."""
        if not hasattr(elem, "get"):
            return

        # Get page reference
        pg = elem.get("/Pg")
        if pg is not None:
            page_idx = page_objgen_to_idx.get(pg.objgen, parent_page_idx)
        else:
            page_idx = parent_page_idx

        # Check if this is a heading
        tag = elem.get("/S")
        if tag:
            tag_str = str(tag).replace("/", "")
            if tag_str in ("H1", "H2", "H3", "H4", "H5", "H6"):
                level = int(tag_str[1])
                title = extract_text_from_elem(elem)
                if not title:
                    title = f"Section ({tag_str})"
                # Truncate long titles
                if len(title) > 80:
                    title = title[:77] + "..."
                if page_idx is not None:
                    headings.append((level, title, page_idx))
            elif tag_str == "H":
                # Generic heading - treat as H1
                title = extract_text_from_elem(elem)
                if not title:
                    title = "Section"
                if page_idx is not None:
                    headings.append((1, title, page_idx))

        # Recurse into children
        k = elem.get("/K")
        if k is not None:
            if is_pdf_array(k):
                for child in k:
                    if hasattr(child, "get"):
                        find_headings(child, page_idx)
            elif hasattr(k, "get"):
                find_headings(k, page_idx)

    # Walk structure tree
    doc_elem = struct_tree.get("/K")
    if doc_elem:
        if is_pdf_array(doc_elem):
            for child in doc_elem:
                find_headings(child)
        else:
            find_headings(doc_elem)

    if not headings:
        return 0

    # Create bookmarks using pikepdf's outline API
    from pikepdf import OutlineItem

    with pdf.open_outline() as outline:
        # Clear existing outlines
        outline.root.clear()

        # Build hierarchical structure
        # Stack of (level, outline_item) for nesting
        stack = []

        for level, title, page_idx in headings:
            item = OutlineItem(title, page_idx)

            # Find the right parent based on level
            while stack and stack[-1][0] >= level:
                stack.pop()

            if stack:
                # Add as child of the most recent lower-level heading
                stack[-1][1].children.append(item)
            else:
                # Top-level item
                outline.root.append(item)

            stack.append((level, item))

    return len(headings)


def generate_bookmarks_from_analysis(pdf, analysis: Dict[str, Any]) -> int:
    """
    Generate bookmarks from content analysis (for newly tagged PDFs).

    Uses the heading detection from analyze_document() to create bookmarks.

    Returns the number of bookmarks created.
    """
    headings = []  # List of (level, title, page_idx)
    heading_sizes = analysis.get("heading_sizes", [])
    body_size = analysis.get("body_size", 12.0)

    for page_data in analysis["pages"]:
        page_idx = page_data["number"]
        for elem in page_data["elements"]:
            if elem["type"] == "text":
                tag = classify_element(elem, heading_sizes, body_size)
                if tag in ("H1", "H2", "H3"):
                    level = int(tag[1])
                    title = elem["text"]
                    if len(title) > 80:
                        title = title[:77] + "..."
                    headings.append((level, title, page_idx))

    if not headings:
        return 0

    from pikepdf import OutlineItem

    with pdf.open_outline() as outline:
        outline.root.clear()

        stack = []
        for level, title, page_idx in headings:
            item = OutlineItem(title, page_idx)

            while stack and stack[-1][0] >= level:
                stack.pop()

            if stack:
                stack[-1][1].children.append(item)
            else:
                outline.root.append(item)

            stack.append((level, item))

    return len(headings)


def inject_mcids_into_page(
    pdf,
    page,
    analysis_elements: List[Dict],
    heading_sizes: List[float],
    start_mcid: int = 0,
    links: Optional[List[Dict]] = None,
) -> tuple:
    """
    Inject BDC/EMC markers into a page's content stream.
    Returns (mcid_info_list, next_mcid) where mcid_info contains tag type for each MCID.
    mcid_info items may include 'link_index' if the text overlaps with a link annotation.
    mcid_info items may include 'table_cell' info for cells (table_idx, row_idx, cell_idx, is_header).

    Only BT...ET blocks with actual content (text operators like Tj, TJ, ', ") get MCIDs.
    Empty font-setup blocks are not tagged to avoid PDF/UA Content errors.
    Graphics painting operators (S, f, f*, etc.) outside marked content are marked as Artifacts.

    If the page already has marked content (BDC/BMC/EMC), it is stripped first to avoid
    duplicate tagging when re-processing an already-tagged PDF.
    """
    raw_instructions = list(pikepdf.parse_content_stream(page))

    # Step 0: Strip existing marked content operators (BDC, BMC, EMC)
    # This allows re-processing of already-tagged PDFs without duplicating markers
    instructions = []
    for operands, operator in raw_instructions:
        op = str(operator)
        if op in ("BDC", "BMC", "EMC"):
            # Skip existing marked content markers
            continue
        instructions.append((operands, operator))

    # First pass: identify which BT...ET blocks have actual text content
    # Text operators: Tj, TJ, ', "
    text_operators = {"Tj", "TJ", "'", '"'}
    bt_has_content = []  # True/False for each BT...ET block

    in_text_block = False
    current_block_has_content = False

    for operands, operator in instructions:
        op = str(operator)
        if op == "BT":
            in_text_block = True
            current_block_has_content = False
        elif op == "ET":
            bt_has_content.append(current_block_has_content)
            in_text_block = False
        elif in_text_block and op in text_operators:
            current_block_has_content = True

    # Graphics painting operators that draw visible content
    # These must be inside marked content (tagged or Artifact) for PDF/UA
    graphics_paint_operators = {"f", "F", "f*", "F*", "s", "S", "b", "B", "b*", "B*"}

    # Path construction operators (these build paths but don't paint)
    path_construction_operators = {"m", "l", "c", "v", "y", "h", "re"}

    # Second pass: inject MCIDs only for blocks with content
    new_instructions = []
    mcid = start_mcid
    mcid_info = []
    links = links or []

    # Track which analysis element we're on (text and table_cell blocks map to analysis)
    text_block_index = 0
    # Include both regular text and table cell text
    text_elements = [
        e for e in analysis_elements if e["type"] in ("text", "table_cell")
    ]
    image_index = 0
    image_elements = [e for e in analysis_elements if e["type"] == "image"]

    bt_block_index = 0  # Index into bt_has_content
    in_tagged_block = False  # Are we in a BT block that we tagged?
    marked_content_depth = 0  # Track nested BDC/BMC...EMC

    # Track path construction for artifact wrapping
    path_buffer = []  # Buffer path construction ops to wrap with paint op
    in_path = False

    for operands, operator in instructions:
        op = str(operator)

        # Track existing marked content nesting
        if op in ("BDC", "BMC"):
            marked_content_depth += 1
        elif op == "EMC":
            marked_content_depth -= 1

        if op == "BT":
            # Flush any pending path as artifact before text block
            if path_buffer:
                new_instructions.append(
                    ([pikepdf.Name("/Artifact")], pikepdf.Operator("BMC"))
                )
                new_instructions.extend(path_buffer)
                new_instructions.append(([], pikepdf.Operator("EMC")))
                path_buffer = []
                in_path = False

            # Check if this block has content
            has_content = (
                bt_block_index < len(bt_has_content) and bt_has_content[bt_block_index]
            )
            bt_block_index += 1

            if has_content:
                # Determine tag type from analysis
                link_index = None
                table_cell_info = None

                if text_block_index < len(text_elements):
                    elem = text_elements[text_block_index]

                    if elem["type"] == "table_cell":
                        # Table cell - use TH or TD based on is_header
                        tag = "TH" if elem.get("is_header") else "TD"
                        table_cell_info = {
                            "table_idx": elem["table_idx"],
                            "row_idx": elem["row_idx"],
                            "cell_idx": elem["cell_idx"],
                            "is_header": elem["is_header"],
                        }
                    else:
                        # Regular text - classify as heading or paragraph
                        tag = classify_element(elem, heading_sizes)

                    # Check if this text overlaps with any link annotation
                    text_bbox = elem.get("bbox")
                    if text_bbox:
                        for idx, link in enumerate(links):
                            link_bbox = link.get("bbox")
                            if link_bbox and boxes_overlap(text_bbox, link_bbox):
                                link_index = idx
                                break

                    text_block_index += 1
                else:
                    tag = "P"

                bdc_operands = [
                    pikepdf.Name(f"/{tag}"),
                    pikepdf.Dictionary({"/MCID": mcid}),
                ]
                new_instructions.append((bdc_operands, pikepdf.Operator("BDC")))
                info = {"type": "text", "tag": tag, "mcid": mcid}
                if link_index is not None:
                    info["link_index"] = link_index
                if table_cell_info is not None:
                    info["table_cell"] = table_cell_info
                mcid_info.append(info)
                in_tagged_block = True
            else:
                in_tagged_block = False

        elif op == "Do":
            # Flush any pending path as artifact before image
            if path_buffer:
                new_instructions.append(
                    ([pikepdf.Name("/Artifact")], pikepdf.Operator("BMC"))
                )
                new_instructions.extend(path_buffer)
                new_instructions.append(([], pikepdf.Operator("EMC")))
                path_buffer = []
                in_path = False

            # Image/XObject
            if image_index < len(image_elements):
                image_index += 1

            bdc_operands = [
                pikepdf.Name("/Figure"),
                pikepdf.Dictionary({"/MCID": mcid}),
            ]
            new_instructions.append((bdc_operands, pikepdf.Operator("BDC")))
            mcid_info.append({"type": "image", "tag": "Figure", "mcid": mcid})
            new_instructions.append((operands, operator))
            new_instructions.append(([], pikepdf.Operator("EMC")))
            mcid += 1
            continue

        elif op in path_construction_operators:
            # Start or continue path construction
            # Only buffer if we're not already inside marked content
            if marked_content_depth == 0 and not in_tagged_block:
                in_path = True
                path_buffer.append((operands, operator))
                continue
            # Otherwise, just add normally (already in marked content)

        elif op in graphics_paint_operators:
            # Paint operator - completes a path
            if marked_content_depth == 0 and not in_tagged_block:
                # Not in any marked content - wrap path + paint as Artifact
                new_instructions.append(
                    ([pikepdf.Name("/Artifact")], pikepdf.Operator("BMC"))
                )
                new_instructions.extend(path_buffer)
                new_instructions.append((operands, operator))
                new_instructions.append(([], pikepdf.Operator("EMC")))
                path_buffer = []
                in_path = False
                continue
            else:
                # Inside marked content - flush buffer and add paint op normally
                new_instructions.extend(path_buffer)
                path_buffer = []
                in_path = False

        elif op == "n":
            # End path without painting - not visible, no artifact needed
            # But we need to flush any buffered path ops
            new_instructions.extend(path_buffer)
            path_buffer = []
            in_path = False

        # Flush path buffer if we hit a non-path operator (except graphics state)
        # Graphics state operators include:
        # - q, Q: save/restore graphics state
        # - cm: transformation matrix
        # - w, J, j, M, d: line width, cap, join, miter limit, dash
        # - ri, i, gs: rendering intent, flatness, graphics state dict
        # - CS, cs, SC, SCN, sc, scn: color space and color
        # - G, g, RG, rg, K, k: gray, RGB, CMYK colors
        graphics_state_ops = {
            "q",
            "Q",
            "cm",
            "w",
            "J",
            "j",
            "M",
            "d",
            "ri",
            "i",
            "gs",
            "CS",
            "cs",
            "SC",
            "SCN",
            "sc",
            "scn",
            "G",
            "g",
            "RG",
            "rg",
            "K",
            "k",
        }
        if (
            in_path
            and op not in path_construction_operators
            and op not in graphics_paint_operators
            and op not in graphics_state_ops
            and op != "n"
        ):
            # Non-path operator while building path - flush buffer
            new_instructions.extend(path_buffer)
            path_buffer = []
            in_path = False

        new_instructions.append((operands, operator))

        if op == "ET" and in_tagged_block:
            new_instructions.append(([], pikepdf.Operator("EMC")))
            mcid += 1
            in_tagged_block = False  # Reset after closing the tag

    # Flush any remaining path buffer at end (shouldn't happen normally)
    if path_buffer:
        new_instructions.append(([pikepdf.Name("/Artifact")], pikepdf.Operator("BMC")))
        new_instructions.extend(path_buffer)
        new_instructions.append(([], pikepdf.Operator("EMC")))

    # Write new content stream
    new_content = pikepdf.unparse_content_stream(new_instructions)
    page.Contents = pdf.make_stream(new_content)

    return mcid_info, mcid


def remediate_pdf(
    input_path: Path | str,
    output_path: Path | str,
    analysis: Dict[str, Any],
    alt_texts: Optional[Dict[int, str]] = None,
    preserve_structure: bool = False,
    diagnostic: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Remediate PDF accessibility issues based on analysis and diagnostic.

    Fixes include: language, title, display title, structure tree, MCIDs,
    ParentTree, tab order, and link tagging.

    Args:
        input_path: Path to source PDF
        output_path: Path for output PDF
        analysis: Content analysis from analyze_document()
        alt_texts: Optional dict mapping figure number to alt text
        preserve_structure: If True, enhance existing structure without rebuilding
        diagnostic: Results from gather_accessibility_info() - used to determine what needs fixing

    Returns the document title.
    """
    # Extract flags from diagnostic if provided, otherwise we'll check directly
    flags = diagnostic.get("flags", {}) if diagnostic else {}
    has_lang = flags.get("language", False)
    has_title = flags.get("title", False)
    has_display_title = flags.get("display_title", False)
    has_tagged = flags.get("tagged", False)

    # Check for bookmarks from diagnostic
    bookmark_info = diagnostic.get("bookmarks", {}) if diagnostic else {}
    has_bookmarks = bookmark_info.get("has_bookmarks", False)

    with pikepdf.open(str(input_path)) as pdf:
        # Set document language (only if diagnostic says it's missing)
        if not has_lang:
            pdf.Root.Lang = pikepdf.String("en-US")
            lang_to_use = "en-US"
        else:
            lang_to_use = str(pdf.Root.get("/Lang", "en-US"))

        # Set ViewerPreferences (only if diagnostic says display title is disabled)
        if not has_display_title:
            if "/ViewerPreferences" not in pdf.Root:
                pdf.Root.ViewerPreferences = pikepdf.Dictionary()
            pdf.Root.ViewerPreferences.DisplayDocTitle = True

        # Set MarkInfo (only if not already tagged or if we're rebuilding)
        if not has_tagged or not preserve_structure:
            pdf.Root.MarkInfo = pikepdf.Dictionary(
                {
                    "/Marked": True,
                    "/Suspects": False,
                }
            )

        # Determine title - use existing if valid, otherwise find from first H1
        if has_title:
            # Get existing title from PDF
            existing_title = None
            if pdf.trailer.get("/Info"):
                existing_title = pdf.trailer.Info.get("/Title")
            title = str(existing_title) if existing_title else Path(input_path).stem
        else:
            # Find title from content
            title = Path(input_path).stem
            for page_data in analysis["pages"]:
                for elem in page_data["elements"]:
                    if elem["type"] == "text":
                        tag = classify_element(elem, analysis["heading_sizes"])
                        if tag == "H1":
                            title = elem["text"][:100]
                            break
                if title != Path(input_path).stem:
                    break

        # Set title metadata and PDF/UA identifier
        # Always set PDF/UA identifier for compliance
        with pdf.open_metadata() as meta:
            # Always set title in XMP (pikepdf may reset metadata on save)
            meta["dc:title"] = title
            meta["dc:language"] = lang_to_use
            # Add PDF/UA identifier (required for PDF/UA compliance)
            meta["pdfuaid:part"] = "1"
            # Also add PDF/A identifier for better compatibility
            meta["pdfaid:part"] = "3"
            meta["pdfaid:conformance"] = "A"

        # Always ensure title is set in Info dict (pikepdf may reset it on save)
        if pdf.trailer.get("/Info"):
            pdf.trailer.Info["/Title"] = pikepdf.String(title)
            pdf.trailer.Info["/Producer"] = pikepdf.String("PDF Accessibility Tool")
            pdf.trailer.Info["/Creator"] = pikepdf.String("PDF Accessibility Tool")
        else:
            pdf.trailer.Info = pikepdf.Dictionary(
                {
                    "/Title": pikepdf.String(title),
                    "/Producer": pikepdf.String("PDF Accessibility Tool"),
                    "/Creator": pikepdf.String("PDF Accessibility Tool"),
                }
            )

        # If preserving structure, enhance existing tags without rebuilding
        # This adds missing ParentTree, StructParents, Tabs, RoleMap
        if preserve_structure:
            struct_tree = pdf.Root.get("/StructTreeRoot")

            # Build page objgen to index mapping (objgen is reliable for comparison)
            page_objgen_to_idx = {}
            for idx, page in enumerate(pdf.pages):
                page_objgen_to_idx[page.obj.objgen] = idx

            # Collect MCIDs per page from existing structure
            # Format: page_idx -> list of (mcid, struct_elem) in MCID order
            page_mcids: Dict[int, List[tuple]] = {i: [] for i in range(len(pdf.pages))}

            def collect_mcids_from_struct(elem, parent_page_idx=None):
                """Recursively collect MCIDs from structure elements."""
                if not hasattr(elem, "get"):
                    return

                # Get page reference from this element or inherit from parent
                pg = elem.get("/Pg")
                if pg is not None:
                    page_idx = page_objgen_to_idx.get(pg.objgen, parent_page_idx)
                else:
                    page_idx = parent_page_idx

                k = elem.get("/K")
                if k is not None:
                    # K can be: int (MCID), dict, or array
                    if is_mcid_value(k):
                        if page_idx is not None:
                            page_mcids[page_idx].append((int(k), elem))
                    elif is_pdf_array(k):
                        for item in k:
                            if is_mcid_value(item):
                                if page_idx is not None:
                                    page_mcids[page_idx].append((int(item), elem))
                            elif hasattr(item, "get"):
                                mcid = item.get("/MCID")
                                if mcid is not None:
                                    if page_idx is not None:
                                        page_mcids[page_idx].append((int(mcid), elem))
                                elif item.get("/S"):  # Child struct elem
                                    collect_mcids_from_struct(item, page_idx)
                    elif hasattr(k, "get"):
                        mcid = k.get("/MCID")
                        if mcid is not None:
                            if page_idx is not None:
                                page_mcids[page_idx].append((int(mcid), elem))
                        elif k.get("/S"):  # Child struct elem
                            collect_mcids_from_struct(k, page_idx)

            # Walk structure tree to collect MCIDs
            if struct_tree:
                doc_elem = struct_tree.get("/K")
                if doc_elem:
                    if is_pdf_array(doc_elem):
                        for child in doc_elem:
                            collect_mcids_from_struct(child)
                    else:
                        collect_mcids_from_struct(doc_elem)

            # Build ParentTree from collected MCIDs
            parent_tree_nums = []
            for page_idx in range(len(pdf.pages)):
                mcid_list = page_mcids[page_idx]
                if mcid_list:
                    # Sort by MCID to ensure correct order
                    mcid_list.sort(key=lambda x: x[0])
                    # Create array of struct elements in MCID order
                    struct_elem_array = pikepdf.Array([elem for _, elem in mcid_list])
                    parent_tree_nums.append(page_idx)
                    parent_tree_nums.append(struct_elem_array)

            # Add or update ParentTree
            if struct_tree and parent_tree_nums:
                parent_tree = pdf.make_indirect(
                    pikepdf.Dictionary({"/Nums": pikepdf.Array(parent_tree_nums)})
                )
                struct_tree["/ParentTree"] = parent_tree
                struct_tree["/ParentTreeNextKey"] = len(pdf.pages)

            # Add empty RoleMap if missing - we use standard tags only
            # Standard PDF 1.7 tags don't need role mapping
            if struct_tree and not struct_tree.get("/RoleMap"):
                struct_tree["/RoleMap"] = pikepdf.Dictionary({})

            # Set Tabs and StructParents on each page
            struct_parent_idx = len(pdf.pages)  # Start after page indices
            for page_num, page in enumerate(pdf.pages):
                if page.get("/Tabs") != pikepdf.Name("/S"):
                    page.Tabs = pikepdf.Name("/S")
                if page.get("/StructParents") is None:
                    page.StructParents = page_num

                # Handle annotations (links, etc.)
                annots = page.get("/Annots")
                if annots:
                    for annot in annots:
                        if annot.get("/StructParent") is None:
                            annot["/StructParent"] = struct_parent_idx
                            struct_parent_idx += 1

            # Generate bookmarks from existing structure if missing
            if not has_bookmarks:
                generate_bookmarks_from_structure(pdf)

            pdf.save(str(output_path))
            return title

        # Process each page - inject MCIDs into content streams
        all_struct_elems = []
        parent_tree_nums = []
        figure_count = 0

        for page_num, page in enumerate(pdf.pages):
            page_analysis = (
                analysis["pages"][page_num]
                if page_num < len(analysis["pages"])
                else {"elements": []}
            )

            # Get links from analysis for this page
            page_links = page_analysis.get("links", [])

            # Inject MCIDs into this page's content stream
            mcid_info, _ = inject_mcids_into_page(
                pdf,
                page,
                page_analysis["elements"],
                analysis["heading_sizes"],
                start_mcid=0,  # MCIDs reset per page
                links=page_links,
            )

            # Create structure elements for this page
            # First, organize table cell MCIDs by table/row/cell
            table_cell_mcids = {}  # (table_idx, row_idx, cell_idx) -> list of mcid_info
            non_table_mcid_info = []

            for info in mcid_info:
                if "table_cell" in info:
                    tc = info["table_cell"]
                    key = (tc["table_idx"], tc["row_idx"], tc["cell_idx"])
                    if key not in table_cell_mcids:
                        table_cell_mcids[key] = []
                    table_cell_mcids[key].append(info)
                else:
                    non_table_mcid_info.append(info)

            # Map MCID -> struct elem for ParentTree
            mcid_to_struct_elem = {}

            # Track page-local figure count for bbox lookup
            page_figure_count = 0

            # Create structure elements for non-table content
            for info in non_table_mcid_info:
                tag = info["tag"]
                mcid = info["mcid"]

                struct_elem = pdf.make_indirect(
                    pikepdf.Dictionary(
                        {
                            "/Type": pikepdf.Name("/StructElem"),
                            "/S": pikepdf.Name(f"/{tag}"),
                            "/Pg": page.obj,
                            "/K": mcid,
                        }
                    )
                )

                # Add alt text and BBox for figures
                if tag == "Figure":
                    figure_count += 1  # Global count for alt_texts mapping
                    page_figure_count += 1  # Page-local count for bbox lookup

                    # Get image bbox from analysis for BBox attribute (required by PDF/UA)
                    elements = page_analysis["elements"]
                    img_elements = [e for e in elements if e["type"] == "image"]
                    img_idx = page_figure_count - 1  # Use page-local index
                    if img_idx < len(img_elements):
                        img_elem = img_elements[img_idx]
                        bbox = img_elem.get("bbox")
                        if bbox:
                            # BBox is [x1, y1, x2, y2] - left, bottom, right, top
                            struct_elem["/A"] = pikepdf.Dictionary(
                                {
                                    "/O": pikepdf.Name("/Layout"),
                                    "/BBox": pikepdf.Array(
                                        [
                                            float(bbox[0]),  # x1 (left)
                                            float(bbox[1]),  # y1 (bottom)
                                            float(bbox[2]),  # x2 (right)
                                            float(bbox[3]),  # y2 (top)
                                        ]
                                    ),
                                }
                            )

                    # Add alt text
                    if alt_texts and figure_count in alt_texts:
                        struct_elem["/Alt"] = pikepdf.String(alt_texts[figure_count])
                    else:
                        # Try to find caption from analysis
                        img_indices = [
                            i for i, e in enumerate(elements) if e["type"] == "image"
                        ]
                        if img_idx < len(img_indices):
                            caption = find_image_caption(elements, img_indices[img_idx])
                            if caption:
                                struct_elem["/Alt"] = pikepdf.String(caption)
                            else:
                                struct_elem["/Alt"] = pikepdf.String(
                                    f"Image {figure_count} - alt text needed"
                                )

                mcid_to_struct_elem[mcid] = struct_elem
                all_struct_elems.append(struct_elem)

            # Create table structure elements with proper MCID linking
            # PDF/UA requires: Table → THead → TR → TH and Table → TBody → TR → TD
            for table_idx, table_data in enumerate(page_analysis.get("tables", [])):
                header_rows = []  # TRs for THead
                body_rows = []  # TRs for TBody

                for row_idx, row_data in enumerate(table_data["rows"]):
                    row_cells = []
                    for cell_idx in range(len(row_data["cells"])):
                        # Get MCIDs for this cell
                        key = (table_idx, row_idx, cell_idx)
                        cell_mcid_infos = table_cell_mcids.get(key, [])

                        # Create TH for header row, TD for data rows
                        cell_tag = "TH" if row_data["is_header"] else "TD"

                        # Build /K value - single MCID or array of MCIDs
                        if len(cell_mcid_infos) == 0:
                            # No content found - use ActualText as fallback
                            cell_content = (
                                str(row_data["cells"][cell_idx])
                                if cell_idx < len(row_data["cells"])
                                else ""
                            )
                            cell_dict = {
                                "/Type": pikepdf.Name("/StructElem"),
                                "/S": pikepdf.Name(f"/{cell_tag}"),
                                "/Pg": page.obj,
                                "/ActualText": pikepdf.String(cell_content),
                            }
                        elif len(cell_mcid_infos) == 1:
                            # Single MCID
                            cell_dict = {
                                "/Type": pikepdf.Name("/StructElem"),
                                "/S": pikepdf.Name(f"/{cell_tag}"),
                                "/Pg": page.obj,
                                "/K": cell_mcid_infos[0]["mcid"],
                            }
                        else:
                            # Multiple MCIDs in this cell
                            mcid_array = pikepdf.Array(
                                [info["mcid"] for info in cell_mcid_infos]
                            )
                            cell_dict = {
                                "/Type": pikepdf.Name("/StructElem"),
                                "/S": pikepdf.Name(f"/{cell_tag}"),
                                "/Pg": page.obj,
                                "/K": mcid_array,
                            }

                        # For TH, add Scope attribute (Column scope)
                        if row_data["is_header"]:
                            cell_dict["/A"] = pikepdf.Dictionary(
                                {
                                    "/O": pikepdf.Name("/Table"),
                                    "/Scope": pikepdf.Name("/Column"),
                                }
                            )

                        cell_elem = pdf.make_indirect(pikepdf.Dictionary(cell_dict))
                        row_cells.append(cell_elem)

                        # Track cell struct elem for ParentTree mapping
                        if cell_mcid_infos:
                            for info in cell_mcid_infos:
                                mcid_to_struct_elem[info["mcid"]] = cell_elem

                    # Create TR (table row) containing the cells
                    tr_elem = pdf.make_indirect(
                        pikepdf.Dictionary(
                            {
                                "/Type": pikepdf.Name("/StructElem"),
                                "/S": pikepdf.Name("/TR"),
                                "/Pg": page.obj,
                                "/K": pikepdf.Array(row_cells),
                            }
                        )
                    )
                    # Set parent reference for cells (will update to THead/TBody later)
                    for cell in row_cells:
                        cell["/P"] = tr_elem

                    # Add to header or body rows
                    if row_data["is_header"]:
                        header_rows.append(tr_elem)
                    else:
                        body_rows.append(tr_elem)

                # Create Table element with THead and TBody children
                if header_rows or body_rows:
                    # Generate table summary from header row
                    header_cells = (
                        table_data["rows"][0]["cells"] if table_data["rows"] else []
                    )
                    if header_cells:
                        summary = f"Table with {table_data['row_count']} rows and {table_data['col_count']} columns. Columns: {', '.join(str(c) for c in header_cells if c)}"
                    else:
                        summary = f"Table with {table_data['row_count']} rows and {table_data['col_count']} columns"

                    table_children = []

                    # Create THead if there are header rows
                    if header_rows:
                        thead_elem = pdf.make_indirect(
                            pikepdf.Dictionary(
                                {
                                    "/Type": pikepdf.Name("/StructElem"),
                                    "/S": pikepdf.Name("/THead"),
                                    "/Pg": page.obj,
                                    "/K": pikepdf.Array(header_rows),
                                }
                            )
                        )
                        # Set parent reference for header TRs
                        for tr in header_rows:
                            tr["/P"] = thead_elem
                        table_children.append(thead_elem)

                    # Create TBody if there are body rows
                    if body_rows:
                        tbody_elem = pdf.make_indirect(
                            pikepdf.Dictionary(
                                {
                                    "/Type": pikepdf.Name("/StructElem"),
                                    "/S": pikepdf.Name("/TBody"),
                                    "/Pg": page.obj,
                                    "/K": pikepdf.Array(body_rows),
                                }
                            )
                        )
                        # Set parent reference for body TRs
                        for tr in body_rows:
                            tr["/P"] = tbody_elem
                        table_children.append(tbody_elem)

                    table_elem = pdf.make_indirect(
                        pikepdf.Dictionary(
                            {
                                "/Type": pikepdf.Name("/StructElem"),
                                "/S": pikepdf.Name("/Table"),
                                "/Pg": page.obj,
                                "/K": pikepdf.Array(table_children),
                                "/Summary": pikepdf.String(summary),
                            }
                        )
                    )
                    # Set parent reference for THead/TBody
                    for child in table_children:
                        child["/P"] = table_elem

                    all_struct_elems.append(table_elem)

            # Build ParentTree array: index = MCID, value = struct elem ref
            max_mcid = max((info["mcid"] for info in mcid_info), default=-1)
            parent_tree_array = [None] * (max_mcid + 1)

            # Fill in from our mcid_to_struct_elem mapping
            for mcid, struct_elem in mcid_to_struct_elem.items():
                if mcid < len(parent_tree_array):
                    parent_tree_array[mcid] = struct_elem

            # Add to ParentTree: page_num -> array of struct elems by MCID order
            valid_entries = [e for e in parent_tree_array if e is not None]
            if valid_entries:
                parent_tree_nums.append(page_num)
                parent_tree_nums.append(pikepdf.Array(parent_tree_array))

            # Set tab order to follow structure (required for PDF/UA)
            page.Tabs = pikepdf.Name("/S")
            # Link page to its ParentTree entry
            page.StructParents = page_num

            # Build map from link_index to MCID and struct elem for this page
            link_mcid_map = {}  # link_index -> (mcid, struct_elem)
            for info in mcid_info:
                if "link_index" in info:
                    link_mcid_map[info["link_index"]] = (
                        info["mcid"],
                        mcid_to_struct_elem.get(info["mcid"]),
                    )

            # Get page height for coordinate conversion (PDF Y is from bottom)
            page_mediabox = page.get("/MediaBox")
            if page_mediabox:
                page_height = float(page_mediabox[3]) - float(page_mediabox[1])
            else:
                page_height = 842.0  # Default A4 height

            # Handle annotations (links, form fields, etc.)
            annots = page.get("/Annots")
            if annots:
                for annot_idx, annot in enumerate(annots):
                    # Calculate unique StructParent index for this annotation
                    # Pages use indices 0 to len(pages)-1, annotations start after
                    annot_struct_parent = len(pdf.pages) + len(all_struct_elems)

                    # Create a Link structure element for this annotation
                    annot_subtype = str(annot.get("/Subtype", ""))
                    if "Link" in annot_subtype:
                        # Convert pikepdf Rect to PyMuPDF coordinates to match page_links
                        annot_rect = annot.get("/Rect")
                        if annot_rect:
                            # PDF coords: (x0, y0_bottom, x1, y1_bottom)
                            # PyMuPDF coords: (x0, y0_top, x1, y1_top)
                            annot_bbox_pymupdf = (
                                float(annot_rect[0]),
                                page_height - float(annot_rect[3]),
                                float(annot_rect[2]),
                                page_height - float(annot_rect[1]),
                            )
                        else:
                            annot_bbox_pymupdf = None

                        # Find matching link_index from page_links by bbox overlap
                        matched_link_index = None
                        if annot_bbox_pymupdf:
                            for link_idx, page_link in enumerate(page_links):
                                link_bbox = page_link.get("bbox")
                                if link_bbox and boxes_overlap(
                                    annot_bbox_pymupdf, link_bbox, tolerance=5.0
                                ):
                                    matched_link_index = link_idx
                                    break

                        # Get MCID for link text if we found a match
                        link_text_mcid = None
                        if (
                            matched_link_index is not None
                            and matched_link_index in link_mcid_map
                        ):
                            link_text_mcid, _ = link_mcid_map[matched_link_index]

                        # Create Link structure with both MCID (text) and OBJR (annotation)
                        objr = pikepdf.Dictionary(
                            {
                                "/Type": pikepdf.Name("/OBJR"),
                                "/Obj": annot,
                            }
                        )

                        if link_text_mcid is not None:
                            # Include both the text MCID and the annotation OBJR
                            k_value = pikepdf.Array([link_text_mcid, objr])
                        else:
                            # Fallback: only OBJR (no matching text found)
                            k_value = objr

                        link_elem = pdf.make_indirect(
                            pikepdf.Dictionary(
                                {
                                    "/Type": pikepdf.Name("/StructElem"),
                                    "/S": pikepdf.Name("/Link"),
                                    "/Pg": page.obj,
                                    "/K": k_value,
                                }
                            )
                        )
                        all_struct_elems.append(link_elem)

                        # Add to ParentTree for this annotation
                        parent_tree_nums.append(annot_struct_parent)
                        parent_tree_nums.append(link_elem)

                        # Set StructParent on the annotation
                        annot["/StructParent"] = annot_struct_parent

        # Create Document element
        doc_elem = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/StructElem"),
                    "/S": pikepdf.Name("/Document"),
                    "/K": pikepdf.Array(all_struct_elems),
                }
            )
        )

        # Set parent references for all struct elems
        for elem in all_struct_elems:
            elem["/P"] = doc_elem

        # Create ParentTree
        parent_tree = pdf.make_indirect(
            pikepdf.Dictionary({"/Nums": pikepdf.Array(parent_tree_nums)})
        )

        # Create RoleMap - only needed for custom (non-standard) tags
        # Standard PDF 1.7 tags (Document, H1-H6, P, Figure, Table, TR, TH, TD, etc.)
        # do NOT need role mapping - they're already recognized.
        # An empty RoleMap is valid and preferred when using only standard tags.
        role_map = pdf.make_indirect(pikepdf.Dictionary({}))

        # Create StructTreeRoot
        struct_tree_root = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/StructTreeRoot"),
                    "/K": doc_elem,
                    "/ParentTree": parent_tree,
                    "/ParentTreeNextKey": len(pdf.pages),
                    "/RoleMap": role_map,
                }
            )
        )

        doc_elem["/P"] = struct_tree_root
        pdf.Root.StructTreeRoot = struct_tree_root

        # Generate bookmarks from content analysis if missing
        if not has_bookmarks:
            generate_bookmarks_from_analysis(pdf, analysis)

        pdf.save(str(output_path))

        return title


def save_check_report(results: Dict[str, Any], report_path: Path, label: str) -> None:
    """Save accessibility check results to a file."""
    report = format_accessibility_report(results)
    with open(report_path, "w") as f:
        f.write(f"# Accessibility Report ({label})\n\n")
        f.write(f"```\n{report}\n```\n")


def process_pdf(input_path: Path, force: bool = False) -> int:
    """Process a PDF file to make it accessible."""
    output_path = input_path.parent / f"{input_path.stem}_accessible.pdf"
    review_dir = input_path.parent / f"{input_path.stem}_review"

    # Run pre-check
    print("Checking current accessibility status...")
    pre_results = gather_accessibility_info(input_path)
    flags = pre_results.get("flags", {})

    # Extract what's already good from pre-check
    has_mcids = flags.get("mcids", False)
    has_tagged = flags.get("tagged", False)
    has_lang = flags.get("language", False)
    has_title = flags.get("title", False)
    has_display_title = flags.get("display_title", False)

    # Preserve existing structure if PDF is already tagged
    # This avoids destroying well-structured content from apps like Pages, Word, etc.
    # Use --force to override and rebuild from scratch
    preserve_structure = has_tagged and not force

    # Report current status
    if pre_results["summary"]["failed"] == 0 and pre_results["summary"]["warned"] == 0:
        print("  ✓ PDF is already fully accessible!")
    else:
        if preserve_structure:
            print("  ✓ PDF is tagged (preserving existing structure)")
        if has_lang:
            print("  ✓ Has language set")
        if has_title:
            print("  ✓ Has title set")
        print(f"  - Found {pre_results['summary']['failed']} issue(s) to fix")
    print()

    print("Analyzing document structure...")
    try:
        analysis = analyze_document(str(input_path))
    except Exception as e:
        print(f"Error analyzing document: {e}")
        return 1

    print("Applying accessibility fixes...")
    if not preserve_structure:
        print("  - Adding document tags with MCIDs")
    if not has_lang:
        print("  - Setting document language (en-US)")

    try:
        title = remediate_pdf(
            str(input_path),
            str(output_path),
            analysis,
            preserve_structure=preserve_structure,
            diagnostic=pre_results,
        )
    except Exception as e:
        print(f"Error making document accessible: {e}")
        return 1

    if not has_title:
        print(
            f"  - Setting document title: {title[:50]}{'...' if len(title) > 50 else ''}"
        )
    if not has_display_title:
        print("  - Enabling display document title")
    print(f"Saving: {output_path}")

    # Report table detection
    total_tables = sum(len(p.get("tables", [])) for p in analysis["pages"])
    if total_tables > 0:
        print(f"  - Found {total_tables} table(s) (auto-tagged with headers)")

    # Handle images
    total_images = len(analysis["images"])
    if total_images > 0:
        # Count images with captions
        captioned = 0
        for page_data in analysis["pages"]:
            elements = page_data["elements"]
            for i, elem in enumerate(elements):
                if elem["type"] == "image":
                    if find_image_caption(elements, i):
                        captioned += 1

        needs_review = total_images - captioned

        if captioned > 0:
            print(f"  - Found {captioned} image(s) with captions (auto-tagged)")

        if needs_review > 0:
            print(f"  - Found {needs_review} image(s) needing alt text")
            review_dir.mkdir(exist_ok=True)
            images = extract_images(str(input_path), review_dir)
            md_path = generate_alt_text_markdown(
                str(input_path), analysis, images, review_dir
            )
    else:
        needs_review = 0

    # Ensure review_dir exists for reports
    review_dir.mkdir(exist_ok=True)

    # Run post-check on the new file
    print()
    print("Verifying accessibility fixes...")
    post_results = gather_accessibility_info(output_path)

    # Save both reports
    pre_report_path = review_dir / "check_before.md"
    post_report_path = review_dir / "check_after.md"
    save_check_report(pre_results, pre_report_path, "Before Processing")
    save_check_report(post_results, post_report_path, "After Processing")

    # Print summary
    pre_failed = pre_results["summary"]["failed"]
    post_failed = post_results["summary"]["failed"]
    post_warned = post_results["summary"]["warned"]
    post_passed = post_results["summary"]["passed"]

    print(f"  - Before: {pre_failed} failed")
    print(
        f"  - After:  {post_passed} passed, {post_warned} warnings, {post_failed} failed"
    )
    print()

    print(f"Done! Accessible PDF saved to:\n  {output_path}")
    print()
    print(f"Accessibility reports saved to:")
    print(f"  - {pre_report_path}")
    print(f"  - {post_report_path}")

    if needs_review > 0:
        md_path = review_dir / "alt_text.md"
        html_path = review_dir / "alt_text.html"
        print()
        print(f"Images need alt text review:")
        print(f"  - View images: {html_path}")
        print(f"  - Edit alt text: {md_path}")
        print(f"  - Then run: uv run pdf_access.py {md_path}")
    elif total_images > 0:
        print()
        print("All images have captions - no manual alt text needed!")

    return 0


def gather_accessibility_info(pdf_path: Path) -> Dict[str, Any]:
    """Gather accessibility information from a PDF without printing."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results: Dict[str, Any] = {
        "filename": pdf_path.name,
        "filepath": str(pdf_path),
        "timestamp": timestamp,
        "pages": 0,
        "checks": [],  # List of (status, category, message) tuples
        "tag_counts": {},
        "summary": {"passed": 0, "warned": 0, "failed": 0},
        "flags": {},  # Initialize flags once here
    }

    def add_check(status: str, category: str, message: str):
        results["checks"].append((status, category, message))
        if status == "pass":
            results["summary"]["passed"] += 1
        elif status == "warn":
            results["summary"]["warned"] += 1
        elif status == "fail":
            results["summary"]["failed"] += 1
        # "info" status doesn't affect pass/warn/fail counts

    # Open with both libraries
    try:
        pdf = pikepdf.open(str(pdf_path))
    except Exception as e:
        results["error"] = str(e)
        return results

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        pdf.close()
        results["error"] = str(e)
        return results

    results["pages"] = len(doc)

    # === Document Settings ===
    mark_info = pdf.Root.get("/MarkInfo")
    if mark_info and mark_info.get("/Marked"):
        add_check("pass", "Document Settings", "Tagged PDF: Yes")
        results["flags"]["tagged"] = True
    else:
        add_check("fail", "Document Settings", "Tagged PDF: No")
        results["flags"]["tagged"] = False

    lang = pdf.Root.get("/Lang")
    if lang:
        add_check("pass", "Document Settings", f"Language: {lang}")
        results["flags"]["language"] = True
    else:
        add_check("fail", "Document Settings", "Language: Not set")
        results["flags"]["language"] = False

    title = None
    if pdf.trailer.get("/Info"):
        title = pdf.trailer.Info.get("/Title")
    if title and str(title).strip():
        title_str = str(title)
        if len(title_str) > 50:
            title_str = title_str[:50] + "..."
        add_check("pass", "Document Settings", f'Title: "{title_str}"')
        results["flags"]["title"] = True
    else:
        add_check("fail", "Document Settings", "Title: Not set")
        results["flags"]["title"] = False

    viewer_prefs = pdf.Root.get("/ViewerPreferences")
    if viewer_prefs and viewer_prefs.get("/DisplayDocTitle"):
        add_check("pass", "Document Settings", "Display Doc Title: Enabled")
        results["flags"]["display_title"] = True
    else:
        add_check("fail", "Document Settings", "Display Doc Title: Disabled")
        results["flags"]["display_title"] = False

    # === Fonts ===
    # PDF/UA requires:
    # 1. All fonts must be embedded (or be one of the 14 standard fonts with proper encoding)
    # 2. Fonts must have ToUnicode CMap or be symbolic fonts
    # 3. No Type3 fonts (generally)
    font_issues = []
    fonts_by_name: Dict[str, Dict] = {}  # Track unique fonts

    # Standard Type1 fonts that don't need embedding (but need proper encoding)
    STANDARD_14_FONTS = {
        "Courier",
        "Courier-Bold",
        "Courier-Oblique",
        "Courier-BoldOblique",
        "Helvetica",
        "Helvetica-Bold",
        "Helvetica-Oblique",
        "Helvetica-BoldOblique",
        "Times-Roman",
        "Times-Bold",
        "Times-Italic",
        "Times-BoldItalic",
        "Symbol",
        "ZapfDingbats",
    }

    # Symbolic fonts that don't need ToUnicode
    SYMBOLIC_FONTS = {"Symbol", "ZapfDingbats"}

    for page_num, page in enumerate(pdf.pages):
        resources = page.get("/Resources")
        if not resources:
            continue
        fonts = resources.get("/Font")
        if not fonts:
            continue

        for font_name, font_ref in fonts.items():
            try:
                font = font_ref
                base_font = str(font.get("/BaseFont", "")).lstrip("/")
                subtype = str(font.get("/Subtype", "")).lstrip("/")
                encoding = font.get("/Encoding")
                to_unicode = font.get("/ToUnicode")
                font_descriptor = font.get("/FontDescriptor")

                # Skip if we've already checked this font
                if base_font in fonts_by_name:
                    continue

                font_info = {
                    "base_font": base_font,
                    "subtype": subtype,
                    "has_encoding": encoding is not None,
                    "has_tounicode": to_unicode is not None,
                    "has_descriptor": font_descriptor is not None,
                    "is_embedded": False,
                    "is_standard": False,
                    "is_symbolic": False,
                    "issues": [],
                }

                # Check if it's a standard font
                # Strip subset prefix (e.g., "ABCDEF+Helvetica" -> "Helvetica")
                clean_name = base_font.split("+")[-1] if "+" in base_font else base_font
                font_info["is_standard"] = clean_name in STANDARD_14_FONTS
                font_info["is_symbolic"] = clean_name in SYMBOLIC_FONTS

                # Check for embedding
                if font_descriptor:
                    has_fontfile = (
                        font_descriptor.get("/FontFile") is not None
                        or font_descriptor.get("/FontFile2") is not None
                        or font_descriptor.get("/FontFile3") is not None
                    )
                    font_info["is_embedded"] = has_fontfile

                # Check for Type3 fonts (problematic for accessibility)
                if subtype == "Type3":
                    font_info["issues"].append(
                        "Type3 font (may cause accessibility issues)"
                    )

                # Check ToUnicode for non-symbolic fonts
                if not font_info["is_symbolic"] and not to_unicode:
                    # Standard fonts with WinAnsiEncoding are usually OK
                    if font_info["is_standard"] and str(encoding) == "/WinAnsiEncoding":
                        pass  # OK - standard encoding is mappable
                    else:
                        font_info["issues"].append(
                            "Missing ToUnicode CMap (text may not be extractable)"
                        )

                # Check embedding for non-standard fonts
                if not font_info["is_standard"] and not font_info["is_embedded"]:
                    font_info["issues"].append(
                        "Not embedded (may not display correctly)"
                    )

                fonts_by_name[base_font] = font_info

            except Exception:
                continue

    # Report font findings
    results["fonts"] = fonts_by_name
    total_fonts = len(fonts_by_name)
    fonts_with_issues = [f for f, info in fonts_by_name.items() if info["issues"]]

    if total_fonts > 0:
        add_check("info", "Fonts", f"Found {total_fonts} unique font(s)")

        if fonts_with_issues:
            for font_name in fonts_with_issues:
                info = fonts_by_name[font_name]
                for issue in info["issues"]:
                    add_check("warn", "Fonts", f"{font_name}: {issue}")
        else:
            add_check("pass", "Fonts", "All fonts appear PDF/UA compliant")

    # === Structure ===
    tag_counts: Dict[str, int] = {}
    figures_with_alt = 0
    figures_with_placeholder = 0
    figures_without_alt = 0

    struct_tree = pdf.Root.get("/StructTreeRoot")
    if struct_tree:
        add_check("pass", "Document Structure", "Structure tree: Present")

        # Count tags and analyze figures
        struct_info = _count_structure_tags(struct_tree)
        tag_counts = struct_info["tag_counts"]
        figures_with_alt = struct_info["figures_with_alt"]
        figures_with_placeholder = struct_info["figures_with_placeholder"]
        figures_without_alt = struct_info["figures_without_alt"]

        results["tag_counts"] = tag_counts
        total_tags = sum(tag_counts.values())
        add_check("pass", "Document Structure", f"Structure elements: {total_tags}")

        parent_tree = struct_tree.get("/ParentTree")
        if parent_tree:
            add_check("pass", "Document Structure", "Parent tree: Present")
        else:
            add_check(
                "warn",
                "Document Structure",
                "Parent tree: Missing (navigation may be limited)",
            )

        # Check heading order (H1 should come before H2, etc.)
        h1_count = tag_counts.get("H1", 0)
        h2_count = tag_counts.get("H2", 0)
        h3_count = tag_counts.get("H3", 0)

        if h2_count > 0 and h1_count == 0:
            add_check(
                "warn", "Document Structure", "H2 found without H1 (heading hierarchy)"
            )
        if h3_count > 0 and h2_count == 0 and h1_count == 0:
            add_check(
                "warn",
                "Document Structure",
                "H3 found without H1/H2 (heading hierarchy)",
            )

    else:
        add_check("fail", "Document Structure", "Structure tree: Missing")

    # === Images/Figures ===
    total_images = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        total_images += len(page.get_images(full=True))

    results["total_images"] = total_images
    results["figures_with_alt"] = figures_with_alt
    results["figures_with_placeholder"] = figures_with_placeholder
    results["figures_without_alt"] = figures_without_alt

    if total_images == 0:
        add_check("pass", "Images & Alt Text", "No images found")
    else:
        # This is informational, not a check
        results["checks"].append(
            ("info", "Images & Alt Text", f"Found {total_images} image(s) in document")
        )

        if struct_tree:
            figure_count = tag_counts.get("Figure", 0)
            if figure_count > 0:
                results["checks"].append(
                    (
                        "info",
                        "Images & Alt Text",
                        f"Found {figure_count} Figure tag(s) in structure",
                    )
                )
                if figures_with_alt > 0:
                    add_check(
                        "pass", "Images & Alt Text", f"{figures_with_alt} with alt text"
                    )
                if figures_with_placeholder > 0:
                    add_check(
                        "warn",
                        "Images & Alt Text",
                        f"{figures_with_placeholder} with placeholder alt text",
                    )
                if figures_without_alt > 0:
                    add_check(
                        "fail",
                        "Images & Alt Text",
                        f"{figures_without_alt} missing alt text",
                    )
            else:
                add_check(
                    "warn", "Images & Alt Text", "No Figure tags found for images"
                )
        else:
            add_check(
                "fail", "Images & Alt Text", "Cannot check alt text (no structure tree)"
            )

    # === Advanced PDF/UA Checks ===
    has_mcids = struct_tree and _check_for_mcids(struct_tree)
    results["flags"]["mcids"] = bool(has_mcids)

    if has_mcids:
        add_check("pass", "Advanced (PDF/UA)", "Content marked with MCIDs")
    else:
        add_check(
            "warn", "Advanced (PDF/UA)", "No MCIDs found (tags not linked to content)"
        )

    # Check for empty marked content blocks (PDF/UA Content compliance)
    empty_content_blocks = _count_empty_marked_content(pdf)
    results["empty_content_blocks"] = empty_content_blocks
    if empty_content_blocks > 0:
        add_check(
            "fail",
            "Advanced (PDF/UA)",
            f"Empty marked content: {empty_content_blocks} blocks have MCIDs but no content",
        )

    # Check for untagged graphics (table borders, decorative lines, etc.)
    untagged_graphics = _count_untagged_graphics(pdf)
    results["untagged_graphics"] = untagged_graphics
    if untagged_graphics["total"] > 0:
        pages_affected = len(untagged_graphics["by_page"])
        add_check(
            "fail",
            "Advanced (PDF/UA)",
            f"Untagged graphics: {untagged_graphics['total']} graphics not marked as Artifact ({pages_affected} page(s))",
        )

    if mark_info:
        suspects = mark_info.get("/Suspects")
        if not suspects:
            add_check("pass", "Advanced (PDF/UA)", "Suspects flag: Clear")
        else:
            add_check(
                "warn",
                "Advanced (PDF/UA)",
                "Suspects flag: Set (structure may need review)",
            )

    # Check tab order on pages
    pages_with_tab_order = 0
    for page in pdf.pages:
        if page.get("/Tabs") == pikepdf.Name("/S"):
            pages_with_tab_order += 1

    if pages_with_tab_order == len(pdf.pages):
        add_check("pass", "Advanced (PDF/UA)", "Tab order: Follows structure")
        results["flags"]["tab_order"] = True
    elif pages_with_tab_order > 0:
        add_check(
            "warn",
            "Advanced (PDF/UA)",
            f"Tab order: Only {pages_with_tab_order}/{len(pdf.pages)} pages set",
        )
        results["flags"]["tab_order"] = False
    else:
        add_check("fail", "Advanced (PDF/UA)", "Tab order: Not set")
        results["flags"]["tab_order"] = False

    # === Deep Structure Analysis (Matterhorn Protocol checks) ===
    if struct_tree:
        deep_analysis = _analyze_structure_deeply(struct_tree, pdf)
        results["deep_analysis"] = deep_analysis

        # Heading hierarchy checks
        if deep_analysis["heading_hierarchy_issues"]:
            for issue in deep_analysis["heading_hierarchy_issues"]:
                add_check("warn", "Headings", f"Hierarchy issue: {issue}")
        elif deep_analysis["headings"]:
            add_check(
                "pass",
                "Headings",
                f"Heading hierarchy: OK ({len(deep_analysis['headings'])} headings)",
            )
        else:
            add_check("warn", "Headings", "No headings found in document")

        # List structure checks
        if deep_analysis["lists"]["count"] > 0:
            if deep_analysis["lists"]["malformed"]:
                for issue in deep_analysis["lists"]["malformed"][:3]:  # Limit to 3
                    add_check("warn", "Lists", f"Structure issue: {issue}")
            else:
                add_check(
                    "pass",
                    "Lists",
                    f"List structure: OK ({deep_analysis['lists']['count']} lists)",
                )

        # Table structure checks
        if deep_analysis["tables"]["count"] > 0:
            if deep_analysis["tables"]["without_headers"] > 0:
                add_check(
                    "warn",
                    "Tables",
                    f"{deep_analysis['tables']['without_headers']} table(s) without header cells",
                )
            if deep_analysis["tables"]["with_headers"] > 0:
                add_check(
                    "pass",
                    "Tables",
                    f"{deep_analysis['tables']['with_headers']} table(s) with proper headers",
                )

        # Link completeness checks
        if deep_analysis["links"]["count"] > 0:
            if deep_analysis["links"]["objr_only"] > 0:
                add_check(
                    "warn",
                    "Links",
                    f"{deep_analysis['links']['objr_only']} link(s) missing text content",
                )
            if deep_analysis["links"]["with_content"] > 0:
                add_check(
                    "pass",
                    "Links",
                    f"{deep_analysis['links']['with_content']} link(s) properly tagged",
                )

        # Document tag check
        if not deep_analysis["has_document_tag"]:
            add_check("warn", "Document Structure", "No Document root tag found")

        # RoleMap check
        if not deep_analysis["has_rolemap"] and deep_analysis["custom_roles"]:
            add_check(
                "warn",
                "Document Structure",
                f"Custom roles without RoleMap: {', '.join(deep_analysis['custom_roles'][:5])}",
            )

    # === Bookmarks/Outline check ===
    bookmark_info = _check_bookmarks(pdf)
    results["bookmarks"] = bookmark_info

    if bookmark_info["has_bookmarks"]:
        add_check(
            "pass",
            "Navigation",
            f"Bookmarks: Present ({bookmark_info['bookmark_count']} entries)",
        )
    else:
        # Only warn if document has headings (should have matching bookmarks)
        if struct_tree and tag_counts.get("H1", 0) + tag_counts.get("H2", 0) > 0:
            add_check(
                "warn",
                "Navigation",
                "No bookmarks (document has headings that could be bookmarked)",
            )

    # === Metadata consistency check ===
    metadata_info = _check_metadata_consistency(pdf)
    results["metadata"] = metadata_info

    if metadata_info["title_mismatch"]:
        add_check(
            "warn", "Metadata", "Title mismatch between document info and XMP metadata"
        )

    doc.close()
    pdf.close()

    return results


def format_accessibility_report(
    results: Dict[str, Any], use_symbols: bool = False
) -> str:
    """
    Format accessibility results as a string report.

    Args:
        results: Accessibility check results from gather_accessibility_info()
        use_symbols: If True, use Unicode symbols (✓/⚠/✗). If False, use [PASS]/[WARN]/[FAIL].
    """
    if "error" in results:
        prefix = "✗ " if use_symbols else ""
        return f"{prefix}Error opening PDF: {results['error']}"

    # Status prefixes
    if use_symbols:
        prefixes = {"pass": "✓", "warn": "⚠", "fail": "✗", "info": ""}
    else:
        prefixes = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]", "info": ""}

    lines = [
        f"Document: {results['filename']}",
        f"Checked: {results['timestamp']}",
        f"Pages: {results['pages']}",
        "",
    ]

    # Group checks by category
    checks_by_category: Dict[str, List] = {}
    for status, category, message in results["checks"]:
        if category not in checks_by_category:
            checks_by_category[category] = []
        checks_by_category[category].append((status, message))

    # Output each category in order
    for category in CATEGORIES:
        if category not in checks_by_category:
            continue

        lines.append(f"{category}:")
        for status, message in checks_by_category[category]:
            prefix = prefixes.get(status, "")
            if prefix:
                lines.append(f"  {prefix} {message}")
            else:
                lines.append(f"  {message}")

        # Show tag breakdown after structure section
        if category == "Document Structure" and results.get("tag_counts"):
            lines.append("    Tags found:")
            tag_counts = results["tag_counts"]
            # Show known tags first, then others
            for tag in KNOWN_TAGS:
                if tag in tag_counts:
                    lines.append(f"      - {tag}: {tag_counts[tag]}")
            other_tags = {k: v for k, v in tag_counts.items() if k not in KNOWN_TAGS}
            for tag, count in sorted(other_tags.items()):
                lines.append(f"      - {tag}: {count}")

        lines.append("")

    # Summary
    summary = results["summary"]
    lines.append("=" * 50)
    lines.append("Summary:")
    if use_symbols:
        lines.append(f"  ✓ Passed:   {summary['passed']}")
        lines.append(f"  ⚠ Warnings: {summary['warned']}")
        lines.append(f"  ✗ Failed:   {summary['failed']}")
    else:
        lines.append(f"  Passed:   {summary['passed']}")
        lines.append(f"  Warnings: {summary['warned']}")
        lines.append(f"  Failed:   {summary['failed']}")
    lines.append("")

    if summary["failed"] == 0 and summary["warned"] == 0:
        lines.append("This PDF should pass Adobe Acrobat's accessibility checker.")
    elif summary["failed"] == 0:
        lines.append("This PDF may pass basic checks but has some warnings.")
        lines.append("Run Adobe Acrobat's full checker for complete validation.")
    else:
        lines.append("This PDF will likely fail Adobe Acrobat's accessibility checker.")

    return "\n".join(lines)


def print_accessibility_report(results: Dict[str, Any]) -> None:
    """Print accessibility results with colored symbols to terminal."""
    print(format_accessibility_report(results, use_symbols=True))


def check_accessibility(pdf_path: Path) -> int:
    """Check PDF accessibility status and report issues."""
    results = gather_accessibility_info(pdf_path)

    if "error" in results:
        print(f"\n✗ Error opening PDF: {results['error']}")
        return 1

    print_accessibility_report(results)

    return 0 if results["summary"]["failed"] == 0 else 1


def process_markdown(md_path: Path) -> int:
    """Apply alt text from markdown file to PDF."""
    print(f"Reading alt text from: {md_path}")

    data = parse_alt_text_markdown(str(md_path))
    pdf_name = data["pdf_name"]
    alt_texts = data["alt_texts"]

    if not alt_texts:
        print("No alt text entries found in markdown.")
        return 1

    # Find the accessible PDF
    review_dir = md_path.parent
    pdf_stem = pdf_name.replace(".pdf", "")
    accessible_pdf = review_dir.parent / f"{pdf_stem}_accessible.pdf"

    if not accessible_pdf.exists():
        # Try looking in same directory
        accessible_pdf = review_dir / f"{pdf_stem}_accessible.pdf"

    if not accessible_pdf.exists():
        print(f"Error: Could not find {pdf_stem}_accessible.pdf")
        print(f"Looked in: {review_dir.parent} and {review_dir}")
        return 1

    print(f"Applying {len(alt_texts)} alt text description(s) to: {accessible_pdf}")
    apply_alt_text_to_pdf(str(accessible_pdf), alt_texts)
    print(f"\nDone! Updated: {accessible_pdf}")

    return 0


def main() -> int:
    """Main entry point."""
    print("\n=== PDF Accessibility Tool ===\n")

    # Parse arguments
    args = sys.argv[1:]
    check_mode = False
    force_mode = False
    input_path = None

    for arg in args:
        if arg in ("--check", "-c"):
            check_mode = True
        elif arg in ("--force", "-f"):
            force_mode = True
        elif not arg.startswith("-"):
            input_path = arg

    # If no file provided, open file picker
    if not input_path:
        print("Select a PDF file to make accessible...\n")
        input_path = select_pdf_file()

    if not input_path:
        print("No file selected. Exiting.")
        return 1

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    try:
        # Check mode - just report accessibility status
        if check_mode:
            if input_file.suffix.lower() != ".pdf":
                print("Error: --check only works with PDF files")
                return 1
            return check_accessibility(input_file)

        # Normal processing mode
        if input_file.suffix.lower() == ".md":
            return process_markdown(input_file)
        elif input_file.suffix.lower() == ".pdf":
            return process_pdf(input_file, force=force_mode)
        else:
            print(f"Error: Unsupported file type: {input_file.suffix}")
            print("Supported: .pdf, .md")
            return 1
    except Exception as e:
        print(f"\nError: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
