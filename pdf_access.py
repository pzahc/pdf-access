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
        print(f"Could not open file picker: {e}")
        print("\nYou can also run with a file path argument:")
        print("  uv run pdf_access.py /path/to/your/file.pdf\n")
        return None


def analyze_document(pdf_path: str) -> Dict[str, Any]:
    """Analyze PDF content to determine structure using pymupdf."""
    import fitz

    doc = fitz.open(pdf_path)

    analysis = {
        "pages": [],
        "font_sizes": set(),
        "images": [],
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_data = {"number": page_num, "elements": []}

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = round(span["size"], 1)
                        text = span["text"].strip()
                        if text:
                            analysis["font_sizes"].add(size)
                            page_data["elements"].append(
                                {
                                    "type": "text",
                                    "text": text,
                                    "size": size,
                                    "bbox": span["bbox"],
                                }
                            )
            elif block["type"] == 1:  # image
                page_data["elements"].append(
                    {
                        "type": "image",
                        "bbox": block["bbox"],
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

        analysis["pages"].append(page_data)

    doc.close()

    # Determine heading sizes (largest fonts are likely headings)
    sizes = sorted(analysis["font_sizes"], reverse=True)
    analysis["heading_sizes"] = sizes[:3] if len(sizes) >= 3 else sizes

    return analysis


def classify_element(elem: Dict, heading_sizes: List[float]) -> str:
    """Classify a text element as heading or paragraph."""
    if elem["type"] == "image":
        return "Figure"

    size = elem["size"]
    if heading_sizes and size == heading_sizes[0]:
        return "H1"
    elif len(heading_sizes) > 1 and size == heading_sizes[1]:
        return "H2"
    elif len(heading_sizes) > 2 and size == heading_sizes[2]:
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


def extract_images(pdf_path: str, output_dir: Path) -> List[Dict[str, Any]]:
    """Extract all images from PDF."""
    import fitz

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    image_list = []
    doc = fitz.open(pdf_path)

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
            except Exception:
                pass

    doc.close()
    return image_list


def generate_alt_text_markdown(
    pdf_path: str,
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

    return md_path


def parse_alt_text_markdown(md_path: str) -> Dict[str, Any]:
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


def apply_alt_text_to_pdf(pdf_path: str, alt_texts: Dict[int, str]) -> None:
    """Apply alt text from markdown to the PDF structure."""
    import pikepdf

    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
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

        pdf.save(pdf_path)


def inject_mcids_into_page(
    pdf,
    page,
    analysis_elements: List[Dict],
    heading_sizes: List[float],
    start_mcid: int = 0,
) -> tuple:
    """
    Inject BDC/EMC markers into a page's content stream.
    Returns (mcid_info_list, next_mcid) where mcid_info contains tag type for each MCID.
    """
    import pikepdf

    instructions = list(pikepdf.parse_content_stream(page))
    new_instructions = []
    mcid = start_mcid
    mcid_info = []

    # Track which analysis element we're on (text blocks map to analysis)
    text_block_index = 0
    text_elements = [e for e in analysis_elements if e["type"] == "text"]
    image_index = 0
    image_elements = [e for e in analysis_elements if e["type"] == "image"]

    for operands, operator in instructions:
        op = str(operator)

        if op == "BT":
            # Determine tag type from analysis
            if text_block_index < len(text_elements):
                elem = text_elements[text_block_index]
                tag = classify_element(elem, heading_sizes)
                text_block_index += 1
            else:
                tag = "P"

            bdc_operands = [
                pikepdf.Name(f"/{tag}"),
                pikepdf.Dictionary({"/MCID": mcid}),
            ]
            new_instructions.append((bdc_operands, pikepdf.Operator("BDC")))
            mcid_info.append({"type": "text", "tag": tag, "mcid": mcid})

        elif op == "Do":
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

        new_instructions.append((operands, operator))

        if op == "ET":
            new_instructions.append(([], pikepdf.Operator("EMC")))
            mcid += 1

    # Write new content stream
    new_content = pikepdf.unparse_content_stream(new_instructions)
    page.Contents = pdf.make_stream(new_content)

    return mcid_info, mcid


def make_accessible(
    input_path: str,
    output_path: str,
    analysis: Dict[str, Any],
    alt_texts: Optional[Dict[int, str]] = None,
) -> str:
    """
    Add accessibility tags to PDF using pikepdf with proper MCID linking.
    Returns the document title.
    """
    import pikepdf

    with pikepdf.open(input_path) as pdf:
        # Set document language
        pdf.Root.Lang = pikepdf.String("en-US")

        # Set ViewerPreferences
        if "/ViewerPreferences" not in pdf.Root:
            pdf.Root.ViewerPreferences = pikepdf.Dictionary()
        pdf.Root.ViewerPreferences.DisplayDocTitle = True

        # Set MarkInfo
        pdf.Root.MarkInfo = pikepdf.Dictionary(
            {
                "/Marked": True,
                "/Suspects": False,
            }
        )

        # Find title from first H1
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

        # Set title metadata
        with pdf.open_metadata() as meta:
            meta["dc:title"] = title
            meta["dc:language"] = "en-US"

        if pdf.trailer.get("/Info"):
            pdf.trailer.Info["/Title"] = pikepdf.String(title)
            pdf.trailer.Info["/Producer"] = pikepdf.String("PDF Accessibility Tool")
            pdf.trailer.Info["/Creator"] = pikepdf.String("PDF Accessibility Tool")

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

            # Inject MCIDs into this page's content stream
            mcid_info, _ = inject_mcids_into_page(
                pdf,
                page,
                page_analysis["elements"],
                analysis["heading_sizes"],
                start_mcid=0,  # MCIDs reset per page
            )

            # Create structure elements for this page
            page_struct_elems = []
            for info in mcid_info:
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

                # Add alt text for figures
                if tag == "Figure":
                    figure_count += 1
                    if alt_texts and figure_count in alt_texts:
                        struct_elem["/Alt"] = pikepdf.String(alt_texts[figure_count])
                    else:
                        # Try to find caption from analysis
                        elements = page_analysis["elements"]
                        img_indices = [
                            i for i, e in enumerate(elements) if e["type"] == "image"
                        ]
                        img_idx = figure_count - 1
                        if img_idx < len(img_indices):
                            caption = find_image_caption(elements, img_indices[img_idx])
                            if caption:
                                struct_elem["/Alt"] = pikepdf.String(caption)
                            else:
                                struct_elem["/Alt"] = pikepdf.String(
                                    f"Image {figure_count} - alt text needed"
                                )

                page_struct_elems.append(struct_elem)
                all_struct_elems.append(struct_elem)

            # Add to ParentTree: page_num -> array of struct elems by MCID order
            if page_struct_elems:
                parent_tree_nums.append(page_num)
                parent_tree_nums.append(pikepdf.Array(page_struct_elems))

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

        # Create StructTreeRoot
        struct_tree_root = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/StructTreeRoot"),
                    "/K": doc_elem,
                    "/ParentTree": parent_tree,
                    "/ParentTreeNextKey": len(pdf.pages),
                }
            )
        )

        doc_elem["/P"] = struct_tree_root
        pdf.Root.StructTreeRoot = struct_tree_root

        pdf.save(output_path)

        return title


def save_check_report(results: Dict[str, Any], report_path: Path, label: str) -> None:
    """Save accessibility check results to a file."""
    report = format_accessibility_report(results)
    with open(report_path, "w") as f:
        f.write(f"# Accessibility Report ({label})\n\n")
        f.write(f"```\n{report}\n```\n")


def is_already_accessible(results: Dict[str, Any]) -> bool:
    """Check if a PDF is already well-tagged with MCIDs."""
    # Check for key indicators of a well-tagged PDF
    checks = results.get("checks", [])

    has_mcids = False
    has_parent_tree = False
    has_struct_tree = False

    for status, category, message in checks:
        if "Content marked with MCIDs" in message and status == "pass":
            has_mcids = True
        if "Parent tree: Present" in message and status == "pass":
            has_parent_tree = True
        if "Structure tree: Present" in message and status == "pass":
            has_struct_tree = True

    # Consider it already accessible if it has MCIDs and ParentTree
    return has_mcids and has_parent_tree and has_struct_tree


def process_pdf(input_path: Path, force: bool = False) -> int:
    """Process a PDF file to make it accessible."""
    output_path = input_path.parent / f"{input_path.stem}_accessible.pdf"
    review_dir = input_path.parent / f"{input_path.stem}_review"

    # Run pre-check
    print("Checking current accessibility status...")
    pre_results = gather_accessibility_info(input_path)

    # Check if already accessible
    if is_already_accessible(pre_results) and not force:
        print()
        print("This PDF is already well-tagged with MCIDs and structure.")
        print(f"  - Passed: {pre_results['summary']['passed']}")
        print(f"  - Warnings: {pre_results['summary']['warned']}")
        print(f"  - Failed: {pre_results['summary']['failed']}")
        print()
        print("No processing needed. Use --force to override and reprocess.")
        print(f"Run: uv run pdf_access.py --check {input_path}")
        return 0

    print(f"  - Found {pre_results['summary']['failed']} issue(s) to fix")
    print()

    print("Analyzing document structure...")
    analysis = analyze_document(str(input_path))

    print("Applying accessibility fixes...")
    print("  - Adding document tags")
    print("  - Setting document language (en-US)")

    title = make_accessible(str(input_path), str(output_path), analysis)
    print(f"  - Setting document title: {title[:50]}{'...' if len(title) > 50 else ''}")
    print("  - Enabling display document title")
    print(f"Saving: {output_path}")

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
        print()
        print(f"Images need alt text review. Edit and re-run:")
        print(f"  1. Edit: {md_path}")
        print(f"  2. Run:  uv run pdf_access.py {md_path}")
    elif total_images > 0:
        print()
        print("All images have captions - no manual alt text needed!")

    return 0


def gather_accessibility_info(pdf_path: Path) -> Dict[str, Any]:
    """Gather accessibility information from a PDF without printing."""
    import pikepdf
    import fitz

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results: Dict[str, Any] = {
        "filename": pdf_path.name,
        "filepath": str(pdf_path),
        "timestamp": timestamp,
        "pages": 0,
        "checks": [],  # List of (status, category, message) tuples
        "tag_counts": {},
        "summary": {"passed": 0, "warned": 0, "failed": 0},
    }

    def add_check(status: str, category: str, message: str):
        results["checks"].append((status, category, message))
        if status == "pass":
            results["summary"]["passed"] += 1
        elif status == "warn":
            results["summary"]["warned"] += 1
        else:
            results["summary"]["failed"] += 1

    # Open with both libraries
    try:
        pdf = pikepdf.open(str(pdf_path))
    except Exception as e:
        results["error"] = str(e)
        return results

    doc = fitz.open(str(pdf_path))
    results["pages"] = len(doc)

    # === Document Settings ===
    mark_info = pdf.Root.get("/MarkInfo")
    if mark_info and mark_info.get("/Marked"):
        add_check("pass", "Document Settings", "Tagged PDF: Yes")
    else:
        add_check("fail", "Document Settings", "Tagged PDF: No")

    lang = pdf.Root.get("/Lang")
    if lang:
        add_check("pass", "Document Settings", f"Language: {lang}")
    else:
        add_check("fail", "Document Settings", "Language: Not set")

    title = None
    if pdf.trailer.get("/Info"):
        title = pdf.trailer.Info.get("/Title")
    if title and str(title).strip():
        title_str = str(title)
        if len(title_str) > 50:
            title_str = title_str[:50] + "..."
        add_check("pass", "Document Settings", f'Title: "{title_str}"')
    else:
        add_check("fail", "Document Settings", "Title: Not set")

    viewer_prefs = pdf.Root.get("/ViewerPreferences")
    if viewer_prefs and viewer_prefs.get("/DisplayDocTitle"):
        add_check("pass", "Document Settings", "Display Doc Title: Enabled")
    else:
        add_check("fail", "Document Settings", "Display Doc Title: Disabled")

    # === Structure ===
    tag_counts: Dict[str, int] = {}
    figures_without_alt = 0
    figures_with_placeholder = 0
    figures_with_alt = 0

    struct_tree = pdf.Root.get("/StructTreeRoot")
    if struct_tree:
        add_check("pass", "Document Structure", "Structure tree: Present")

        def count_tags(elem):
            nonlocal figures_without_alt, figures_with_placeholder, figures_with_alt
            if not hasattr(elem, "get"):
                return

            tag = str(elem.get("/S", "")).replace("/", "")
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

                if tag == "Figure":
                    alt = elem.get("/Alt")
                    if alt:
                        alt_str = str(alt)
                        if "alt text needed" in alt_str.lower() or alt_str == "":
                            figures_with_placeholder += 1
                        else:
                            figures_with_alt += 1
                    else:
                        figures_without_alt += 1

            kids = elem.get("/K")
            if kids:
                if hasattr(kids, "__iter__") and not isinstance(kids, (str, bytes)):
                    for kid in kids:
                        count_tags(kid)
                else:
                    count_tags(kids)

        doc_elem = struct_tree.get("/K")
        if doc_elem:
            count_tags(doc_elem)

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
    has_mcids = False

    def is_mcid_value(val):
        """Check if value is an MCID (integer or pikepdf integer)."""
        try:
            int(val)
            return not hasattr(val, "get")  # Not a dict
        except (TypeError, ValueError):
            return False

    def is_array(val):
        """Check if value is a pikepdf Array (iterable but not dict/str)."""
        if isinstance(val, (str, bytes)):
            return False
        if not hasattr(val, "__iter__"):
            return False
        # pikepdf dicts have .get(), arrays don't (they have .append())
        return hasattr(val, "append") or not hasattr(val, "get")

    def check_mcids(elem):
        nonlocal has_mcids
        if has_mcids:
            return
        if not hasattr(elem, "get"):
            return

        kids = elem.get("/K")
        if kids is None:
            return

        # /K can be: integer (MCID), dict with /MCID, or array of these
        # Check if /K is an integer (direct MCID reference)
        if is_mcid_value(kids):
            has_mcids = True
            return

        # /K is an array
        if is_array(kids):
            for kid in kids:
                if is_mcid_value(kid):
                    has_mcids = True
                    return
                if hasattr(kid, "get"):
                    if kid.get("/MCID") is not None or kid.get("/Type") == "/MCR":
                        has_mcids = True
                        return
                    check_mcids(kid)
            return

        # /K is a dictionary with /MCID or a struct elem
        if hasattr(kids, "get"):
            if kids.get("/MCID") is not None or kids.get("/Type") == "/MCR":
                has_mcids = True
                return
            check_mcids(kids)

    if struct_tree:
        doc_elem = struct_tree.get("/K")
        if doc_elem:
            check_mcids(doc_elem)

    if has_mcids:
        add_check("pass", "Advanced (PDF/UA)", "Content marked with MCIDs")
    else:
        add_check(
            "warn", "Advanced (PDF/UA)", "No MCIDs found (tags not linked to content)"
        )

    if mark_info:
        suspects = mark_info.get("/Suspects")
        if suspects == False or suspects is None:
            add_check("pass", "Advanced (PDF/UA)", "Suspects flag: Clear")
        else:
            add_check(
                "warn",
                "Advanced (PDF/UA)",
                "Suspects flag: Set (structure may need review)",
            )

    doc.close()
    pdf.close()

    return results


def format_accessibility_report(
    results: Dict[str, Any], include_header: bool = True
) -> str:
    """Format accessibility results as a string report."""
    lines = []

    if "error" in results:
        return f"Error opening PDF: {results['error']}"

    if include_header:
        lines.append(f"Document: {results['filename']}")
        lines.append(f"Checked: {results['timestamp']}")
        lines.append(f"Pages: {results['pages']}")
        lines.append("")

    # Group checks by category
    categories: Dict[str, List] = {}
    for status, category, message in results["checks"]:
        if category not in categories:
            categories[category] = []
        categories[category].append((status, message))

    # Output each category
    for category in [
        "Document Settings",
        "Document Structure",
        "Images & Alt Text",
        "Advanced (PDF/UA)",
    ]:
        if category not in categories:
            continue

        lines.append(f"{category}:")
        for status, message in categories[category]:
            if status == "pass":
                lines.append(f"  [PASS] {message}")
            elif status == "warn":
                lines.append(f"  [WARN] {message}")
            elif status == "fail":
                lines.append(f"  [FAIL] {message}")
            else:  # info
                lines.append(f"  {message}")

        # Show tag breakdown after structure section
        if category == "Document Structure" and results.get("tag_counts"):
            lines.append("    Tags found:")
            for tag in [
                "Document",
                "H1",
                "H2",
                "H3",
                "P",
                "Figure",
                "Table",
                "L",
                "LI",
            ]:
                if tag in results["tag_counts"]:
                    lines.append(f"      - {tag}: {results['tag_counts'][tag]}")
            other_tags = {
                k: v
                for k, v in results["tag_counts"].items()
                if k
                not in ["Document", "H1", "H2", "H3", "P", "Figure", "Table", "L", "LI"]
            }
            for tag, count in sorted(other_tags.items()):
                lines.append(f"      - {tag}: {count}")

        lines.append("")

    # Summary
    summary = results["summary"]
    lines.append("=" * 50)
    lines.append("Summary:")
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
    if "error" in results:
        print(f"\n✗ Error opening PDF: {results['error']}")
        return

    print(f"Document: {results['filename']}")
    print(f"Checked: {results['timestamp']}")
    print(f"Pages: {results['pages']}")
    print()

    # Group checks by category
    categories: Dict[str, List] = {}
    for status, category, message in results["checks"]:
        if category not in categories:
            categories[category] = []
        categories[category].append((status, message))

    # Output each category
    for category in [
        "Document Settings",
        "Document Structure",
        "Images & Alt Text",
        "Advanced (PDF/UA)",
    ]:
        if category not in categories:
            continue

        print(f"{category}:")
        for status, message in categories[category]:
            if status == "pass":
                print(f"  ✓ {message}")
            elif status == "warn":
                print(f"  ⚠ {message}")
            elif status == "fail":
                print(f"  ✗ {message}")
            else:  # info
                print(f"  {message}")

        # Show tag breakdown after structure section
        if category == "Document Structure" and results.get("tag_counts"):
            print("    Tags found:")
            for tag in [
                "Document",
                "H1",
                "H2",
                "H3",
                "P",
                "Figure",
                "Table",
                "L",
                "LI",
            ]:
                if tag in results["tag_counts"]:
                    print(f"      - {tag}: {results['tag_counts'][tag]}")
            other_tags = {
                k: v
                for k, v in results["tag_counts"].items()
                if k
                not in ["Document", "H1", "H2", "H3", "P", "Figure", "Table", "L", "LI"]
            }
            for tag, count in sorted(other_tags.items()):
                print(f"      - {tag}: {count}")

        print()

    # Summary
    summary = results["summary"]
    print("=" * 50)
    print("Summary:")
    print(f"  ✓ Passed:   {summary['passed']}")
    print(f"  ⚠ Warnings: {summary['warned']}")
    print(f"  ✗ Failed:   {summary['failed']}")
    print()

    if summary["failed"] == 0 and summary["warned"] == 0:
        print("This PDF should pass Adobe Acrobat's accessibility checker.")
    elif summary["failed"] == 0:
        print("This PDF may pass basic checks but has some warnings.")
        print("Run Adobe Acrobat's full checker for complete validation.")
    else:
        print("This PDF will likely fail Adobe Acrobat's accessibility checker.")


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
