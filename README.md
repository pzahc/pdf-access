# PDF Accessibility Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![Made with pikepdf](https://img.shields.io/badge/made%20with-pikepdf-green.svg)](https://pikepdf.readthedocs.io/)

A free tool to make PDFs accessible for screen readers. Automatically tags document structure and provides a markdown-based workflow for adding image alt text.

## Quick Start

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv run pdf_access.py document.pdf        # Process a PDF
uv run pdf_access.py --check document.pdf # Check accessibility status
uv run pdf_access.py alt_text.md          # Apply alt text edits
```

## What It Does

| Issue | Fix |
|-------|-----|
| Missing document tags | Auto-tags headings (H1-H3), paragraphs, figures |
| No document language | Sets `en-US` |
| Missing/hidden title | Extracts from first heading, enables display |
| Images without alt text | Markdown file for easy editing + auto-detects captions |

## Workflow

**1. Process your PDF:**
```bash
uv run pdf_access.py annual_report.pdf
```

Creates `annual_report_accessible.pdf` plus a review folder with:
- `alt_text.md` - Edit image descriptions here
- `images/` - Extracted images for reference  
- `check_before.md` / `check_after.md` - Accessibility reports

**2. Edit alt text** in the markdown file, then apply:
```bash
uv run pdf_access.py annual_report_review/alt_text.md
```

## Check Mode

Inspect accessibility without modifying:

```bash
uv run pdf_access.py --check document.pdf
```

```
Document: document.pdf
Checked: 2026-01-29 14:32:01
Pages: 7

Document Settings:
  ✓ Tagged PDF: Yes
  ✓ Language: en-US
  ✓ Title: "Annual Report"
  ✓ Display Doc Title: Enabled

Document Structure:
  ✓ Structure tree: Present
  ✓ Structure elements: 132
  ⚠ Parent tree: Missing

Images & Alt Text:
  ✓ 1 with alt text
  ⚠ 4 with placeholder alt text

Summary:
  ✓ Passed: 7  ⚠ Warnings: 3  ✗ Failed: 0
```

## Limitations

- **No MCID linking** - Tags aren't linked to content streams. For full PDF/UA compliance, run Adobe Acrobat Pro's "Make Accessible" after this tool.
- **Font-based heading detection** - Works best with consistent styles.
- **Basic table support** - Complex tables may need manual review.

## Dependencies

Automatically installed by `uv`:

- [pikepdf](https://pikepdf.readthedocs.io/) (MPL-2.0) - PDF structure
- [PyMuPDF](https://pymupdf.readthedocs.io/) (AGPL-3.0) - Content analysis

## License

MIT - see [LICENSE](LICENSE)

---

*Inspired by my girlfriend yelling at PDFs.*
