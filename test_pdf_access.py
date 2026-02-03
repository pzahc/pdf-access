import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pdf_access

# ============================================================================
# Unit Tests - Type Checking Helpers
# ============================================================================


def test_is_mcid_value():
    assert pdf_access.is_mcid_value(123) is True
    assert pdf_access.is_mcid_value("123") is True  # int("123") works
    assert pdf_access.is_mcid_value({"a": 1}) is False
    assert pdf_access.is_mcid_value([1, 2]) is False
    assert pdf_access.is_mcid_value(None) is False


def test_is_pdf_array():
    assert pdf_access.is_pdf_array([1, 2, 3]) is True
    assert pdf_access.is_pdf_array("string") is False
    assert pdf_access.is_pdf_array(b"bytes") is False
    assert pdf_access.is_pdf_array({"key": "value"}) is False

    # Mocking a pikepdf Array behavior - just use a list which is the simplest
    # Real pikepdf.Array objects are handled by isinstance check
    assert pdf_access.is_pdf_array([]) is True
    assert pdf_access.is_pdf_array(tuple()) is True  # tuples are also array-like


def test_is_struct_elem_with_tag():
    """Test is_struct_elem returns True for dict-like objects with /S key."""
    mock_elem = MagicMock()
    mock_elem.get.return_value = "/P"  # Has a /S tag
    assert pdf_access.is_struct_elem(mock_elem) is True


def test_is_struct_elem_without_tag():
    """Test is_struct_elem returns False for dict-like objects without /S key."""
    mock_elem = MagicMock()
    mock_elem.get.return_value = None  # No /S tag
    assert pdf_access.is_struct_elem(mock_elem) is False


def test_is_struct_elem_non_dict():
    """Test is_struct_elem returns False for non-dict values."""
    assert pdf_access.is_struct_elem(123) is False
    assert pdf_access.is_struct_elem("string") is False
    assert pdf_access.is_struct_elem([1, 2, 3]) is False
    assert pdf_access.is_struct_elem(None) is False


def test_is_struct_elem_exception():
    """Test is_struct_elem handles exceptions gracefully."""
    mock_elem = MagicMock()
    mock_elem.get.side_effect = TypeError("test error")
    assert pdf_access.is_struct_elem(mock_elem) is False


# ============================================================================
# Unit Tests - Structure Tree Analysis
# ============================================================================


def test_count_structure_tags_empty():
    """Test _count_structure_tags with empty/minimal structure tree."""
    mock_tree = MagicMock()
    mock_tree.get.return_value = None  # No /K element

    result = pdf_access._count_structure_tags(mock_tree)

    assert result["tag_counts"] == {}
    assert result["figures_with_alt"] == 0
    assert result["figures_with_placeholder"] == 0
    assert result["figures_without_alt"] == 0


def test_count_structure_tags_with_figures():
    """Test _count_structure_tags correctly counts tags and categorizes figures."""
    # Create mock structure: Document -> [P, Figure(with alt), Figure(placeholder), Figure(no alt)]
    # Note: The code calls elem.get("/S", "") with a default, so lambda needs *args

    def make_mock_elem(data):
        """Helper to create a mock element with proper .get() behavior."""
        mock = MagicMock()
        mock.get.side_effect = lambda k, *args: data.get(k, args[0] if args else None)
        return mock

    mock_p = make_mock_elem({"/S": "/P", "/K": None})
    mock_fig_with_alt = make_mock_elem(
        {"/S": "/Figure", "/Alt": "A nice image", "/K": None}
    )
    mock_fig_placeholder = make_mock_elem(
        {"/S": "/Figure", "/Alt": "Image 1 - alt text needed", "/K": None}
    )
    mock_fig_no_alt = make_mock_elem({"/S": "/Figure", "/Alt": None, "/K": None})

    # The document element with children in /K
    children = [mock_p, mock_fig_with_alt, mock_fig_placeholder, mock_fig_no_alt]
    mock_doc = make_mock_elem({"/S": "/Document", "/K": children})

    # The struct_tree.get("/K") returns the document element
    mock_tree = MagicMock()
    mock_tree.get.return_value = mock_doc

    result = pdf_access._count_structure_tags(mock_tree)

    # Document is counted, plus 1 P and 3 Figures
    assert result["tag_counts"].get("Document", 0) == 1
    assert result["tag_counts"].get("P", 0) == 1
    assert result["tag_counts"].get("Figure", 0) == 3
    assert result["figures_with_alt"] == 1
    assert result["figures_with_placeholder"] == 1
    assert result["figures_without_alt"] == 1


def test_check_for_mcids_found_integer():
    """Test _check_for_mcids returns True when MCID is an integer."""
    mock_elem = MagicMock()
    mock_elem.get.side_effect = lambda k: {"/S": "/P", "/K": 0}.get(k)  # MCID = 0

    mock_tree = MagicMock()
    mock_tree.get.return_value = mock_elem

    assert pdf_access._check_for_mcids(mock_tree) is True


def test_check_for_mcids_found_dict():
    """Test _check_for_mcids returns True when MCID is in a dict."""
    mock_mcid_ref = MagicMock()
    mock_mcid_ref.get.side_effect = lambda k: {"/MCID": 0}.get(k)

    mock_elem = MagicMock()
    mock_elem.get.side_effect = lambda k: {"/S": "/P", "/K": [mock_mcid_ref]}.get(k)

    mock_tree = MagicMock()
    mock_tree.get.return_value = mock_elem

    assert pdf_access._check_for_mcids(mock_tree) is True


def test_check_for_mcids_not_found():
    """Test _check_for_mcids returns False when no MCIDs exist."""
    mock_elem = MagicMock()
    mock_elem.get.side_effect = lambda k: {"/S": "/P", "/K": None}.get(k)

    mock_tree = MagicMock()
    mock_tree.get.return_value = mock_elem

    assert pdf_access._check_for_mcids(mock_tree) is False


def test_check_for_mcids_empty_tree():
    """Test _check_for_mcids returns False for empty tree."""
    mock_tree = MagicMock()
    mock_tree.get.return_value = None

    assert pdf_access._check_for_mcids(mock_tree) is False


# ============================================================================
# Unit Tests - Element Classification
# ============================================================================


def test_classify_element():
    # Test standard hierarchy
    heading_sizes = [24.0, 18.0, 14.0]

    # H1
    elem_h1 = {"type": "text", "size": 24.0, "text": "Title"}
    assert pdf_access.classify_element(elem_h1, heading_sizes) == "H1"

    # H2
    elem_h2 = {"type": "text", "size": 18.0, "text": "Subtitle"}
    assert pdf_access.classify_element(elem_h2, heading_sizes) == "H2"

    # H3
    elem_h3 = {"type": "text", "size": 14.0, "text": "Section"}
    assert pdf_access.classify_element(elem_h3, heading_sizes) == "H3"

    # P (size not in headings)
    elem_p = {"type": "text", "size": 12.0, "text": "Paragraph text"}
    assert pdf_access.classify_element(elem_p, heading_sizes) == "P"

    # Image
    elem_img = {"type": "image", "size": 0, "text": ""}
    assert pdf_access.classify_element(elem_img, heading_sizes) == "Figure"


def test_classify_element_empty_headings():
    heading_sizes = []
    elem = {"type": "text", "size": 12.0, "text": "Text"}
    assert pdf_access.classify_element(elem, heading_sizes) == "P"


def test_find_image_caption():
    elements = [
        {"type": "image", "text": ""},
        {"type": "text", "text": "Figure 1: A chart"},
        {"type": "text", "text": "Some other text"},
    ]

    # Direct caption
    assert pdf_access.find_image_caption(elements, 0) == "Figure 1: A chart"

    # No caption (not image type next)
    elements_no_cap = [{"type": "image", "text": ""}, {"type": "image", "text": ""}]
    assert pdf_access.find_image_caption(elements_no_cap, 0) is None

    # No caption (text doesn't start with keyword)
    elements_wrong_text = [
        {"type": "image", "text": ""},
        {"type": "text", "text": "Just normal text"},
    ]
    assert pdf_access.find_image_caption(elements_wrong_text, 0) is None

    # End of list
    assert pdf_access.find_image_caption(elements, 2) is None


# ============================================================================
# Unit Tests - Report Formatting
# ============================================================================


def test_format_accessibility_report_basic():
    """Test format_accessibility_report produces correct output."""
    results = {
        "filename": "test.pdf",
        "timestamp": "2024-01-01 12:00:00",
        "pages": 5,
        "checks": [
            ("pass", "Document Settings", "Tagged PDF: Yes"),
            ("fail", "Document Settings", "Title: Not set"),
            ("warn", "Document Structure", "H2 found without H1"),
        ],
        "tag_counts": {"Document": 1, "P": 10},
        "summary": {"passed": 1, "warned": 1, "failed": 1},
    }

    report = pdf_access.format_accessibility_report(results, use_symbols=False)

    assert "Document: test.pdf" in report
    assert "Pages: 5" in report
    assert "[PASS] Tagged PDF: Yes" in report
    assert "[FAIL] Title: Not set" in report
    assert "[WARN] H2 found without H1" in report
    assert "Passed:   1" in report
    assert "Warnings: 1" in report
    assert "Failed:   1" in report


def test_format_accessibility_report_with_symbols():
    """Test format_accessibility_report uses Unicode symbols when requested."""
    results = {
        "filename": "test.pdf",
        "timestamp": "2024-01-01 12:00:00",
        "pages": 1,
        "checks": [
            ("pass", "Document Settings", "Tagged PDF: Yes"),
            ("fail", "Document Settings", "Title: Not set"),
        ],
        "tag_counts": {},
        "summary": {"passed": 1, "warned": 0, "failed": 1},
    }

    report = pdf_access.format_accessibility_report(results, use_symbols=True)

    assert "✓ Tagged PDF: Yes" in report
    assert "✗ Title: Not set" in report


def test_format_accessibility_report_with_error():
    """Test format_accessibility_report handles error dict gracefully."""
    results = {"error": "Could not open file"}

    report = pdf_access.format_accessibility_report(results, use_symbols=False)
    assert "Error opening PDF: Could not open file" in report

    report_symbols = pdf_access.format_accessibility_report(results, use_symbols=True)
    assert "✗ Error opening PDF: Could not open file" in report_symbols


def test_format_accessibility_report_fully_accessible():
    """Test report message when PDF is fully accessible."""
    results = {
        "filename": "test.pdf",
        "timestamp": "2024-01-01 12:00:00",
        "pages": 1,
        "checks": [("pass", "Document Settings", "All good")],
        "tag_counts": {},
        "summary": {"passed": 1, "warned": 0, "failed": 0},
    }

    report = pdf_access.format_accessibility_report(results)
    assert "This PDF should pass Adobe Acrobat's accessibility checker" in report


def test_format_accessibility_report_warnings_only():
    """Test report message when PDF has warnings but no failures."""
    results = {
        "filename": "test.pdf",
        "timestamp": "2024-01-01 12:00:00",
        "pages": 1,
        "checks": [("warn", "Document Structure", "Minor issue")],
        "tag_counts": {},
        "summary": {"passed": 0, "warned": 1, "failed": 0},
    }

    report = pdf_access.format_accessibility_report(results)
    assert "This PDF may pass basic checks but has some warnings" in report


# ============================================================================
# Unit Tests - File Operations
# ============================================================================


def test_save_check_report():
    """Test save_check_report writes formatted report to file."""
    results = {
        "filename": "test.pdf",
        "timestamp": "2024-01-01 12:00:00",
        "pages": 1,
        "checks": [("pass", "Document Settings", "Tagged PDF: Yes")],
        "tag_counts": {},
        "summary": {"passed": 1, "warned": 0, "failed": 0},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "report.md"
        pdf_access.save_check_report(results, report_path, "Test Label")

        assert report_path.exists()
        content = report_path.read_text()
        assert "# Accessibility Report (Test Label)" in content
        assert "Document: test.pdf" in content
        assert "```" in content  # Code block markers


def test_generate_alt_text_markdown():
    """Test generate_alt_text_markdown creates correct markdown structure."""
    analysis = {
        "pages": [
            {
                "number": 0,
                "elements": [
                    {"type": "text", "text": "Hello", "size": 12.0},
                    {"type": "image", "bbox": [0, 0, 100, 100]},
                    {"type": "text", "text": "Figure 1: A test image", "size": 10.0},
                    {"type": "image", "bbox": [0, 0, 50, 50]},
                    {"type": "text", "text": "Some other text", "size": 10.0},
                ],
            }
        ],
        "heading_sizes": [12.0],
        "images": [],
    }

    images = [
        {"page": 1, "index": 1, "filename": "page1_img1.png"},
        {"page": 1, "index": 2, "filename": "page1_img2.png"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        md_path = pdf_access.generate_alt_text_markdown(
            "test.pdf", analysis, images, output_dir
        )

        assert md_path.exists()
        content = md_path.read_text()

        # Check header
        assert "# Alt Text for test.pdf" in content

        # Check first image (has caption)
        assert "## Image 1 (Page 1)" in content
        assert "![Image 1](images/page1_img1.png)" in content
        assert "**Auto-detected caption:** Figure 1: A test image" in content
        assert "Alt text: Figure 1: A test image" in content

        # Check second image (no caption)
        assert "## Image 2 (Page 1)" in content
        assert "Alt text: [Describe this image]" in content


# ============================================================================
# Mocked Integration Tests - gather_accessibility_info
# ============================================================================


@patch("pdf_access.fitz.open")
@patch("pdf_access.pikepdf.open")
def test_gather_accessibility_info_pikepdf_error(mock_pikepdf, mock_fitz):
    """Test gather_accessibility_info returns error when pikepdf fails."""
    mock_pikepdf.side_effect = Exception("Cannot open PDF")

    result = pdf_access.gather_accessibility_info(Path("bad.pdf"))

    assert "error" in result
    assert "Cannot open PDF" in result["error"]
    mock_fitz.assert_not_called()  # Should not try fitz if pikepdf failed


@patch("pdf_access.fitz.open")
@patch("pdf_access.pikepdf.open")
def test_gather_accessibility_info_fitz_error(mock_pikepdf, mock_fitz):
    """Test gather_accessibility_info returns error when fitz fails."""
    mock_pdf = MagicMock()
    mock_pikepdf.return_value = mock_pdf
    mock_fitz.side_effect = Exception("Fitz cannot open")

    result = pdf_access.gather_accessibility_info(Path("bad.pdf"))

    assert "error" in result
    assert "Fitz cannot open" in result["error"]
    mock_pdf.close.assert_called_once()  # Should close pikepdf handle


# ============================================================================
# Mocked Integration Tests - Document Analysis
# ============================================================================


@patch("pdf_access.fitz.open")
def test_analyze_document(mock_open):
    # Setup mock document structure
    mock_doc = MagicMock()
    mock_open.return_value = mock_doc
    mock_doc.__len__.return_value = 1

    mock_page = MagicMock()
    mock_doc.__getitem__.return_value = mock_page

    # Mock get_text("dict") output
    mock_page.get_text.return_value = {
        "blocks": [
            {
                "type": 0,  # text
                "lines": [
                    {
                        "spans": [
                            {
                                "size": 12.0,
                                "text": "Hello World",
                                "bbox": [0, 0, 100, 20],
                            }
                        ]
                    }
                ],
            },
            {
                "type": 1,  # image
                "bbox": [0, 30, 100, 100],
                "width": 100,
                "height": 70,
            },
        ]
    }

    result = pdf_access.analyze_document(Path("dummy.pdf"))

    assert len(result["pages"]) == 1
    assert len(result["font_sizes"]) == 1
    assert 12.0 in result["font_sizes"]
    assert len(result["images"]) == 1
    assert result["images"][0]["page"] == 1

    # Verify page elements
    elements = result["pages"][0]["elements"]
    assert len(elements) == 2
    assert elements[0]["type"] == "text"
    assert elements[0]["text"] == "Hello World"
    assert elements[1]["type"] == "image"


@patch("pdf_access.pikepdf.open")
def test_remediate_pdf_preserves_structure(mock_open):
    # Test that preserve_structure=True bypasses the heavy lifting
    mock_pdf = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_pdf

    # Setup minimal required structure
    mock_pdf.Root = MagicMock()
    mock_pdf.Root.get.return_value = None  # No existing mark info

    # Use MagicMock for trailer so it supports attribute assignment (pdf.trailer.Info = ...)
    mock_pdf.trailer = MagicMock()
    mock_pdf.trailer.get.return_value = None  # No existing Info

    analysis = {"pages": [], "heading_sizes": []}

    # Diagnostic with flags indicating PDF is already accessible
    diagnostic = {
        "flags": {
            "language": True,
            "title": True,
            "display_title": True,
            "tagged": True,
        }
    }

    pdf_access.remediate_pdf(
        "in.pdf", "out.pdf", analysis, preserve_structure=True, diagnostic=diagnostic
    )

    # Should save once
    mock_pdf.save.assert_called_once_with("out.pdf")

    # Verify we returned early by checking that a method called strictly *after*
    # the early return was NOT called.
    # remediate_pdf calls `pdf.make_indirect` multiple times if it proceeds.
    mock_pdf.make_indirect.assert_not_called()


def test_parse_alt_text_markdown():
    markdown_content = """# Alt Text for test.pdf

Edit the alt text below each image.

---

## Image 1 (Page 1)

![Image 1](images/page1_img1.png)

**Auto-detected caption:** Figure 1

Alt text: A beautiful sunset

---

## Image 2 (Page 2)

![Image 2](images/page2_img1.png)

Alt text: [Describe this image]

---
"""
    with patch("builtins.open", mock_open(read_data=markdown_content)):
        result = pdf_access.parse_alt_text_markdown("dummy.md")

        assert result["pdf_name"] == "test.pdf"
        assert len(result["alt_texts"]) == 1
        assert result["alt_texts"][1] == "A beautiful sunset"
        # Image 2 should be skipped because it still has the placeholder
