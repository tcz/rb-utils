r"""
Tests for find_element_in_dom() -- verifying it returns the full element range
(opening tag through closing tag, inclusive of content and children).

Run:
    cd /Users/tcz/Dropbox/Reverse\ Browser/V2/training
    python -m pytest utils/test_find_element_range.py -v
"""
import os
import sys
import pytest

# Add parent dir so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.token_rewards import find_element_in_dom


# ── Simple element tests ──────────────────────────────────────────────

class TestSimpleElements:
    def test_simple_div_full_range(self):
        """find_element_in_dom should return range from <div> through </div>."""
        dom = '<html><body><div class="box">content</div></body></html>'
        prefix = '<div class="box">content</div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        start, end = result
        extracted = dom[start:end]
        assert extracted == '<div class="box">content</div>'

    def test_simple_div_with_text(self):
        """Full range includes text content."""
        dom = '<div>hello world</div>'
        prefix = '<div>hello world</div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        assert dom[result[0]:result[1]] == '<div>hello world</div>'

    def test_div_with_attributes(self):
        """Full range for element with multiple attributes."""
        dom = '<html><body><div id="main" class="container" style="color:red">stuff</div></body></html>'
        prefix = '<div id="main" class="container" style="color:red">stuff</div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<div id="main" class="container" style="color:red">stuff</div>'

    def test_empty_div(self):
        """Full range for empty div still includes closing tag."""
        dom = '<html><body><div class="empty"></div></body></html>'
        prefix = '<div class="empty"></div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<div class="empty"></div>'

    def test_span_element(self):
        """Works for non-div elements too."""
        dom = '<p>Some <span class="highlight">text</span> here</p>'
        prefix = '<span class="highlight">text</span>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        assert dom[result[0]:result[1]] == '<span class="highlight">text</span>'


# ── Nested element tests ─────────────────────────────────────────────

class TestNestedElements:
    def test_nested_same_tag(self):
        """Finds correct closing tag when same-name tags are nested."""
        dom = '<div class="outer"><div class="inner">content</div></div>'
        prefix = '<div class="outer"><div class="inner">content</div></div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<div class="outer"><div class="inner">content</div></div>'

    def test_nested_inner_element(self):
        """Finds the inner element when searching for it specifically."""
        dom = '<div class="outer"><div class="inner">content</div></div>'
        prefix = '<div class="inner">content</div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<div class="inner">content</div>'

    def test_deeply_nested(self):
        """Handles deep nesting (3+ levels of same tag)."""
        dom = '<div class="a"><div class="b"><div class="c">deep</div></div></div>'
        prefix = '<div class="a"><div class="b"><div class="c">deep</div></div></div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == dom  # The whole thing

    def test_nested_middle_level(self):
        """Correctly finds middle-level element in 3-level nesting."""
        dom = '<div class="a"><div class="b"><div class="c">deep</div></div></div>'
        prefix = '<div class="b"><div class="c">deep</div></div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<div class="b"><div class="c">deep</div></div>'

    def test_sibling_elements(self):
        """Correctly finds first sibling, not extending into the second."""
        dom = '<div class="a">first</div><div class="b">second</div>'
        prefix = '<div class="a">first</div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<div class="a">first</div>'

    def test_mixed_nested_tags(self):
        """Nested elements of different types don't confuse depth tracking."""
        dom = '<div class="card"><h2>Title</h2><p>Body text</p></div>'
        prefix = '<div class="card"><h2>Title</h2><p>Body text</p></div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<div class="card"><h2>Title</h2><p>Body text</p></div>'


# ── Void/self-closing element tests ──────────────────────────────────

class TestVoidElements:
    def test_img_tag(self):
        """<img> is a void element — return just the tag."""
        dom = '<div><img src="photo.jpg" alt="pic"><p>caption</p></div>'
        prefix = '<img src="photo.jpg" alt="pic">'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        # Should be just the img tag, not extending to </div>
        assert extracted == '<img src="photo.jpg" alt="pic">'

    def test_br_tag(self):
        """<br> is a void element."""
        dom = '<p>line one<br>line two</p>'
        prefix = '<br>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<br>'

    def test_hr_tag(self):
        """<hr> is a void element."""
        dom = '<div><hr></div>'
        prefix = '<hr>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<hr>'

    def test_input_tag(self):
        """<input> is a void element."""
        dom = '<form><input type="text" name="q"></form>'
        prefix = '<input type="text" name="q">'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<input type="text" name="q">'

    def test_self_closing_syntax(self):
        """Self-closing syntax (e.g., <img />) should be handled."""
        dom = '<div><img src="pic.jpg" /></div>'
        prefix = '<img src="pic.jpg" />'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<img src="pic.jpg" />'

    def test_meta_tag(self):
        """<meta> is a void element."""
        dom = '<head><meta charset="utf-8"><title>Test</title></head>'
        prefix = '<meta charset="utf-8">'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == '<meta charset="utf-8">'


# ── No match tests ───────────────────────────────────────────────────

class TestNoMatch:
    def test_no_match_returns_none(self):
        """Returns None when element is not found in DOM."""
        dom = '<div class="box">content</div>'
        prefix = '<span class="nonexistent">nothing</span>'
        result = find_element_in_dom(dom, prefix)
        assert result is None

    def test_empty_dom(self):
        """Returns None for empty DOM."""
        result = find_element_in_dom('', '<div>something</div>')
        assert result is None

    def test_empty_prefix(self):
        """Returns None for empty prefix."""
        result = find_element_in_dom('<div>content</div>', '')
        assert result is None


# ── Fallback behavior tests ──────────────────────────────────────────

class TestFallback:
    def test_prefix_partial_match_opening_tag(self):
        """When prefix is truncated (outerHTML[:500]), still finds element.
        The prefix may not contain the closing tag, but the function
        should find the full range in the DOM."""
        dom = '<div class="card"><h2>Long title here</h2><p>Long paragraph content goes here</p></div>'
        # Simulate a prefix that's only the first part of outerHTML
        prefix = '<div class="card"><h2>Long title'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        # Should find the full div including closing tag
        assert extracted == '<div class="card"><h2>Long title here</h2><p>Long paragraph content goes here</p></div>'

    def test_opening_tag_regex_fallback(self):
        """When exact prefix doesn't match but opening tag pattern does."""
        dom = '<html><body><div class="box box-1">different content</div></body></html>'
        # Prefix has different content than what's in the DOM
        prefix = '<div class="box box-1">original content that changed</div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        # Should find via opening tag pattern and include full range
        assert extracted == '<div class="box box-1">different content</div>'

    def test_closing_tag_not_found_falls_back_to_opening(self):
        """If closing tag can't be found (malformed HTML), fall back to opening tag only."""
        # This is malformed HTML - the div is never closed
        dom = '<div class="broken">content<span>nested'
        prefix = '<div class="broken">content<span>nested'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        # Should at least return the opening tag range as fallback
        start, end = result
        assert dom[start:end].startswith('<div class="broken">')


# ── Integration-style tests ──────────────────────────────────────────

class TestIntegration:
    def test_realistic_box_layout_element(self):
        """Test with a realistic box-layout DOM structure."""
        dom = """<html><head><style>
body { margin: 0; }
.container { display: flex; gap: 10px; }
.box { width: 100px; height: 100px; }
</style></head><body>
<div class="container">
  <div class="box box-1" style="background: red"></div>
  <div class="box box-2" style="background: blue"></div>
</div>
</body></html>"""
        prefix = '<div class="container">\n  <div class="box box-1"'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        # Should contain the full container div including both children
        assert '<div class="container">' in extracted
        assert '</div>' in extracted
        assert 'box-1' in extracted
        assert 'box-2' in extracted

    def test_element_with_void_children(self):
        """Element containing void elements should work correctly."""
        dom = '<div class="form-group"><label>Name</label><input type="text"><br><span>hint</span></div>'
        prefix = '<div class="form-group"><label>Name</label><input type="text"><br><span>hint</span></div>'
        result = find_element_in_dom(dom, prefix)
        assert result is not None
        extracted = dom[result[0]:result[1]]
        assert extracted == dom
