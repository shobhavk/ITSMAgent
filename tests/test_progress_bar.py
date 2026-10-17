"""The analysis progress bar: step-based, accessible, no emoji, and compatible with its callers."""
import re

import pytest

from ui.components import AGENT_PROGRESS_HTML, _agent_progress_html
from ui.constants import _ANALYZE_STEPS
from ui.styles import CUSTOM_CSS

_EMOJI = re.compile("[\u2190-\u2BFF\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")


@pytest.mark.parametrize("step", [1, 2, 3])
def test_steps_are_marked_done_active_todo(step):
    html = _agent_progress_html("Working", step, _ANALYZE_STEPS)
    assert html.count("agent-progress-step-item") == _ANALYZE_STEPS
    assert html.count("step-item done") == step - 1
    assert html.count("step-item active") == 1
    assert html.count("step-item todo") == _ANALYZE_STEPS - step
    assert f"Step {step} of {_ANALYZE_STEPS}" in html
    assert f'aria-valuenow="{step}"' in html and 'role="progressbar"' in html


def test_default_and_stepless_variants_render_an_indeterminate_bar():
    assert "agent-progress-fill" in AGENT_PROGRESS_HTML and "step-item" not in AGENT_PROGRESS_HTML
    assert "agent-progress-fill" in _agent_progress_html("Loading")


def test_no_emoji_and_no_old_icon_markup():
    for html in (AGENT_PROGRESS_HTML, *(_agent_progress_html("x", i, 3) for i in (1, 2, 3))):
        assert not _EMOJI.search(html)
        assert "agent-progress-icon" not in html and 'class="dots"' not in html
    assert "agent-progress-icon" not in CUSTOM_CSS and "agent-bounce" not in CUSTOM_CSS


def test_stage_text_is_escaped_and_bad_steps_are_clamped():
    assert "<script>" not in _agent_progress_html("<script>alert(1)</script>", 1, 3)
    assert "&amp;" in _agent_progress_html("Validating, categorizing & scoring", 2, 3)
    assert "Step 3 of 3" in _agent_progress_html("x", 99, 3)
    assert "Step 1 of 3" in _agent_progress_html("x", -5, 3)


def test_unlabelled_step_counts_still_render():
    html = _agent_progress_html("x", 2, 5)
    assert html.count("step-item") == 5 and "agent-progress-step-label" not in html


def test_stylesheet_defines_every_class_the_markup_uses():
    html = _agent_progress_html("x", 2, 3) + AGENT_PROGRESS_HTML
    for cls in set(re.findall(r'class="([^"]+)"', html)):
        for part in cls.split():
            if part in ("done", "active", "todo"):
                continue
            assert f".{part}" in CUSTOM_CSS, part
