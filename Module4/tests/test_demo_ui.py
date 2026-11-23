"""demo.py front end (no models): status line, tab choice, --auth parsing, UI builds."""
import importlib.util
import os

import pytest

from m4_settings import REPO


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo_ui", os.path.join(REPO, "demo.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_status_for_a_finished_run(demo):
    assert demo.run_status(">> Module 1 — perception on a.jpg\n   3 garments\n>> done in 4.2s") \
        == "✅ Done in 4.2s"


def test_status_shows_why_a_run_stopped(demo):
    log = ">> Module 1 — perception on dog.jpg\n   No person found: this looks like a dog."
    assert demo.run_status(log) == "⚠️ No person found: this looks like a dog."


def test_status_for_an_error_points_to_the_log(demo):
    log = ">> Module 1 — perception on a.jpg\n\n[error] boom\nTraceback (most recent call last):\n  x"
    assert demo.run_status(log) == "❌ Error: boom (details in the Log tab)"


def test_status_for_a_message_without_a_run(demo):
    assert demo.run_status("Please upload or paste an outfit photo first.") \
        == "⚠️ Please upload or paste an outfit photo first."
    assert demo.run_status("") == ""


def test_warnings_get_the_orange_box_and_nothing_else_does(demo):
    warn = demo.status_html(">> Module 1 — perception on dog.jpg\n   No person <found> here.")
    assert warn == '<div class="warn-box">⚠️ No person &lt;found&gt; here.</div>'
    assert demo.status_html(">> done in 4.2s") == "<div>✅ Done in 4.2s</div>"
    assert "warn-box" not in demo.status_html("\n[error] boom")
    assert demo.status_html("") == ""


def test_single_module_runs_open_their_tab(demo):
    assert [demo.tab_for(m) for m in (demo.MODULE_1, demo.MODULE_2, demo.MODULE_3, demo.MODULE_4)] \
        == ["m1", "m2", "m3", "m4"]
    assert demo.tab_for(demo.ALL) is None


def test_auth_parsing(demo):
    assert demo.parse_auth("team:s3cret:x") == ("team", "s3cret:x")
    assert demo.parse_auth(None) is None
    for bad in ("nocolon", ":pw", "user:"):
        with pytest.raises(ValueError):
            demo.parse_auth(bad)


@pytest.mark.parametrize("advanced", [False, True])
def test_ui_builds_and_hides_advanced_controls(demo, advanced):
    ui = demo.build_ui(advanced=advanced)
    acc = [b for b in ui.blocks.values() if type(b).__name__ == "Accordion" and "advanced" in str(b.label)]
    assert len(acc) == 1 and acc[0].visible is advanced
