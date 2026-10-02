from copy import deepcopy

from app.services.access import LOCKED_SECTIONS, build_preview, build_teaser

# ── Helpers ───────────────────────────────────────────────────────────────────

def _full() -> dict:
    return {
        "temporal": {
            "overview": {"total_messages": 100, "first_date": "2026-01-01"},
            "response_time": {"median": 30},
            "initiative_balance": {"Alice": 0.6},
            "activity_patterns": {"hours": [1, 2]},
            "conversation_gaps": [{"from": "2026-02-01"}],
            "message_length": {"avg": 5},
            "response_decay": {
                "turning_point": {"date": "2026-03-04", "person": "Alice"},
                "closing_phase": {"start": "2026-04-01"},
            },
            "delayed_replies": [],
        },
        "sentiment": {"recent": {"shift": -0.4, "from": "2026-03-01"}, "per_person": {}},
        "conflict": {"episodes": [{"date": "2026-02-02"}, {"date": "2026-02-09"}]},
        "narrative": {
            "resumen": "A short summary",
            "dinamica": "dyn",
            "punto_de_quiebre": "pq",
            "estado_actual": "now",
            "reflexion": "ref",
        },
    }


# ── build_preview ─────────────────────────────────────────────────────────────

def test_preview_keeps_only_allowed_sections():
    preview = build_preview(_full())
    assert set(preview) == {"temporal", "narrative"}
    assert set(preview["temporal"]) == {
        "overview", "response_time", "activity_patterns", "message_length"
    }


def test_preview_narrative_keeps_only_resumen():
    narrative = build_preview(_full())["narrative"]
    assert narrative["resumen"] == "A short summary"
    for key in ("dinamica", "punto_de_quiebre", "estado_actual", "reflexion"):
        assert narrative[key] is None


def test_preview_does_not_mutate_input():
    data = _full()
    snapshot = deepcopy(data)
    preview = build_preview(data)
    preview["temporal"]["overview"]["total_messages"] = -1
    assert data == snapshot


def test_preview_keeps_error_analyzers():
    data = {"temporal": {"error": "insufficient_data"}, "narrative": {"error": "boom"}}
    preview = build_preview(data)
    assert preview["temporal"] == {"error": "insufficient_data"}
    assert preview["narrative"] == {"error": "boom"}


def test_preview_tolerates_missing_analyzers():
    assert build_preview({}) == {}
    assert build_preview({"sentiment": {"error": "x"}}) == {}


# ── build_teaser ──────────────────────────────────────────────────────────────

def test_teaser_reports_facts():
    teaser = build_teaser(_full())
    assert teaser == {
        "conflict_episodes": 2,
        "turning_point_detected": True,
        "closing_phase_detected": True,
    }


def test_teaser_contains_no_dates_or_names():
    text = str(build_teaser(_full()))
    for leaked in ("2026", "Alice", "date", "person"):
        assert leaked not in text


def test_teaser_with_no_data():
    teaser = build_teaser({"conflict": {"error": "insufficient_data"}, "temporal": {"error": "x"}})
    assert teaser == {
        "conflict_episodes": None,
        "turning_point_detected": False,
        "closing_phase_detected": False,
    }


def test_locked_sections_cover_hidden_analyzers():
    assert "sentiment" in LOCKED_SECTIONS
    assert "conflict" in LOCKED_SECTIONS


def test_locked_sections_name_sentiment_and_narrative_whole():
    assert "sentiment" in LOCKED_SECTIONS
    assert "narrative" in LOCKED_SECTIONS


def test_preview_of_result_without_llm_analyzers():
    data = {k: v for k, v in _full().items() if k in ("temporal", "conflict")}
    assert set(build_preview(data)) == {"temporal"}
