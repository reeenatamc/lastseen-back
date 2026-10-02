"""Redaction tests: structured identifiers are hidden, everyday numbers are not."""
import pytest

from app.analyzers.redaction import redact


# ── e-mails ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("escríbeme a ana.perez+x@gmail.com ya", "escríbeme a [email] ya"),
    ("mail: juan_89@correo.com.ec.", "mail: [email]."),
])
def test_email_redacted(text, expected):
    assert redact(text) == expected


def test_at_sign_without_address_untouched():
    assert redact("nos vemos @ las 5") == "nos vemos @ las 5"


# ── links ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("mira https://youtu.be/abc123?t=5 jaja", "mira [link] jaja"),
    ("http://example.com/a/b", "[link]"),
    ("entra a www.tienda.com.", "entra a [link]."),
    ("(www.foo.org/x)", "([link])"),
])
def test_link_redacted(text, expected):
    assert redact(text) == expected


def test_plain_word_with_dot_is_not_a_link():
    assert redact("ok.nos vemos") == "ok.nos vemos"


# ── numbers ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "0991234567",                 # mobile
    "1712345678",                 # 10-digit national ID
    "1712345678001",              # 13-digit tax ID
    "+593 99 123 4567",           # international with spaces
    "+593-99-123-4567",           # hyphens
    "(02) 245-6789",              # parentheses
    "099.123.4567",               # dots
    "4111 1111 1111 1111",        # card
    "123456",                     # exactly six digits
])
def test_number_redacted(text):
    assert redact(text) == "[number]"
    assert redact(f"mi dato es {text} gracias") == "mi dato es [number] gracias"


@pytest.mark.parametrize("text", [
    "nos vemos a las 10:30",
    "el 12/05 es el cumple",
    "pagué $25.50 ayer",
    "son 1500 en total",
    "nací en 1998",
    "12345",                      # five digits
    "son las 10:30:45",
    "el 12-05-2024 fue lindo",    # full date with separators
    "4 5 6",                      # few digits with spaces
])
def test_small_numbers_untouched(text):
    assert redact(text) == text


# ── mixed ─────────────────────────────────────────────────────────────────────

def test_several_items_in_one_message():
    text = "llámame al 099 123 4567 o escribe a a@b.com, mira www.x.com a las 10:30 que cuesta $25.50"
    assert redact(text) == "llámame al [number] o escribe a [email], mira [link] a las 10:30 que cuesta $25.50"


def test_email_with_digits_not_split_into_number():
    assert redact("a1234567@mail.com") == "[email]"


def test_empty_and_plain_text():
    assert redact("") == ""
    assert redact("te quiero mucho ❤️") == "te quiero mucho ❤️"


# ── Performance (ReDoS) ───────────────────────────────────────────────────────

def test_long_input_without_spaces_is_linear():
    import time

    hostile = (("a@" * 20 + "www." * 20) * 834)[:100_000]  # no spaces
    assert len(hostile) == 100_000
    t0 = time.perf_counter()
    redact(hostile)
    assert time.perf_counter() - t0 < 0.5


def test_long_local_part_without_at_is_linear():
    import time

    t0 = time.perf_counter()
    redact("a" * 100_000)
    redact("a." * 50_000)
    assert time.perf_counter() - t0 < 0.5


def test_link_trailing_punctuation_kept_outside_marker():
    assert redact("mira www.x.com.") == "mira [link]."
    assert redact("www.,") == "www.,"
