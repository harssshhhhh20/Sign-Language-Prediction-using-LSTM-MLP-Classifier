"""Assemble committed signs into readable text.

Sign languages are not word-for-word encodings of spoken languages: ASL has
its own grammar, drops copulas and articles, and marks tense with time signs
rather than inflection. So a raw gloss stream reads as
``I NAME WHAT`` rather than "What is my name?".

This module keeps the two representations separate. The gloss stream is what
the model actually recognised and is never altered; the readable line is a
best-effort rendering of it. Anything that changes wording is presentational,
and the gloss stays available so a user can see what was actually signed.

Fingerspelled letters arriving from the static pathway are accumulated into
a word and flushed when a word sign arrives or the signer pauses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# Glosses that are recognised as signs but should render differently in text.
GLOSS_TEXT = {
    "i-me": "I",
    "he-she": "he",
    "thank-you": "thank you",
    "please-repeat": "please repeat",
    "slow-down": "slow down",
    "__idle__": "",
}

# Signs that carry punctuation rather than a word.
TERMINATORS = {"finished", "finish"}
QUESTION_WORDS = {"what", "where", "when", "who", "why", "how", "which"}


@dataclass
class SentenceBuilder:
    """Accumulates committed glosses and fingerspelled letters into text."""

    max_glosses: int = 40
    letter_timeout: float = 2.0        # flush a spelled word after this pause

    glosses: list[str] = field(default_factory=list)
    _letters: list[str] = field(default_factory=list)
    _last_letter_time: float = 0.0

    # ------------------------------------------------------------- inputs

    def add_gloss(self, gloss: str) -> None:
        """A word-level sign was committed."""
        self.flush_letters()
        if gloss in GLOSS_TEXT and not GLOSS_TEXT[gloss]:
            return
        self.glosses.append(gloss)
        del self.glosses[: max(0, len(self.glosses) - self.max_glosses)]

    def add_letter(self, letter: str) -> None:
        """A fingerspelled letter was committed."""
        now = time.time()
        if self._letters and now - self._last_letter_time > self.letter_timeout:
            self.flush_letters()
        if letter.upper() == "SPACE":
            self.flush_letters()
        else:
            self._letters.append(letter.upper())
        self._last_letter_time = now

    def tick(self) -> None:
        """Call each frame so a spelled word flushes on pause, not only on
        the next sign."""
        if (self._letters
                and time.time() - self._last_letter_time > self.letter_timeout):
            self.flush_letters()

    def flush_letters(self) -> None:
        if self._letters:
            self.glosses.append("".join(self._letters).lower())
            self._letters.clear()

    def clear(self) -> None:
        self.glosses.clear()
        self._letters.clear()

    def undo(self) -> None:
        if self._letters:
            self._letters.pop()
        elif self.glosses:
            self.glosses.pop()

    # ------------------------------------------------------------ outputs

    @property
    def pending_word(self) -> str:
        return "".join(self._letters)

    def gloss_line(self) -> str:
        """What was actually recognised, unaltered."""
        out = " ".join(g.upper() for g in self.glosses)
        if self._letters:
            out = (out + " " + "".join(self._letters)).strip()
        return out

    def text_line(self) -> str:
        """Best-effort readable rendering. Presentational only."""
        if not self.glosses and not self._letters:
            return ""

        words = []
        for g in self.glosses:
            if g in TERMINATORS:
                if words:
                    words[-1] = words[-1] + "."
                continue
            words.append(GLOSS_TEXT.get(g, g.replace("-", " ")))

        if self._letters:
            words.append("".join(self._letters).lower())
        if not words:
            return ""

        text = " ".join(words)
        text = text[0].upper() + text[1:]

        if not text.endswith("."):
            starts_question = self.glosses and self.glosses[0] in QUESTION_WORDS
            has_question = any(g in QUESTION_WORDS for g in self.glosses)
            text += "?" if (starts_question or has_question) else ""
        return text
