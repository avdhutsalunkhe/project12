"""
Refusal Engine — guards against out-of-scope and adversarial inputs.

Blocks:
  1. Prompt injection / jailbreak attempts
  2. General hiring / interview / HR advice unrelated to SHL assessments
  3. Legal, compliance, or salary/compensation advice
  4. Attempts to elicit system internals or override persona

Usage:
    from app.services.refusal_engine import refusal_engine

    result = refusal_engine.check(user_message)
    if result.should_refuse:
        return result.reply
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Result Schema
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RefusalResult:
    """Outcome of a refusal check."""
    should_refuse: bool
    category: Optional[str] = None   # e.g. "prompt_injection", "legal_advice"
    reply: Optional[str] = None      # Ready-to-send refusal message


# ═══════════════════════════════════════════════════════════════
#  Rule Definitions
# ═══════════════════════════════════════════════════════════════

# Each rule: (category_label, compiled_regex_list, reply_text)
# Rules are evaluated in order; first match wins.
_RULES: List[Tuple[str, List[re.Pattern], str]] = []


def _rule(category: str, patterns: List[str], reply: str) -> None:
    """Register a refusal rule."""
    compiled = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]
    _RULES.append((category, compiled, reply))


# ── 1. Prompt Injection / Jailbreak ──────────────────────────
_rule(
    category="prompt_injection",
    patterns=[
        # Classic role-override attempts
        r"\bignore\s+(all\s+)?(previous|prior|above|earlier|your)\s+(instructions?|prompt|rules?|constraints?|guidelines?)\b",
        r"\bforget\s+(everything|all|your\s+instructions?|your\s+rules?)\b",
        r"\bact\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:a\s+)?(?:different|another|new|unrestricted|jailbreak)\b",
        r"\byou\s+are\s+now\s+(?:a\s+)?(?:DAN|GPT|evil|unrestricted|uncensored|free)\b",
        r"\bDAN\s+mode\b|\bjailbreak\b|\bdo\s+anything\s+now\b",
        # System prompt leaking
        r"\brepeat\s+(your\s+)?(system\s+)?prompt\b",
        r"\bshow\s+(me\s+)?(your\s+)?(system\s+|original\s+)?instructions?\b",
        r"\bwhat\s+(are\s+)?your\s+(system\s+)?instructions?\b",
        r"\bprint\s+(your\s+)?(system\s+)?prompt\b",
        r"\breveal\s+(your\s+)?(hidden\s+)?(prompt|instructions?|rules?)\b",
        r"\btell\s+me\s+your\s+(system\s+)?prompt\b",
        # Persona override
        r"\bpretend\s+(you('re|\s+are)\s+)?(?:not\s+an?\s+AI|human|a\s+different)\b",
        r"\byou\s+have\s+no\s+(restrictions?|limits?|rules?|constraints?)\b",
        r"\bdisable\s+(your\s+)?(safety|filter|restriction|constraint)\b",
        r"\boverride\s+(your\s+)?(safety|filter|restriction|instruction)\b",
        # Encoding tricks
        r"base64\s*:\s*[A-Za-z0-9+/=]{20,}",
        r"\btranslate\s+(?:and\s+)?execute\b",
    ],
    reply=(
        "I'm here to help with SHL assessment selection only. "
        "I can't respond to that type of request."
    ),
)

# ── 2. Legal / Compliance / Discrimination advice ────────────
_rule(
    category="legal_advice",
    patterns=[
        # Compliance / regulatory bodies
        r"\b(?:employment|labour|labor|discrimination|EEOC|OFCCP|ADA|Title\s+VII|GDPR|adverse\s+impact)\s+(?:law|complian|regulat)",
        r"\bEEOC\b|\bOFCCP\b|\bTitle\s+VII\b|\badverse\s+impact\b",
        r"\bis\s+(?:this\s+)?(?:assessment|test|hiring\s+process)\s+(?:legal|compliant|lawful|discriminatory)\b",
        r"\bcan\s+(?:I|we)\s+(?:legally|lawfully)\s+(?:use|administer|require)\b",
        r"\b(?:legal|compliance|regulatory)\s+(?:risk|liability|requirement|obligation)\b",
        r"\blegal\s+(?:risk|liability|exposure|concern)\b",
        r"\bwhat\s+are\s+the\s+(?:legal|compliance)\s+(?:risks?|implications?)\b",
        r"\blawsuit\b|\blitigat\b|\bcourt\b|\bsue\b|\bsued\b",
        r"\bprotected\s+class\b|\bequal\s+opportunity\b|\bfair\s+(?:hiring|employment)\s+law\b",
        r"\bI\s+(?:need|want)\s+(?:legal|a\s+lawyer|an?\s+attorney)\s+(?:advice|opinion|guidance)\b",
    ],
    reply=(
        "I'm not able to provide legal or compliance advice. "
        "Please consult your legal team or HR counsel for guidance on employment law and regulatory requirements. "
        "I can help you find the right SHL assessments for your role."
    ),
)

# ── 3. Salary / Compensation advice ──────────────────────────
_rule(
    category="salary_advice",
    patterns=[
        r"\b(?:salary|salaries|compensation|pay\s+scale|pay\s+grade|wage|remuneration|bonus|package|CTC|cost\s+to\s+company)\b",
        r"\bhow\s+much\s+(?:should\s+I\s+pay|to\s+pay|does\s+(?:a\s+)?[\w\s]+earn|is\s+(?:a\s+)?[\w\s]+paid)\b",
        r"\bmarket\s+rate\b|\bbenchmark\s+(?:pay|salary|compensation)\b",
        r"\bwhat\s+is\s+the\s+(?:average|typical|standard)\s+(?:salary|pay|compensation)\b",
    ],
    reply=(
        "Salary and compensation advice is outside my scope. "
        "I focus exclusively on SHL assessment recommendations. "
        "Try a resource like Glassdoor or your HR compensation team for pay benchmarks."
    ),
)

# ── 4. General Hiring / Interview Advice ─────────────────────
_rule(
    category="unrelated_hiring_advice",
    patterns=[
        # Interview coaching
        r"\bhow\s+(?:do\s+I|to|should\s+I|can\s+I)\s+(?:prepare\s+for|ace|pass|crack|nail|answer)\s+(?:an?\s+)?interview\b",
        r"\binterview\s+(?:tips?|tricks?|questions?|preparation|coaching|guide)\b",
        r"\bcommon\s+interview\s+questions?\b",
        r"\bwhat\s+(?:to\s+say|to\s+answer|should\s+I\s+say)\s+in\s+(?:an?\s+)?interview\b",
        # Resume / CV
        r"\b(?:how\s+to\s+(?:write|improve|fix|update|create))\s+(?:a\s+)?(?:resume|cv|cover\s+letter)\b",
        r"\b(?:resume|cv)\s+(?:template|example|sample|tips?|review|format)\b",
        # Job search
        r"\bhow\s+(?:do\s+I|to|can\s+I)\s+(?:find|get|land)\s+(?:a\s+)?job\b",
        r"\b(?:job\s+search|career\s+advice|career\s+coaching|career\s+path)\b",
        r"\blinkedin\s+(?:profile|tips?|optimization)\b",
        # Background checks / onboarding (unrelated to assessments)
        r"\bbackground\s+check\b",
        r"\bemployee\s+onboarding\b|\bonboarding\s+process\b",
        # Recruiter / talent strategy (not assessment-related)
        r"\bhow\s+(?:do\s+I|to|should\s+I)\s+source\s+candidates?\b",
        r"\bboolean\s+search\b|\bcandidate\s+sourcing\b|\btalent\s+acquisition\s+strateg\b",
    ],
    reply=(
        "That's outside what I can help with — I specialise in SHL assessment recommendations only. "
        "Tell me about the role you're hiring for and I'll suggest the most relevant assessments."
    ),
)

# ── 5. Completely Off-Topic (catch-all for very distant topics) ──
_rule(
    category="off_topic",
    patterns=[
        r"\b(?:recipe|weather|stock\s+price|sports?\s+(?:score|result)|movie|film|song|music|game|travel|hotel|flight)\b",
        r"\bwrite\s+(?:me\s+)?(?:a\s+)?(?:poem|story|essay|code|script|email)\b(?!.*assessment)",
        r"\b(?:politics?|religion|philosophy|dating|relationship)\b",
        # Geography / general knowledge
        r"\bwhat\s+is\s+the\s+(?:capital|population|currency|area|size)\s+of\s+\w",
        r"what.{0,10}capital.{0,10}of\s+\w",
        r"\bwho\s+(?:is|was)\s+the\s+(?:president|prime\s+minister|king|queen)\s+of\b",
        r"\btranslate\s+(?:this|the\s+following)?\s*(?:to|into|from)\s+\w+\b(?!.*assessment)",
        r"\b(?:ChatGPT|GPT-4|Claude|Gemini|OpenAI|Anthropic)\b",
    ],
    reply=(
        "I'm the SHL Assessment Recommender — I can only help you select and compare SHL assessments. "
        "What role or skills are you looking to evaluate?"
    ),
)


# ═══════════════════════════════════════════════════════════════
#  Engine
# ═══════════════════════════════════════════════════════════════

class RefusalEngine:
    """
    Evaluates user input against all refusal rules.

    Call .check(text) before processing any user message.
    """

    def check(self, text: str) -> RefusalResult:
        """
        Check whether a message should be refused.

        Returns a RefusalResult. If should_refuse is True, the `reply`
        field contains a ready-to-send message.
        """
        if not text or not text.strip():
            return RefusalResult(should_refuse=False)

        for category, patterns, reply in _RULES:
            for pattern in patterns:
                if pattern.search(text):
                    logger.warning(
                        "Refusal triggered: category=%s, pattern=%s",
                        category,
                        pattern.pattern[:60],
                    )
                    return RefusalResult(
                        should_refuse=True,
                        category=category,
                        reply=reply,
                    )

        return RefusalResult(should_refuse=False)

    def check_messages(self, messages: list) -> RefusalResult:
        """
        Check only the most recent user message in a messages[] list.

        This avoids false positives from old, already-approved messages.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content", "").strip():
                return self.check(msg["content"])
        return RefusalResult(should_refuse=False)


# Module-level singleton
refusal_engine = RefusalEngine()
