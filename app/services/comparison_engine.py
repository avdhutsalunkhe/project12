"""
Assessment Comparison Engine — compares SHL assessments using catalog data only.

Compares across five dimensions:
  - Purpose      : What the assessment is designed to measure / its goal
  - Type         : Test type (Ability, Personality, Knowledge, Simulation, etc.)
  - Duration     : Time commitment in minutes
  - Skills       : Technical / knowledge domains evaluated (extracted from description)
  - Use Cases    : Job levels, titles, and scenarios where the assessment applies

All data is sourced exclusively from catalog.json — no external LLM calls,
no hallucination.

Usage:
    from app.services.comparison_engine import comparison_engine

    result = comparison_engine.compare(["Java (New)", "Python (New)"])
    result = comparison_engine.compare_by_names(["Java (New)", "Angular 6 (New)"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════

@dataclass
class AssessmentProfile:
    """Structured profile of a single assessment derived from catalog data."""
    name: str
    url: Optional[str]
    test_type: Optional[str]
    test_type_code: Optional[str]
    duration_minutes: Optional[int]
    job_levels: List[str]
    languages: List[str]
    description: str

    # Derived from description
    purpose: str = ""
    skills: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)


@dataclass
class DimensionComparison:
    """Comparison result for a single dimension."""
    dimension: str
    values: Dict[str, Any]   # assessment_name -> value
    notes: str = ""           # Human-readable insight


@dataclass
class ComparisonResult:
    """Full comparison result between two or more assessments."""
    assessments: List[str]             # Names of compared assessments
    profiles: List[AssessmentProfile]  # Full profiles
    dimensions: List[DimensionComparison]
    summary: str                       # Concise text summary
    not_found: List[str] = field(default_factory=list)  # Names not found


# ═══════════════════════════════════════════════════════════════
#  Dimension Extractors
# ═══════════════════════════════════════════════════════════════

# Keyword → skill label mapping for description parsing
_SKILL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bnumerical\b", re.I), "Numerical Reasoning"),
    (re.compile(r"\bverbal\s+reason\b", re.I), "Verbal Reasoning"),
    (re.compile(r"\binductive\b", re.I), "Inductive Reasoning"),
    (re.compile(r"\bdeductive\b", re.I), "Deductive Reasoning"),
    (re.compile(r"\bsituational\s+judg\b", re.I), "Situational Judgment"),
    (re.compile(r"\bcognitive\b", re.I), "Cognitive Ability"),
    (re.compile(r"\bpersonality\b", re.I), "Personality"),
    (re.compile(r"\bleadership\b", re.I), "Leadership"),
    (re.compile(r"\bsales\b", re.I), "Sales"),
    (re.compile(r"\bcustomer\s+(?:service|focus)\b", re.I), "Customer Service"),
    (re.compile(r"\bteam\s*work\b|collaboration", re.I), "Teamwork"),
    (re.compile(r"\bproblem\s+solv\b", re.I), "Problem Solving"),
    (re.compile(r"\bdecision\s+mak\b", re.I), "Decision Making"),
    (re.compile(r"\bcritical\s+think\b", re.I), "Critical Thinking"),
    (re.compile(r"\bmanagement\b", re.I), "Management"),
    (re.compile(r"\bsupervision\b", re.I), "Supervision"),
    (re.compile(r"\bcommunicat\b", re.I), "Communication"),
    (re.compile(r"\bwriting\b|written", re.I), "Writing"),
    (re.compile(r"\bdata\s+entry\b", re.I), "Data Entry"),
    (re.compile(r"\bnumerical\s+data\b|\bfinancial\s+records?\b", re.I), "Financial Data"),
    (re.compile(r"\bjava\b(?!\s*script)", re.I), "Java"),
    (re.compile(r"\bpython\b", re.I), "Python"),
    (re.compile(r"\bjavascript\b|js\b", re.I), "JavaScript"),
    (re.compile(r"\bc\+\+\b|cpp\b", re.I), "C++"),
    (re.compile(r"\bc#\b|\.net\b", re.I), "C# / .NET"),
    (re.compile(r"\bsql\b", re.I), "SQL"),
    (re.compile(r"\baws\b|amazon web\b", re.I), "AWS"),
    (re.compile(r"\bazure\b", re.I), "Azure"),
    (re.compile(r"\bdocker\b", re.I), "Docker"),
    (re.compile(r"\bkubernetes\b|k8s\b", re.I), "Kubernetes"),
    (re.compile(r"\bagile\b", re.I), "Agile"),
    (re.compile(r"\bscrum\b", re.I), "Scrum"),
    (re.compile(r"\bmachine\s+learning\b", re.I), "Machine Learning"),
    (re.compile(r"\bdata\s+science\b", re.I), "Data Science"),
    (re.compile(r"\baccounting\b", re.I), "Accounting"),
    (re.compile(r"\bbookkeeping\b", re.I), "Bookkeeping"),
    (re.compile(r"\bsap\b", re.I), "SAP"),
    (re.compile(r"\bsalesforce\b", re.I), "Salesforce"),
    (re.compile(r"\bsecurity\b", re.I), "Security"),
    (re.compile(r"\bnetwork\b", re.I), "Networking"),
    (re.compile(r"\bembedded\b", re.I), "Embedded Systems"),
    (re.compile(r"\baerospace\b|aeronautic", re.I), "Aerospace"),
    (re.compile(r"\bmechanical\b", re.I), "Mechanical Engineering"),
    (re.compile(r"\belectrical\b|electronics\b", re.I), "Electrical Engineering"),
    (re.compile(r"\bcivil\s+eng\b", re.I), "Civil Engineering"),
    (re.compile(r"\bpharmac\b", re.I), "Pharmaceutical"),
    (re.compile(r"\bnursing\b|healthcare\b", re.I), "Healthcare"),
    (re.compile(r"\bhr\b|human\s+resource", re.I), "Human Resources"),
    (re.compile(r"\bmarketing\b", re.I), "Marketing"),
    (re.compile(r"\bproject\s+management\b", re.I), "Project Management"),
]


def _extract_skills(description: str) -> List[str]:
    """Extract skill labels from a description using keyword matching."""
    seen: Set[str] = set()
    skills: List[str] = []
    for pattern, label in _SKILL_PATTERNS:
        if label not in seen and pattern.search(description):
            skills.append(label)
            seen.add(label)
    return skills


def _extract_purpose(description: str) -> str:
    """
    Extract the first 1-2 sentences as the 'purpose' of the assessment.

    Falls back to a truncated description if sentence detection fails.
    """
    if not description:
        return "Not specified."

    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", description.strip())
    if sentences:
        # First sentence is usually the purpose
        purpose = sentences[0].strip()
        # If short, grab the second sentence too
        if len(purpose) < 80 and len(sentences) > 1:
            purpose = purpose + " " + sentences[1].strip()
        return purpose

    return description[:250]


def _extract_use_cases(assessment: Dict[str, Any]) -> List[str]:
    """
    Build a list of use cases from job levels + title mentions in description.
    """
    use_cases: List[str] = list(assessment.get("job_levels", []))

    # Extract mentioned job titles from description
    description = assessment.get("description", "")
    title_pattern = re.compile(
        r"Potential job titles? (?:that use this solution are|include)[:\s]+([^.]+)\.",
        re.IGNORECASE,
    )
    m = title_pattern.search(description)
    if m:
        raw = m.group(1)
        # Parse comma/and-separated titles
        titles = [t.strip() for t in re.split(r",\s*(?:and\s+)?|and\s+", raw) if t.strip()]
        use_cases.extend(titles)

    # Remove duplicates while preserving order
    seen: Set[str] = set()
    deduplicated: List[str] = []
    for uc in use_cases:
        if uc and uc.lower() not in seen:
            seen.add(uc.lower())
            deduplicated.append(uc)

    return deduplicated


def _profile(assessment: Dict[str, Any]) -> AssessmentProfile:
    """Build a full AssessmentProfile from a raw catalog dict."""
    description = assessment.get("description", "")
    return AssessmentProfile(
        name=assessment.get("name", ""),
        url=assessment.get("url"),
        test_type=assessment.get("test_type"),
        test_type_code=assessment.get("test_type_code"),
        duration_minutes=assessment.get("duration_minutes"),
        job_levels=assessment.get("job_levels", []),
        languages=assessment.get("languages", []),
        description=description,
        purpose=_extract_purpose(description),
        skills=_extract_skills(description),
        use_cases=_extract_use_cases(assessment),
    )


# ═══════════════════════════════════════════════════════════════
#  Dimension Builders
# ═══════════════════════════════════════════════════════════════

def _compare_purpose(profiles: List[AssessmentProfile]) -> DimensionComparison:
    values = {p.name: p.purpose or "Not specified" for p in profiles}

    # Check if purposes are identical or very different
    unique_purposes = set(v.lower()[:80] for v in values.values())
    if len(unique_purposes) == 1:
        note = "All assessments share a similar stated purpose."
    else:
        note = "Each assessment targets a distinct purpose."

    return DimensionComparison(
        dimension="Purpose",
        values=values,
        notes=note,
    )


def _compare_type(profiles: List[AssessmentProfile]) -> DimensionComparison:
    values = {
        p.name: (
            f"{p.test_type} ({p.test_type_code})"
            if p.test_type and p.test_type_code
            else p.test_type or p.test_type_code or "Not specified"
        )
        for p in profiles
    }

    unique_types = set(values.values())
    if len(unique_types) == 1:
        note = "All assessments are the same test type."
    else:
        note = "Assessments span different test types — consider combining for a fuller candidate view."

    return DimensionComparison(
        dimension="Type",
        values=values,
        notes=note,
    )


def _compare_duration(profiles: List[AssessmentProfile]) -> DimensionComparison:
    values = {
        p.name: (f"{p.duration_minutes} minutes" if p.duration_minutes else "Not specified")
        for p in profiles
    }

    durations = [p.duration_minutes for p in profiles if p.duration_minutes]
    if len(durations) >= 2:
        min_d, max_d = min(durations), max(durations)
        if min_d == max_d:
            note = f"All assessments take the same time ({min_d} min)."
        else:
            diff = max_d - min_d
            fastest_name = min(profiles, key=lambda p: p.duration_minutes or 9999).name
            note = (
                f"Duration ranges from {min_d} to {max_d} minutes (diff: {diff} min). "
                f"**{fastest_name}** is the shortest option."
            )
    else:
        note = "Duration information not available for all assessments."

    return DimensionComparison(
        dimension="Duration",
        values=values,
        notes=note,
    )


def _compare_skills(profiles: List[AssessmentProfile]) -> DimensionComparison:
    values = {
        p.name: (", ".join(p.skills) if p.skills else "No specific skills detected")
        for p in profiles
    }

    # Find shared and unique skills
    all_skill_sets = [set(p.skills) for p in profiles]
    if len(all_skill_sets) >= 2:
        shared = set.intersection(*all_skill_sets)
        all_union = set.union(*all_skill_sets)

        if shared:
            note = f"Shared skills: {', '.join(sorted(shared))}. "
        else:
            note = "No overlapping skill domains detected. "

        if len(all_union) > len(shared):
            note += "Assessments complement each other by covering distinct areas."
        else:
            note += "Assessments cover identical skill domains."
    else:
        note = ""

    return DimensionComparison(
        dimension="Skills Evaluated",
        values=values,
        notes=note,
    )


def _compare_use_cases(profiles: List[AssessmentProfile]) -> DimensionComparison:
    values = {
        p.name: (", ".join(p.use_cases) if p.use_cases else "Not specified")
        for p in profiles
    }

    # Check for shared job levels
    all_level_sets = [set(p.job_levels) for p in profiles]
    if len(all_level_sets) >= 2:
        shared = set.intersection(*all_level_sets)
        if shared:
            note = f"Both assessments apply to: {', '.join(sorted(shared))}."
        else:
            note = "Assessments target different seniority levels."
    else:
        note = ""

    return DimensionComparison(
        dimension="Use Cases",
        values=values,
        notes=note,
    )


# ═══════════════════════════════════════════════════════════════
#  Summary Builder
# ═══════════════════════════════════════════════════════════════

def _build_summary(profiles: List[AssessmentProfile], not_found: List[str]) -> str:
    if not profiles:
        if not_found:
            return f"Could not find assessments: {', '.join(not_found)}."
        return "No assessments to compare."

    names = [p.name for p in profiles]
    lines: List[str] = []

    lines.append(f"**Comparing {len(names)} assessments:** {' vs. '.join(names)}\n")

    # Type summary
    types = list({p.test_type for p in profiles if p.test_type})
    if len(types) == 1:
        lines.append(f"Both are **{types[0]}** assessments.")
    elif types:
        lines.append(f"Assessment types differ: {', '.join(types)}.")

    # Duration summary
    durations = [(p.name, p.duration_minutes) for p in profiles if p.duration_minutes]
    if durations:
        dur_str = " | ".join(f"{n}: {d} min" for n, d in durations)
        lines.append(f"Duration — {dur_str}.")

    # Use case overlap
    all_levels = [set(p.job_levels) for p in profiles]
    if len(all_levels) >= 2:
        shared = set.intersection(*all_levels)
        if shared:
            lines.append(f"Both suit: {', '.join(sorted(shared))}.")
        else:
            lines.append("They target different seniority levels.")

    # Not found note
    if not_found:
        lines.append(
            f"\n⚠️ Not found in catalog: {', '.join(not_found)}. "
            "Please verify the assessment name(s)."
        )

    lines.append(
        "\nTip: These can be used together for a more comprehensive candidate evaluation."
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Comparison Engine
# ═══════════════════════════════════════════════════════════════

class ComparisonEngine:
    """
    Compares SHL assessments across five dimensions using catalog data only.

    Must be called after CatalogService has been loaded.
    """

    def compare(self, names: List[str]) -> ComparisonResult:
        """
        Compare assessments by exact or fuzzy name match.

        Args:
            names: 2–5 assessment names to compare.

        Returns:
            ComparisonResult with profiles and dimension breakdowns.
        """
        from app.services.catalog_service import catalog_service

        if not catalog_service.loaded:
            logger.error("ComparisonEngine called before CatalogService is loaded")
            return ComparisonResult(
                assessments=names,
                profiles=[],
                dimensions=[],
                summary="Assessment catalog is not loaded. Please try again.",
                not_found=names,
            )

        if len(names) < 2:
            return ComparisonResult(
                assessments=names,
                profiles=[],
                dimensions=[],
                summary="Please provide at least 2 assessment names to compare.",
                not_found=[],
            )

        if len(names) > 5:
            names = names[:5]
            logger.warning("Comparison limited to 5 assessments; truncating.")

        # Resolve each name against catalog
        found: List[AssessmentProfile] = []
        not_found: List[str] = []

        for name in names:
            raw = self._resolve(name, catalog_service)
            if raw:
                found.append(_profile(raw))
            else:
                logger.warning("Assessment not found in catalog: %s", name)
                not_found.append(name)

        if not found:
            return ComparisonResult(
                assessments=names,
                profiles=[],
                dimensions=[],
                summary=f"None of the requested assessments were found in the catalog: {', '.join(not_found)}",
                not_found=not_found,
            )

        # Build dimension comparisons
        dimensions = [
            _compare_purpose(found),
            _compare_type(found),
            _compare_duration(found),
            _compare_skills(found),
            _compare_use_cases(found),
        ]

        summary = _build_summary(found, not_found)

        return ComparisonResult(
            assessments=[p.name for p in found],
            profiles=found,
            dimensions=dimensions,
            summary=summary,
            not_found=not_found,
        )

    def _resolve(
        self,
        name: str,
        catalog_service,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a name to a catalog entry.

        Strategy (in order):
          1. Exact match (case-insensitive)
          2. Prefix match
          3. TF-IDF top-1 search (fuzzy fallback)
        """
        # 1. Exact
        exact = catalog_service.get_by_name(name)
        if exact:
            return exact

        # 2. Prefix match
        name_lower = name.lower().strip()
        for a in catalog_service._assessments:
            if a.get("name", "").lower().startswith(name_lower):
                logger.info("Prefix match: '%s' → '%s'", name, a["name"])
                return a

        # 3. Fuzzy / TF-IDF fallback
        results = catalog_service.search(query=name, max_results=1)
        if results:
            matched = results[0]
            logger.info(
                "Fuzzy match: '%s' → '%s' (score=%.2f)",
                name,
                matched.get("name"),
                matched.get("match_score", 0),
            )
            return matched

        return None

    def format_as_text(self, result: ComparisonResult) -> str:
        """
        Format a ComparisonResult as a markdown-style text block
        suitable for the chat response field.
        """
        if not result.profiles:
            return result.summary

        lines: List[str] = [result.summary, ""]

        for dim in result.dimensions:
            lines.append(f"### {dim.dimension}")
            for name, value in dim.values.items():
                lines.append(f"- **{name}**: {value}")
            if dim.notes:
                lines.append(f"\n_{dim.notes}_")
            lines.append("")

        return "\n".join(lines)


# Module-level singleton
comparison_engine = ComparisonEngine()
