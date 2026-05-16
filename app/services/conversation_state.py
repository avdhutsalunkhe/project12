"""
Stateless conversation state reconstruction.

Processes the full messages[] array on every call and deterministically
extracts a normalized ConversationState — no memory, no sessions.

Pipeline:
  1. Iterate messages oldest-to-newest (user messages only)
  2. Extract dimensions via regex + keyword matching
  3. Later messages refine/override earlier extractions
  4. Return a frozen, normalized state object
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
#  State Schema
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConversationState:
    """Normalized structured state extracted from conversation."""

    # Job role being assessed
    role: Optional[str] = None

    # Seniority level (normalized)
    seniority: Optional[str] = None

    # Technical skills mentioned
    technical_skills: List[str] = field(default_factory=list)

    # Personality / behavioral requirements
    personality_requirements: List[str] = field(default_factory=list)

    # Communication requirements
    communication_requirements: List[str] = field(default_factory=list)

    # Assessment preferences
    max_duration: Optional[int] = None
    preferred_test_type: Optional[str] = None
    language: Optional[str] = None

    # Search query built from accumulated context
    search_query: str = ""

    # Confidence: how many dimensions were extracted
    extraction_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "seniority": self.seniority,
            "technical_skills": self.technical_skills,
            "personality_requirements": self.personality_requirements,
            "communication_requirements": self.communication_requirements,
            "max_duration": self.max_duration,
            "preferred_test_type": self.preferred_test_type,
            "language": self.language,
            "search_query": self.search_query,
            "extraction_confidence": self.extraction_confidence,
        }


# ═══════════════════════════════════════════════════════════════
#  Keyword Dictionaries
# ═══════════════════════════════════════════════════════════════

# Seniority normalization (keyword -> canonical label)
_SENIORITY_MAP = {
    "intern": "Entry-Level",
    "entry": "Entry-Level",
    "entry-level": "Entry-Level",
    "entry level": "Entry-Level",
    "junior": "Entry-Level",
    "fresher": "Entry-Level",
    "graduate": "Graduate",
    "grad": "Graduate",
    "campus": "Graduate",
    "mid": "Mid-Professional",
    "mid-level": "Mid-Professional",
    "mid level": "Mid-Professional",
    "mid-senior": "Mid-Professional",
    "intermediate": "Mid-Professional",
    "senior": "Mid-Professional",
    "experienced": "Mid-Professional",
    "lead": "Front Line Manager",
    "team lead": "Front Line Manager",
    "front line": "Front Line Manager",
    "supervisor": "Supervisor",
    "manager": "Manager",
    "management": "Manager",
    "director": "Director",
    "vp": "Executive",
    "vice president": "Executive",
    "executive": "Executive",
    "c-level": "Executive",
    "cto": "Executive",
    "cfo": "Executive",
    "ceo": "Executive",
}

# Technical skills (pattern -> canonical name)
_TECHNICAL_SKILLS = {
    r"\bjava\b(?!\s*script)": "Java",
    r"\bjavascript\b|\bjs\b": "JavaScript",
    r"\btypescript\b|\bts\b": "TypeScript",
    r"\bpython\b": "Python",
    r"\bc\+\+\b|\bcpp\b": "C++",
    r"\bc#\b|\.net\b|dotnet\b": "C# / .NET",
    r"\bruby\b": "Ruby",
    r"\bphp\b": "PHP",
    r"\bswift\b": "Swift",
    r"\bkotlin\b": "Kotlin",
    r"\brust\b": "Rust",
    r"\bgo\b(?:lang)?\b": "Go",
    r"\bscala\b": "Scala",
    r"\br\b(?:\s+programming|\s+language)": "R",
    r"\bsql\b": "SQL",
    r"\bnosql\b|\bmongodb\b": "NoSQL",
    r"\breact\b(?:js)?": "React",
    r"\bangular\b": "Angular",
    r"\bvue\b(?:\.?js)?": "Vue.js",
    r"\bnode\b(?:\.?js)?": "Node.js",
    r"\bspring\b": "Spring",
    r"\bdjango\b": "Django",
    r"\bflask\b": "Flask",
    r"\bdocker\b": "Docker",
    r"\bkubernetes\b|\bk8s\b": "Kubernetes",
    r"\baws\b|\bamazon web": "AWS",
    r"\bazure\b": "Azure",
    r"\bgcp\b|\bgoogle cloud\b": "GCP",
    r"\bdevops\b": "DevOps",
    r"\bci\s*/?\s*cd\b": "CI/CD",
    r"\bmachine learning\b|\bml\b": "Machine Learning",
    r"\bdeep learning\b|\bdl\b": "Deep Learning",
    r"\bdata science\b": "Data Science",
    r"\bdata engineer": "Data Engineering",
    r"\bdata analy": "Data Analysis",
    r"\bbig data\b": "Big Data",
    r"\bhadoop\b": "Hadoop",
    r"\bspark\b": "Apache Spark",
    r"\btableau\b": "Tableau",
    r"\bpower bi\b": "Power BI",
    r"\bsalesforce\b": "Salesforce",
    r"\bsap\b": "SAP",
    r"\bagile\b": "Agile",
    r"\bscrum\b": "Scrum",
    r"\bselenium\b": "Selenium",
    r"\blinux\b": "Linux",
    r"\bmicroservices\b": "Microservices",
    r"\brest\s*(?:ful)?\s*api": "REST API",
    r"\bgraphql\b": "GraphQL",
    r"\bcyber\s*security\b|\binfo\s*sec\b": "Cybersecurity",
    r"\bnetwork": "Networking",
    r"\bembedded\b": "Embedded Systems",
    r"\bmobile\s+(?:app|dev)": "Mobile Development",
    r"\bandroid\b": "Android",
    r"\bios\b(?:\s+dev)?": "iOS",
    r"\bfrontend\b|\bfront-end\b|\bfront end\b": "Frontend",
    r"\bbackend\b|\bback-end\b|\bback end\b": "Backend",
    r"\bfull\s*stack\b": "Full Stack",
    r"\baccounting\b|\bfinance\b|\bfinancial\b": "Accounting / Finance",
    r"\bbookkeeping\b": "Bookkeeping",
    r"\bproject\s+management\b|\bpmp\b": "Project Management",
    r"\bhr\b|\bhuman\s+resource": "Human Resources",
    r"\bmarketing\b|\bseo\b|\bsem\b": "Marketing",
    r"\bnursing\b|\bhealthcare\b|\bmedical\b": "Healthcare",
    r"\belectrical\b|\belectronics\b": "Electrical/Electronics",
    r"\bmechanical\b": "Mechanical Engineering",
    r"\bcivil\b(?:\s+eng)": "Civil Engineering",
    r"\bpharmac": "Pharmaceutical",
    r"\baerospace\b|\baeronautic": "Aerospace",
}

# Personality & behavioral traits
_PERSONALITY_TRAITS = {
    r"\bleadership\b": "Leadership",
    r"\bteam\s*work\b|\bteam\s+player\b|\bcollabor": "Teamwork & Collaboration",
    r"\bproblem\s+solv": "Problem Solving",
    r"\bcritical\s+think": "Critical Thinking",
    r"\bdecision\s+mak": "Decision Making",
    r"\bcustomer\s+(?:focus|orient|service|facing)": "Customer Focus",
    r"\badaptab|\bflexib|\bresilien": "Adaptability & Resilience",
    r"\binitiative\b|\bself\s*start|\bproactiv": "Initiative & Drive",
    r"\battention\s+to\s+detail\b|\bdetail\s*orient": "Attention to Detail",
    r"\binterpersonal\b|\bpeople\s+skills\b": "Interpersonal Skills",
    r"\borganiz|\bplanning\b|\bprioritiz": "Planning & Organization",
    r"\bcoach|\bmentor": "Coaching & Mentoring",
    r"\binnovati|\bcreativi": "Innovation & Creativity",
    r"\bnegotiat": "Negotiation",
    r"\binfluenc|\bpersuasi": "Influencing & Persuasion",
    r"\bstress\b|\bpressure\b|\bcalm\b": "Stress Tolerance",
    r"\bintegrity\b|\bethic|\bhonest": "Integrity & Ethics",
    r"\bmotivat": "Motivation",
    r"\bstrategic\s+think": "Strategic Thinking",
    r"\btime\s+management\b": "Time Management",
    r"\bconflict\s+(?:resolution|management)": "Conflict Resolution",
    r"\bsales\s+(?:orient|driv|abilit|candid|role|position|profess)": "Sales Orientation",
    r"\banalytical\b": "Analytical Thinking",
    r"\bwork\s+ethic\b|\bdependab|\breliab": "Work Ethic & Dependability",
    r"\bremote\s+work\b|\bwork\s+from\s+home\b": "Remote Work Readiness",
    r"\bemotional\s+intelligen": "Emotional Intelligence",
}

# Communication requirements
_COMMUNICATION_REQS = {
    r"\bverbal\s+communicat": "Verbal Communication",
    r"\bwritten\s+communicat|\bwriting\s+skills\b": "Written Communication",
    r"\bcommunicat\w+\s+skills\b": "Communication Skills",
    r"\bpresentation\s+skills\b|\bpublic\s+speak": "Presentation Skills",
    r"\bbilingual\b": "Bilingual",
    r"\bspanish\b": "Spanish Language",
    r"\bfrench\b": "French Language",
    r"\bgerman\b": "German Language",
    r"\bmandarin\b|\bchinese\b": "Chinese Language",
    r"\bjapanese\b": "Japanese Language",
    r"\bkorean\b": "Korean Language",
    r"\bportuguese\b": "Portuguese Language",
    r"\besl\b|\benglish\s+(?:as\s+a\s+)?second": "ESL",
    r"\breading\s+comprehen": "Reading Comprehension",
    r"\blistening\b": "Listening Skills",
    r"\bcustomer\s+communicat": "Customer Communication",
    r"\bemail\s+writ|\bbusiness\s+writ": "Business Writing",
    r"\btechnical\s+writ": "Technical Writing",
    r"\breport\s+writ": "Report Writing",
    r"\bproofread": "Proofreading",
}

# Job role patterns (regex -> normalized role name)
_JOB_ROLES = {
    r"\bsoftware\s+(?:developers?|engineers?)": "Software Developer",
    r"\bweb\s+developers?\b": "Web Developer",
    r"\bfrontend\s+(?:developers?|engineers?)": "Frontend Developer",
    r"\bbackend\s+(?:developers?|engineers?)": "Backend Developer",
    r"\bfull\s*stack\s+(?:developers?|engineers?)": "Full Stack Developer",
    r"\bdevops\s+engineers?\b": "DevOps Engineer",
    r"\bdata\s+scientists?\b": "Data Scientist",
    r"\bdata\s+analysts?\b": "Data Analyst",
    r"\bdata\s+engineers?\b": "Data Engineer",
    # Indirect: "assess Java developers" -> infer Software Developer
    r"\bassess\s+\w+\s+developers?\b": "Software Developer",
    r"\bml\s+engineer\b|\bmachine\s+learning\s+engineer": "ML Engineer",
    r"\bqa\b|\btest\s+engineer|\bquality\s+assurance": "QA Engineer",
    r"\bproject\s+manager\b": "Project Manager",
    r"\bproduct\s+manager\b": "Product Manager",
    r"\baccount\s+manager\b": "Account Manager",
    r"\bbranch\s+manager\b": "Branch Manager",
    r"\bsales\s+(?:rep|manager|executive|agent)": "Sales Professional",
    r"\bcustomer\s+service\b|\bcustomer\s+support\b": "Customer Service Rep",
    r"\bcall\s+center\b|\bcontact\s+center\b": "Call Center Agent",
    r"\badmin\w*\s+(?:assistant|professional|clerk)": "Administrative Assistant",
    r"\baccountant\b|\baccounting\s+clerk\b|\bauditor\b": "Accounting Professional",
    r"\bcashier\b": "Cashier",
    r"\bbank\s+(?:teller|clerk|officer)": "Banking Professional",
    r"\bnurse\b|\bnursing\b": "Nurse",
    r"\breceptionist\b": "Receptionist",
    r"\bsupervisor\b": "Supervisor",
    r"\bteam\s+lead(?:er)?\b": "Team Leader",
    r"\bagency\s+manager\b": "Agency Manager",
    r"\bcollection\s+agent\b|\bcollector\b": "Collections Agent",
    r"\breservation\s+agent\b": "Reservation Agent",
    r"\bwarehouse\b|\blogistic": "Warehouse / Logistics",
    r"\boperator\b|\btechnician\b": "Operator / Technician",
    r"\bhiring\s+(?:for\s+)?(?:a\s+)?(\w[\w\s]{2,30})": None,  # generic catch
}

# Duration extraction patterns
_DURATION_PATTERNS = [
    r"(?:under|max|maximum|less\s+than|within|no\s+more\s+than|up\s+to|shorter\s+than)\s+(\d+)\s*(?:min|minute)",
    r"(\d+)\s*(?:min|minute)\s+(?:or\s+less|max|limit)",
    r"(?:duration|time|length)\s*(?:of|:)?\s*(?:under|max|<|<=)?\s*(\d+)\s*(?:min|minute)?",
]

# Test type preference patterns
_TEST_TYPE_PREFS = {
    r"\baptitude\b|\bability\b|\bcognitive\b|\bnumerical\s+reason|\bverbal\s+reason|\binductive\b|\bdeductive\b": "A",
    r"\bsituational\s+judg|\bsjt\b|\bbiodata\b|\bjudgment\s+test": "B",
    r"\bcompetenc": "C",
    r"\bdevelopment\b.*\b(?:report|360)\b|\b360\b": "D",
    r"\bexercise\b|\bassessment\s+cent|\brole\s+play": "E",
    r"\bknowledge\b.*\btest|\btechnical\s+test|\bskills?\s+test|\bcoding\s+test": "K",
    r"\bpersonality\b|\bbehavio(?:u)?ral\b|\bopq\b|\bpsychometric": "P",
    r"\bsimulation\b|\bwork\s+sample": "S",
}


# ═══════════════════════════════════════════════════════════════
#  Extraction Engine
# ═══════════════════════════════════════════════════════════════

def _match_keywords(
    text: str,
    patterns: Dict[str, str],
    seen: Set[str],
) -> List[str]:
    """Match regex patterns against text, returning new canonical values."""
    results = []
    for pattern, canonical in patterns.items():
        if canonical and canonical not in seen:
            if re.search(pattern, text, re.IGNORECASE):
                results.append(canonical)
                seen.add(canonical)
    return results


def _extract_seniority(text: str) -> Optional[str]:
    """Extract the most specific seniority level from text."""
    text_lower = text.lower()
    # Check multi-word keys first (more specific), then single-word
    for keyword in sorted(_SENIORITY_MAP, key=len, reverse=True):
        if keyword in text_lower:
            return _SENIORITY_MAP[keyword]
    return None


def _extract_role(text: str) -> Optional[str]:
    """Extract a job role from text using pattern matching."""
    for pattern, role in _JOB_ROLES.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if role is not None:
                return role
            # Generic "hiring for X" catch — extract the captured group
            if match.lastindex and match.lastindex >= 1:
                return match.group(1).strip().title()
    return None


def _extract_duration(text: str) -> Optional[int]:
    """Extract max duration constraint from text."""
    for pattern in _DURATION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
    return None


def _extract_test_type(text: str) -> Optional[str]:
    """Extract preferred assessment type code from text."""
    for pattern, code in _TEST_TYPE_PREFS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return code
    return None


def _extract_language(text: str) -> Optional[str]:
    """Extract language preference for the assessment."""
    lang_map = {
        r"\bspanish\b": "Spanish",
        r"\bfrench\b": "French",
        r"\bgerman\b": "German",
        r"\bchinese\b|\bmandarin\b": "Chinese",
        r"\bjapanese\b": "Japanese",
        r"\bkorean\b": "Korean",
        r"\bportuguese\b": "Portuguese",
        r"\bdutch\b": "Dutch",
        r"\brussian\b": "Russian",
    }
    for pattern, lang in lang_map.items():
        if re.search(pattern, text, re.IGNORECASE):
            return lang
    return None


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

def reconstruct_state(
    messages: List[Dict[str, str]],
    context: Optional[Dict[str, Any]] = None,
) -> ConversationState:
    """
    Reconstruct conversation state from the full message history.

    Processes messages oldest-to-newest. Each user message can refine
    scalar fields (role, seniority, duration) or accumulate set fields
    (skills, personality, communication). This makes later messages
    act as refinement updates on the earlier context.

    Args:
        messages: List of {role, content} dicts (full history).
        context: Optional explicit context overrides from the request.

    Returns:
        A normalized ConversationState dataclass.
    """
    state = ConversationState()
    context = context or {}

    # Tracking sets to deduplicate accumulated fields
    seen_skills: Set[str] = set()
    seen_personality: Set[str] = set()
    seen_communication: Set[str] = set()

    # Collect all user message texts for query building
    user_texts: List[str] = []

    # ── Phase 1: Process messages oldest to newest ──────────
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role != "user" or not content.strip():
            continue

        user_texts.append(content)

        # Scalar fields: last mention wins (refinement)
        extracted_role = _extract_role(content)
        if extracted_role:
            state.role = extracted_role

        extracted_seniority = _extract_seniority(content)
        if extracted_seniority:
            state.seniority = extracted_seniority

        extracted_duration = _extract_duration(content)
        if extracted_duration:
            state.max_duration = extracted_duration

        extracted_type = _extract_test_type(content)
        if extracted_type:
            state.preferred_test_type = extracted_type

        extracted_lang = _extract_language(content)
        if extracted_lang:
            state.language = extracted_lang

        # Accumulating fields: all unique mentions preserved
        state.technical_skills.extend(
            _match_keywords(content, _TECHNICAL_SKILLS, seen_skills)
        )
        state.personality_requirements.extend(
            _match_keywords(content, _PERSONALITY_TRAITS, seen_personality)
        )
        state.communication_requirements.extend(
            _match_keywords(content, _COMMUNICATION_REQS, seen_communication)
        )

    # ── Phase 2: Apply explicit context overrides ───────────
    if context.get("job_level"):
        override = _SENIORITY_MAP.get(
            context["job_level"].lower().strip(),
            context["job_level"],
        )
        state.seniority = override

    if context.get("role"):
        state.role = context["role"]

    if context.get("max_duration"):
        try:
            state.max_duration = int(context["max_duration"])
        except (ValueError, TypeError):
            pass

    if context.get("test_type"):
        state.preferred_test_type = context["test_type"]

    if context.get("language"):
        state.language = context["language"]

    # ── Phase 3: Build composite search query ───────────────
    query_parts: List[str] = []
    if state.role:
        query_parts.append(state.role)
    if state.seniority:
        query_parts.append(state.seniority)
    query_parts.extend(state.technical_skills[:5])
    query_parts.extend(state.personality_requirements[:3])
    query_parts.extend(state.communication_requirements[:2])
    # Always include the latest user message
    if user_texts:
        query_parts.append(user_texts[-1])
    state.search_query = " ".join(query_parts)

    # ── Phase 4: Compute extraction confidence ──────────────
    filled = sum([
        state.role is not None,
        state.seniority is not None,
        len(state.technical_skills) > 0,
        len(state.personality_requirements) > 0,
        len(state.communication_requirements) > 0,
    ])
    state.extraction_confidence = round(filled / 5.0, 2)

    logger.info(
        "State reconstructed: role=%s, seniority=%s, skills=%d, "
        "personality=%d, communication=%d, confidence=%.0f%%",
        state.role,
        state.seniority,
        len(state.technical_skills),
        len(state.personality_requirements),
        len(state.communication_requirements),
        state.extraction_confidence * 100,
    )

    return state
