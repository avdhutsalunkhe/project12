"""
Test suite for the stateless conversation state reconstruction system.

Validates extraction of role, seniority, skills, personality,
communication, duration, and test type across various conversation patterns.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.conversation_state import reconstruct_state


def _print_state(label, state):
    """Pretty-print a conversation state."""
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    d = state.to_dict()
    for k, v in d.items():
        if isinstance(v, list) and v:
            print(f"  {k:30s}: {', '.join(v)}")
        elif v is not None and v != "" and v != 0.0:
            print(f"  {k:30s}: {v}")


def test_single_message():
    """Single message with multiple signals."""
    messages = [
        {"role": "user", "content": "I need to assess Java developers for a mid-senior role. The test should be under 30 minutes and include problem solving."}
    ]
    state = reconstruct_state(messages)
    _print_state("TEST 1: Single rich message", state)

    assert state.role == "Software Developer", f"Expected 'Software Developer', got '{state.role}'"
    assert state.seniority == "Mid-Professional", f"Expected 'Mid-Professional', got '{state.seniority}'"
    assert "Java" in state.technical_skills
    assert state.max_duration == 30
    assert "Problem Solving" in state.personality_requirements
    print("  >> PASSED")


def test_multi_turn_refinement():
    """Multi-turn conversation where later messages refine earlier ones."""
    messages = [
        {"role": "user", "content": "I'm hiring for a customer service position."},
        {"role": "assistant", "content": "What seniority level?"},
        {"role": "user", "content": "Entry level, and they need good communication skills and teamwork."},
        {"role": "assistant", "content": "Any duration preference?"},
        {"role": "user", "content": "Actually make it a supervisor role. Under 20 minutes please."},
    ]
    state = reconstruct_state(messages)
    _print_state("TEST 2: Multi-turn refinement", state)

    # Last mention of seniority should win
    assert state.seniority == "Supervisor", f"Expected 'Supervisor', got '{state.seniority}'"
    assert state.max_duration == 20
    assert "Communication Skills" in state.communication_requirements
    assert "Teamwork & Collaboration" in state.personality_requirements
    print("  >> PASSED")


def test_technical_skills_accumulation():
    """Skills should accumulate across messages, not reset."""
    messages = [
        {"role": "user", "content": "We need someone who knows Python and Docker."},
        {"role": "assistant", "content": "Got it. Any other requirements?"},
        {"role": "user", "content": "Yes, also AWS and Kubernetes experience. Full stack developer role."},
    ]
    state = reconstruct_state(messages)
    _print_state("TEST 3: Skill accumulation", state)

    assert "Python" in state.technical_skills
    assert "Docker" in state.technical_skills
    assert "AWS" in state.technical_skills
    assert "Kubernetes" in state.technical_skills
    assert state.role == "Full Stack Developer"
    print("  >> PASSED")


def test_context_overrides():
    """Explicit context dict should override message extractions."""
    messages = [
        {"role": "user", "content": "I need a junior developer assessment."},
    ]
    context = {"job_level": "executive", "max_duration": "15"}
    state = reconstruct_state(messages, context=context)
    _print_state("TEST 4: Context overrides", state)

    assert state.seniority == "Executive"
    assert state.max_duration == 15
    print("  >> PASSED")


def test_personality_extraction():
    """Extract personality and behavioral requirements."""
    messages = [
        {"role": "user", "content": "We need someone with strong leadership, decision making ability, and customer focus."},
        {"role": "user", "content": "Also important: attention to detail, stress tolerance, and integrity."},
    ]
    state = reconstruct_state(messages)
    _print_state("TEST 5: Personality extraction", state)

    expected = {"Leadership", "Decision Making", "Customer Focus",
                "Attention to Detail", "Stress Tolerance", "Integrity & Ethics"}
    found = set(state.personality_requirements)
    assert expected.issubset(found), f"Missing: {expected - found}"
    print("  >> PASSED")


def test_communication_extraction():
    """Extract communication and language requirements."""
    messages = [
        {"role": "user", "content": "The role requires bilingual candidates with strong written communication and presentation skills."},
        {"role": "user", "content": "They also need proofreading ability. Assessment in Spanish please."},
    ]
    state = reconstruct_state(messages)
    _print_state("TEST 6: Communication extraction", state)

    assert "Bilingual" in state.communication_requirements
    assert "Written Communication" in state.communication_requirements
    assert "Presentation Skills" in state.communication_requirements
    assert "Proofreading" in state.communication_requirements
    assert state.language == "Spanish"
    print("  >> PASSED")


def test_assessment_type_preference():
    """Extract assessment type preferences."""
    messages = [
        {"role": "user", "content": "I want a personality test for sales candidates."},
    ]
    state = reconstruct_state(messages)
    _print_state("TEST 7: Assessment type preference", state)

    assert state.preferred_test_type == "P"
    assert "Sales Orientation" in state.personality_requirements
    print("  >> PASSED")


def test_empty_conversation():
    """Handle empty or system-only messages gracefully."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    state = reconstruct_state(messages)
    _print_state("TEST 8: Empty conversation", state)

    assert state.role is None
    assert state.seniority is None
    assert state.technical_skills == []
    assert state.extraction_confidence == 0.0
    print("  >> PASSED")


def test_complex_realistic():
    """Realistic multi-turn conversation with mixed signals."""
    messages = [
        {"role": "user", "content": "Hi, I'm looking for assessments for our IT department."},
        {"role": "assistant", "content": "I can help! What roles are you hiring for?"},
        {"role": "user", "content": "We need to test data analysts. They should know SQL, Python, and Tableau."},
        {"role": "assistant", "content": "What level of experience?"},
        {"role": "user", "content": "Mid-level. They need analytical thinking and attention to detail."},
        {"role": "assistant", "content": "Any constraints on the assessment?"},
        {"role": "user", "content": "Knowledge test, under 20 minutes. Also check their written communication."},
    ]
    state = reconstruct_state(messages)
    _print_state("TEST 9: Complex realistic scenario", state)

    assert state.role == "Data Analyst"
    assert state.seniority == "Mid-Professional"
    assert "SQL" in state.technical_skills
    assert "Python" in state.technical_skills
    assert "Tableau" in state.technical_skills
    assert "Analytical Thinking" in state.personality_requirements
    assert "Attention to Detail" in state.personality_requirements
    assert "Written Communication" in state.communication_requirements
    assert state.preferred_test_type == "K"
    assert state.max_duration == 20
    assert state.extraction_confidence >= 0.8
    print("  >> PASSED")


def main():
    tests = [
        test_single_message,
        test_multi_turn_refinement,
        test_technical_skills_accumulation,
        test_context_overrides,
        test_personality_extraction,
        test_communication_extraction,
        test_assessment_type_preference,
        test_empty_conversation,
        test_complex_realistic,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  >> FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  >> ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 65}")
    print(f"  RESULTS: {passed}/{passed + failed} passed")
    print(f"{'=' * 65}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
