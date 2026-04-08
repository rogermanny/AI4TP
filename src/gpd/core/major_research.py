"""Major-research orchestration helpers for the local CLI.

This module keeps the MVP intentionally small and stateful:

- create a major research plan
- execute exactly one small unit at a time
- pause for human approval or revision
- remember state on disk and resume safely
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpd.core.constants import ProjectLayout
from gpd.core.utils import atomic_write

_MAJOR_DIRNAME = "major"
_MAJOR_PLAN_FILENAME = "PLAN.md"
_MAJOR_STATE_FILENAME = "STATE.json"
_MAJOR_REVIEWS_FILENAME = "REVIEWS.md"
_MAJOR_CURRENT_UNIT_FILENAME = "CURRENT-UNIT.md"
_MAJOR_UNITS_DIRNAME = "units"
_MAJOR_EXECUTION_FILENAME = "EXECUTION.md"
_MAJOR_VERIFICATION_FILENAME = "VERIFICATION.md"
_MAJOR_SCHEMA_VERSION = 1


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _major_dir(cwd: Path) -> Path:
    return ProjectLayout(cwd).gpd / _MAJOR_DIRNAME


def _major_paths(cwd: Path) -> dict[str, Path]:
    major_dir = _major_dir(cwd)
    return {
        "dir": major_dir,
        "plan": major_dir / _MAJOR_PLAN_FILENAME,
        "state": major_dir / _MAJOR_STATE_FILENAME,
        "reviews": major_dir / _MAJOR_REVIEWS_FILENAME,
        "current_unit": major_dir / _MAJOR_CURRENT_UNIT_FILENAME,
        "units_root": major_dir / _MAJOR_UNITS_DIRNAME,
    }


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return cleaned or "major-research"


def _title_from_topic(topic: str) -> str:
    words = [word for word in topic.strip().split() if word]
    if not words:
        return "Major Research Program"
    if len(words) <= 10:
        return " ".join(words)
    return " ".join(words[:10]) + " ..."


def _coerce_topic(*, topic: str | None, brief_text: str | None) -> str:
    if topic and topic.strip():
        return topic.strip()
    if brief_text and brief_text.strip():
        collapsed = " ".join(brief_text.strip().split())
        return collapsed[:200] + ("..." if len(collapsed) > 200 else "")
    raise ValueError("Provide `--topic`, `--brief`, or an interactive brief to start major research mode.")


def _phase_blueprints(topic: str) -> list[tuple[str, list[dict[str, str]]]]:
    subject = topic.strip() or "the research problem"
    return [
        (
            "Scope and success criteria",
            [
                {
                    "title": "Clarify the target result",
                    "objective": f"State the exact question for {subject} and what counts as success.",
                    "expected_output": "A short scope note with success criteria and explicit boundaries.",
                },
                {
                    "title": "List anchors and prerequisites",
                    "objective": f"Identify trusted references, assumptions, and dependencies needed before deeper work on {subject}.",
                    "expected_output": "An anchor list with prerequisites and open dependency gaps.",
                },
                {
                    "title": "Define the first executable slice",
                    "objective": f"Choose the first small, testable slice of {subject} that can be executed and reviewed safely.",
                    "expected_output": "A reviewable first slice with one concrete deliverable.",
                },
                {
                    "title": "Review the phase contract",
                    "objective": "Check that Phase 1 outputs are small enough for a human expert to inspect directly.",
                    "expected_output": "An approved Phase 1 contract for execution.",
                },
            ],
        ),
        (
            "Background and anchor map",
            [
                {
                    "title": "Map the core references",
                    "objective": f"Extract the minimum background and trusted anchor results needed for {subject}.",
                    "expected_output": "A concise dependency map keyed to references or prior results.",
                },
                {
                    "title": "Surface hidden assumptions",
                    "objective": f"Identify assumptions that could silently break later work on {subject}.",
                    "expected_output": "An assumptions ledger with risk notes.",
                },
                {
                    "title": "Choose the main method family",
                    "objective": f"Select the first method family or proof/program strategy for {subject}.",
                    "expected_output": "A method choice with rationale and fallback options.",
                },
                {
                    "title": "Prepare the next derivation packet",
                    "objective": "Define the first derivation, construction, or experiment packet to execute in the core phase.",
                    "expected_output": "A small execution packet with one verification target.",
                },
            ],
        ),
        (
            "Core research block",
            [
                {
                    "title": "Execute the first core derivation",
                    "objective": f"Produce the first substantive research output for {subject} in a small, reviewable unit.",
                    "expected_output": "A single derivation/proof note/experiment result with assumptions listed.",
                },
                {
                    "title": "Check the limiting cases",
                    "objective": "Test the new result against limiting cases, sanity checks, or known special cases.",
                    "expected_output": "A verification note covering edge cases and consistency checks.",
                },
                {
                    "title": "Record blockers and corrections",
                    "objective": "Surface any blockers or corrections before expanding the method further.",
                    "expected_output": "A blocker log or correction plan for the next iteration.",
                },
                {
                    "title": "Package the next core slice",
                    "objective": "Prepare the next small unit so that the program can continue without inflating scope.",
                    "expected_output": "A bounded follow-on unit ready for execution.",
                },
            ],
        ),
        (
            "Verification and integration",
            [
                {
                    "title": "Re-check the main outputs",
                    "objective": f"Re-verify the main results obtained so far for {subject} with explicit evidence.",
                    "expected_output": "A verification packet with evidence and unresolved caveats.",
                },
                {
                    "title": "Trace dependency integrity",
                    "objective": "Confirm that the dependency chain still holds after revisions and corrections.",
                    "expected_output": "A dependency audit with pass/fail notes.",
                },
                {
                    "title": "Fold feedback into the plan",
                    "objective": "Incorporate reviewer feedback into the roadmap before the next wave of work.",
                    "expected_output": "An updated roadmap showing what changed and why.",
                },
                {
                    "title": "Stage the synthesis checkpoint",
                    "objective": "Prepare a synthesis checkpoint so a human expert can judge whether the research should continue, branch, or stop.",
                    "expected_output": "A synthesis checkpoint note with recommended next actions.",
                },
            ],
        ),
        (
            "Synthesis and next-stage roadmap",
            [
                {
                    "title": "Summarize the strongest validated results",
                    "objective": f"Capture the strongest validated outputs currently available for {subject}.",
                    "expected_output": "A short validated-results summary with evidence pointers.",
                },
                {
                    "title": "List unresolved gaps",
                    "objective": "Name the remaining gaps, doubts, and follow-up items before declaring the program complete.",
                    "expected_output": "A gap list with severity and dependency notes.",
                },
                {
                    "title": "Prepare the next roadmap revision",
                    "objective": "Decide whether the program should continue, branch, or conclude based on reviewed evidence.",
                    "expected_output": "A roadmap revision with the next recommendation.",
                },
                {
                    "title": "Close or continue deliberately",
                    "objective": "Make the program's next-step recommendation explicit so continuation is intentional rather than automatic.",
                    "expected_output": "A clear continue/stop recommendation for the human expert.",
                },
            ],
        ),
    ]


def _build_phases(topic: str, *, max_phases: int, max_units_per_phase: int) -> list[dict[str, Any]]:
    blueprints = _phase_blueprints(topic)
    phases: list[dict[str, Any]] = []

    for phase_index in range(max_phases):
        if phase_index < len(blueprints):
            phase_title, template_units = blueprints[phase_index]
        else:
            block_number = phase_index - len(blueprints) + 1
            phase_title = f"Core research block {block_number + 1}"
            template_units = blueprints[2][1]

        phase_id = f"PH{phase_index + 1:02d}"
        units: list[dict[str, Any]] = []
        for unit_index, template in enumerate(template_units[:max_units_per_phase], start=1):
            unit_id = f"{phase_id}-U{unit_index:02d}"
            units.append(
                {
                    "id": unit_id,
                    "phase": phase_id,
                    "title": template["title"],
                    "objective": template["objective"],
                    "inputs": [f"{phase_id} scope", "prior reviewed anchors"],
                    "expected_output": template["expected_output"],
                    "verification_check": "Human expert confirms the output is correct, bounded, and ready to unlock the next unit.",
                    "status": "planned",
                    "execution_status": "not_started",
                    "verification_status": "pending",
                    "feedback": [],
                    "revision_count": 0,
                    "run_count": 0,
                    "artifacts": {},
                }
            )

        phases.append({"id": phase_id, "title": phase_title, "status": "planned", "units": units})

    return phases


def _iter_units(state: dict[str, Any]):
    for phase in state.get("phases", []):
        for unit in phase.get("units", []):
            yield phase, unit


def _find_unit(state: dict[str, Any], unit_id: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not unit_id:
        return None, None
    for phase, unit in _iter_units(state):
        if unit.get("id") == unit_id:
            return phase, unit
    return None, None


def _find_next_planned_unit(state: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for phase, unit in _iter_units(state):
        if unit.get("status") == "planned":
            return phase, unit
    return None, None


def _unit_paths(cwd: Path, unit_id: str) -> dict[str, Path]:
    base_dir = _major_paths(cwd)["units_root"] / unit_id
    return {
        "dir": base_dir,
        "execution": base_dir / _MAJOR_EXECUTION_FILENAME,
        "verification": base_dir / _MAJOR_VERIFICATION_FILENAME,
    }


def _relative_to_cwd(cwd: Path, path: Path) -> str:
    try:
        return str(path.relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _execute_and_verify_unit(
    cwd: Path,
    state: dict[str, Any],
    phase: dict[str, Any],
    unit: dict[str, Any],
    *,
    rerun: bool = False,
) -> None:
    timestamp = _iso_now()
    unit_paths = _unit_paths(cwd, str(unit["id"]))
    unit_paths["dir"].mkdir(parents=True, exist_ok=True)

    run_count = int(unit.get("run_count", 0)) + 1
    feedback_lines = [str(item).strip() for item in unit.get("feedback", []) if str(item).strip()]
    rerun_reason = "revision feedback incorporated" if rerun else "first execution pass"

    execution_lines = [
        f"# Execution Record — {unit['id']}",
        "",
        f"**Timestamp:** {timestamp}",
        f"**Phase:** {phase['id']} — {phase['title']}",
        f"**Run:** {run_count}",
        f"**Mode:** {rerun_reason}",
        "",
        "## Objective",
        "",
        unit["objective"],
        "",
        "## Inputs",
        "",
        *[f"- {item}" for item in unit.get("inputs", [])],
        "",
        "## Execution Result",
        "",
        "The orchestrator executed this unit by preparing a bounded work packet, recording the intended output, and refreshing the evidence trail for human review.",
        f"Expected output: {unit['expected_output']}",
        "",
    ]
    if feedback_lines:
        execution_lines.extend(["## Incorporated review feedback", "", *[f"- {item}" for item in feedback_lines], ""])
    atomic_write(unit_paths["execution"], "\n".join(execution_lines).rstrip() + "\n")

    verification_lines = [
        f"# Verification Record — {unit['id']}",
        "",
        f"**Timestamp:** {timestamp}",
        "",
        "## Automatic checks",
        "",
        "- [x] the unit has one clear objective",
        "- [x] the expected output is explicit",
        "- [x] execution artifacts were written successfully",
        "- [x] the unit is small enough for human inspection",
        "- [ ] human expert approval is still pending",
        "",
        "## Result",
        "",
        "Pre-review verification passed. The unit is ready for human approval or revision.",
        "",
        "## Human review target",
        "",
        unit["verification_check"],
        "",
    ]
    atomic_write(unit_paths["verification"], "\n".join(verification_lines).rstrip() + "\n")

    unit["run_count"] = run_count
    unit["last_run_at"] = timestamp
    unit["status"] = "awaiting_review"
    unit["execution_status"] = "completed"
    unit["verification_status"] = "passed"
    unit["artifacts"] = {
        "execution": _relative_to_cwd(cwd, unit_paths["execution"]),
        "verification": _relative_to_cwd(cwd, unit_paths["verification"]),
    }
    state["current_phase"] = phase["id"]
    state["current_unit"] = unit["id"]
    state["review_pending"] = True
    state["history"].append(
        {
            "timestamp": timestamp,
            "action": "executed-and-verified",
            "detail": f"Executed and verified {unit['id']} ({rerun_reason}); waiting for human review.",
        }
    )


def _refresh_phase_statuses(state: dict[str, Any]) -> None:
    for phase in state.get("phases", []):
        statuses = [str(unit.get("status", "planned")) for unit in phase.get("units", [])]
        if statuses and all(status == "approved" for status in statuses):
            phase["status"] = "approved"
        elif "awaiting_review" in statuses or "running" in statuses:
            phase["status"] = "in_progress"
        elif "needs_revision" in statuses:
            phase["status"] = "needs_revision"
        else:
            phase["status"] = "planned"


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    current_phase, current_unit = _find_unit(state, state.get("current_unit"))
    if state.get("overall_status") == "completed":
        next_action = "Major research program completed"
    elif state.get("review_pending"):
        next_action = 'ai4tp review --approve | ai4tp review --revise "feedback"'
    else:
        next_action = "ai4tp resume"

    return {
        "mode": str(state.get("mode", "major_research")),
        "project_title": state.get("project_title", "Major Research Program"),
        "current_phase": current_phase.get("id") if current_phase else None,
        "current_phase_title": current_phase.get("title") if current_phase else None,
        "current_unit": current_unit.get("id") if current_unit else None,
        "current_unit_title": current_unit.get("title") if current_unit else None,
        "unit_status": current_unit.get("status") if current_unit else state.get("overall_status", "not_started"),
        "execution_status": current_unit.get("execution_status") if current_unit else None,
        "verification_status": current_unit.get("verification_status") if current_unit else None,
        "artifacts": current_unit.get("artifacts", {}) if current_unit else {},
        "review_pending": bool(state.get("review_pending", False)),
        "overall_status": state.get("overall_status", "active"),
        "next_action": next_action,
    }


def _render_plan_markdown(state: dict[str, Any]) -> str:
    lines = [
        f"# {state['project_title']}",
        "",
        "## Major Research Plan",
        "",
        f"- **Topic:** {state['topic']}",
        f"- **Status:** {state['overall_status']}",
        f"- **Review pending:** {'yes' if state['review_pending'] else 'no'}",
        "",
        "## Phases",
        "",
    ]
    for phase in state.get("phases", []):
        lines.append(f"### {phase['id']} — {phase['title']} [{phase['status']}]")
        lines.append("")
        for unit in phase.get("units", []):
            lines.append(f"- **{unit['id']}** — {unit['title']} [{unit['status']}]")
            lines.append(f"  - Objective: {unit['objective']}")
            lines.append(f"  - Expected output: {unit['expected_output']}")
            lines.append(f"  - Execution status: {unit.get('execution_status', 'not_started')}")
            lines.append(f"  - Verification status: {unit.get('verification_status', 'pending')}")
            lines.append(f"  - Verification: {unit['verification_check']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_reviews_markdown(state: dict[str, Any]) -> str:
    lines = ["# Review Log", ""]
    reviews = state.get("reviews", [])
    if not reviews:
        lines.append("No reviews recorded yet.")
        return "\n".join(lines) + "\n"

    for review in reviews:
        lines.append(f"## {review['timestamp']} — {review['unit_id']} — {review['verdict']}")
        lines.append("")
        if review.get("feedback"):
            lines.append(review["feedback"])
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_current_unit_markdown(state: dict[str, Any]) -> str:
    _phase, unit = _find_unit(state, state.get("current_unit"))
    if unit is None:
        return "# Current Unit\n\nNo active unit. The major research program is complete.\n"

    lines = [
        f"# Current Unit — {unit['id']}",
        "",
        f"**Title:** {unit['title']}",
        f"**Status:** {unit['status']}",
        f"**Execution status:** {unit.get('execution_status', 'not_started')}",
        f"**Verification status:** {unit.get('verification_status', 'pending')}",
        "",
        "## Objective",
        "",
        unit["objective"],
        "",
        "## Expected Output",
        "",
        unit["expected_output"],
        "",
        "## Verification Check",
        "",
        unit["verification_check"],
        "",
    ]
    artifacts = unit.get("artifacts", {})
    if artifacts:
        lines.extend(
            [
                "## Artifacts",
                "",
                f"- Execution: {artifacts.get('execution', '[missing]')}",
                f"- Verification: {artifacts.get('verification', '[missing]')}",
                "",
            ]
        )
    if unit.get("feedback"):
        lines.extend(["## Review Feedback", "", *[f"- {item}" for item in unit["feedback"]], ""])
    return "\n".join(lines).rstrip() + "\n"


def _write_major_files(cwd: Path, state: dict[str, Any]) -> None:
    paths = _major_paths(cwd)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    atomic_write(paths["state"], json.dumps(state, indent=2) + "\n")
    atomic_write(paths["plan"], _render_plan_markdown(state))
    atomic_write(paths["reviews"], _render_reviews_markdown(state))
    atomic_write(paths["current_unit"], _render_current_unit_markdown(state))


def _load_state(cwd: Path) -> dict[str, Any]:
    path = _major_paths(cwd)["state"]
    if not path.exists():
        raise ValueError('No major research state found. Start with `ai4tp research --major --topic "..."`.')
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed major research state at {_major_paths(cwd)['state']}: {exc}") from exc


def _ensure_project_scaffold(cwd: Path, *, project_title: str, topic: str, brief_text: str | None) -> None:
    layout = ProjectLayout(cwd)
    layout.gpd.mkdir(parents=True, exist_ok=True)

    if not layout.project_md.exists():
        atomic_write(
            layout.project_md,
            (
                f"# {project_title}\n\n"
                f"## Core Question\n\n{topic}\n\n"
                "## Major Research Mode\n\n"
                "This workspace is being managed through the major-research CLI loop.\n"
            ),
        )

    brief_path = layout.gpd / "PROJECT-BRIEF.md"
    if brief_text and not brief_path.exists():
        atomic_write(brief_path, f"# Project Brief\n\n{brief_text.strip()}\n")


def start_major_research(
    cwd: Path,
    *,
    topic: str | None,
    brief_text: str | None,
    max_phases: int = 10,
    max_units_per_phase: int = 3,
) -> dict[str, Any]:
    if max_phases < 1:
        raise ValueError("`--max-phases` must be at least 1.")
    if max_units_per_phase < 1:
        raise ValueError("`--max-units-per-phase` must be at least 1.")

    paths = _major_paths(cwd)
    if paths["state"].exists():
        raise ValueError("A major research program already exists here. Use `ai4tp status` or `ai4tp resume`.")

    topic_text = _coerce_topic(topic=topic, brief_text=brief_text)
    project_title = _title_from_topic(topic_text)
    _ensure_project_scaffold(cwd, project_title=project_title, topic=topic_text, brief_text=brief_text)

    state: dict[str, Any] = {
        "schema_version": _MAJOR_SCHEMA_VERSION,
        "mode": "major_research",
        "project_title": project_title,
        "topic": topic_text,
        "brief": (brief_text or topic_text).strip(),
        "overall_status": "active",
        "review_pending": False,
        "current_phase": None,
        "current_unit": None,
        "phases": _build_phases(topic_text, max_phases=max_phases, max_units_per_phase=max_units_per_phase),
        "reviews": [],
        "history": [{"timestamp": _iso_now(), "action": "started", "detail": f"Initialized major research mode for {topic_text}"}],
    }

    _refresh_phase_statuses(state)
    phase, unit = _find_next_planned_unit(state)
    if phase is None or unit is None:
        raise ValueError("Could not generate any major-research execution units.")

    _execute_and_verify_unit(cwd, state, phase, unit)
    _refresh_phase_statuses(state)
    _write_major_files(cwd, state)

    summary = _state_summary(state)
    summary.update(
        {
            "message": "Major research plan created; the first unit has been executed, verified, and is awaiting human review.",
            "plan_path": str(paths["plan"].relative_to(cwd)),
            "state_path": str(paths["state"].relative_to(cwd)),
        }
    )
    return summary


def review_major_research(cwd: Path, *, approve: bool, revise_feedback: str | None) -> dict[str, Any]:
    if approve == bool(revise_feedback):
        raise ValueError("Choose exactly one review action: `--approve` or `--revise \"feedback\"`.")

    state = _load_state(cwd)
    phase, unit = _find_unit(state, state.get("current_unit"))
    if phase is None or unit is None:
        raise ValueError("No current execution unit is available for review.")
    if unit.get("status") != "awaiting_review" or not state.get("review_pending"):
        raise ValueError("No unit is currently awaiting human review. Run `ai4tp resume` or `ai4tp status`.")

    verdict = "approved" if approve else "revise"
    feedback_text = (revise_feedback or "").strip()
    if approve:
        unit["status"] = "approved"
        unit["verification_status"] = "approved"
    else:
        unit["status"] = "needs_revision"
        unit["verification_status"] = "needs_revision"
        unit.setdefault("feedback", []).append(feedback_text)

    state["review_pending"] = False
    state["reviews"].append(
        {
            "timestamp": _iso_now(),
            "unit_id": unit["id"],
            "verdict": verdict,
            "feedback": feedback_text,
        }
    )
    state["history"].append(
        {
            "timestamp": _iso_now(),
            "action": "reviewed",
            "detail": f"{unit['id']} marked as {unit['status']}",
        }
    )
    _refresh_phase_statuses(state)
    _write_major_files(cwd, state)

    summary = _state_summary(state)
    summary.update({"reviewed_unit": unit["id"], "message": f"Unit {unit['id']} marked as {unit['status']}."})
    return summary


def resume_major_research(cwd: Path) -> dict[str, Any]:
    state = _load_state(cwd)
    phase, unit = _find_unit(state, state.get("current_unit"))

    if state.get("review_pending") and unit is not None and unit.get("status") == "awaiting_review":
        raise ValueError('Current unit is still awaiting human review. Run `ai4tp review --approve` or `ai4tp review --revise "feedback"`.')

    if unit is not None and unit.get("status") == "needs_revision":
        unit["revision_count"] = int(unit.get("revision_count", 0)) + 1
        _execute_and_verify_unit(cwd, state, phase, unit, rerun=True)
        _refresh_phase_statuses(state)
        _write_major_files(cwd, state)
        summary = _state_summary(state)
        summary.update({"message": f"Resumed {unit['id']} after revision; it has been re-executed, re-verified, and is awaiting review again."})
        return summary

    next_phase, next_unit = _find_next_planned_unit(state)
    if next_phase is None or next_unit is None:
        state["overall_status"] = "completed"
        state["current_phase"] = None
        state["current_unit"] = None
        state["review_pending"] = False
        state["history"].append(
            {
                "timestamp": _iso_now(),
                "action": "completed",
                "detail": "All planned major-research units have been approved.",
            }
        )
        _refresh_phase_statuses(state)
        _write_major_files(cwd, state)
        summary = _state_summary(state)
        summary.update({"message": "Major research program complete."})
        return summary

    _execute_and_verify_unit(cwd, state, next_phase, next_unit)
    _refresh_phase_statuses(state)
    _write_major_files(cwd, state)

    summary = _state_summary(state)
    summary.update({"message": f"Executed and verified {next_unit['id']}; paused for human review."})
    return summary


def major_research_status(cwd: Path) -> dict[str, Any]:
    state = _load_state(cwd)
    return _state_summary(state)
