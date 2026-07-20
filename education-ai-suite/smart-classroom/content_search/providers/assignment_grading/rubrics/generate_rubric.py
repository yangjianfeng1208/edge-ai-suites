"""
Usage:
  python generate_rubric.py \
	  --input rubrics/example_1/input/input.json \
	  --output rubrics/example_1/generated_output/rubric.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate single rubric JSON from UI JSON")
	parser.add_argument("--input", required=True, help="Input JSON path")
	parser.add_argument("--output", required=True, help="Output JSON path (single file)")
	parser.add_argument(
		"--question-key",
		default="rubric",
		help="Key name used under each question in output (default: rubric)",
	)
	parser.add_argument(
		"--subjective-template",
		default=None,
		help="Optional JSON file with default rubric rules for subjective questions",
	)
	return parser.parse_args()


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


def sorted_question_items(raw: Any) -> list[tuple[str, dict[str, Any]]]:
	items: list[tuple[str, dict[str, Any]]] = []

	def extract_payload(obj: dict[str, Any]) -> dict[str, Any] | None:
		# Recommended keys first, keep "meta" as legacy fallback.
		for key in ["rubric", "grading", "config", "meta"]:
			val = obj.get(key)
			if isinstance(val, dict):
				return val
		return obj if isinstance(obj, dict) else None

	if isinstance(raw, dict):
		question_pool = raw.get("questions", raw)
		if not isinstance(question_pool, dict):
			question_pool = raw

		for k, v in question_pool.items():
			if isinstance(v, dict):
				payload = extract_payload(v)
				if isinstance(payload, dict):
					items.append((str(k), payload))
	elif isinstance(raw, list):
		for obj in raw:
			if not isinstance(obj, dict):
				continue
			idx = str(obj.get("index", "")).strip()
			payload = extract_payload(obj)
			if idx and isinstance(payload, dict):
				items.append((idx, payload))

	def key_fn(pair: tuple[str, dict[str, Any]]) -> tuple[int, str]:
		key = pair[0]
		return (int(key), key) if key.isdigit() else (10**9, key)

	return sorted(items, key=key_fn)


def normalize_answer(answer: Any) -> list[str]:
	if isinstance(answer, list):
		return [str(x).strip() for x in answer if str(x).strip()]
	if isinstance(answer, str) and answer.strip():
		return [answer.strip()]
	return []


def normalize_text_list(value: Any) -> list[str]:
	if isinstance(value, list):
		return [str(x).strip() for x in value if str(x).strip()]
	if isinstance(value, str) and value.strip():
		return [value.strip()]
	return []


def has_non_empty(value: Any) -> bool:
	if isinstance(value, str):
		return bool(value.strip())
	if isinstance(value, list):
		return any(str(x).strip() for x in value)
	if isinstance(value, dict):
		return len(value) > 0
	return value is not None


def pick_known_fields(meta: dict[str, Any]) -> dict[str, Any]:
	# Keep only fields that come from input/UI and are expected by rubric consumers.
	known_keys = [
		"index",
		"catalog",
		"type",
		"answer_mode",
		"score",
		"format",
		"match_mode",
		"answer",
		"scoring_criteria",
	]

	out: dict[str, Any] = {}
	for key in known_keys:
		if key not in meta:
			continue
		if key == "answer":
			out[key] = normalize_answer(meta.get(key))
		elif key == "scoring_criteria":
			out[key] = normalize_text_list(meta.get(key))
		else:
			out[key] = meta.get(key)
	return out


def build_default_scoring_standard(template: dict[str, Any]) -> dict[str, Any]:
	# If template already contains a direct scoring block, prefer it.
	if isinstance(template.get("scoring_standard"), dict):
		return copy.deepcopy(template["scoring_standard"])

	return {
		"process_required": bool(template.get("process_required", True)),
		"answer_only_score": template.get("answer_only_score", 0),
		"no_process_score": template.get("no_process_score", 0),
		"empty_response_score": template.get("empty_response_score", 0),
		"partial_credit_enabled": bool(template.get("partial_credit_enabled", True)),
		"scoring_principles": copy.deepcopy(template.get("scoring_principles", [])),
		"default_scoring_points": copy.deepcopy(template.get("default_scoring_points", [])),
		"deduction_guidelines": copy.deepcopy(template.get("deduction_guidelines", [])),
	}


def build_single_rubric(
	question_items: list[tuple[str, dict[str, Any]]],
	question_key: str,
	subjective_template: dict[str, Any] | None,
) -> dict[str, Any]:
	questions: dict[str, Any] = {}

	for qid, raw_meta in question_items:
		if not isinstance(raw_meta, dict):
			continue

		normalized = dict(raw_meta)
		normalized.setdefault("index", qid)
		normalized = pick_known_fields(normalized)

		# Skip totally empty rows.
		if not normalized:
			continue

		question_obj: dict[str, Any] = {question_key: normalized}

		is_subjective = str(normalized.get("catalog", "")).strip().lower() == "subjective"
		if is_subjective:
			custom_criteria = normalized.get("scoring_criteria")
			if has_non_empty(custom_criteria):
				question_obj["scoring_standard"] = {
					"source": "input",
					"scoring_criteria": copy.deepcopy(custom_criteria),
				}
			elif isinstance(subjective_template, dict):
				question_obj["scoring_standard"] = {
					"source": "template_default",
					**build_default_scoring_standard(subjective_template),
				}

		questions[qid] = question_obj

	return {"questions": questions}


def load_optional_template(path_value: str | None) -> dict[str, Any] | None:
	if not path_value:
		return None

	template_path = Path(path_value)
	if not template_path.exists():
		raise FileNotFoundError(f"subjective template not found: {template_path}")

	data = load_json(template_path)
	if not isinstance(data, dict):
		raise ValueError("subjective template must be a JSON object")
	return data


def validate_input_questions(question_items: list[tuple[str, dict[str, Any]]]) -> list[str]:
	errors: list[str] = []
	if not question_items:
		return ["questions is missing or empty"]

	score_presence: list[bool] = []

	for qid, payload in question_items:
		if not isinstance(payload, dict):
			errors.append(f"question {qid}: question config must be an object")
			continue

		catalog = payload.get("catalog")
		if not isinstance(catalog, str) or catalog not in {"objective", "subjective"}:
			errors.append(
				f"question {qid}: catalog must be 'objective' or 'subjective'"
			)

		answer = payload.get("answer")
		answer_ok = isinstance(answer, list) and any(str(x).strip() for x in answer)
		if not answer_ok:
			errors.append(f"question {qid}: answer must exist and be non-empty")

		has_score = "score" in payload
		score_presence.append(has_score)

	if score_presence:
		all_have = all(score_presence)
		none_have = not any(score_presence)
		if not (all_have or none_have):
			errors.append(
				"score consistency error: score must be present in all questions or none"
			)

	return errors


def main() -> int:
	args = parse_args()

	input_path = Path(args.input)
	output_path = Path(args.output)

	if not input_path.exists():
		print(f"ERROR: input file not found: {input_path}")
		return 1

	raw = load_json(input_path)
	question_items = sorted_question_items(raw)
	if not question_items:
		print("ERROR: no valid question entries found in input JSON")
		return 1

	input_validation_errors = validate_input_questions(question_items)
	if input_validation_errors:
		print("ERROR: input validation failed")
		for err in input_validation_errors:
			print(f"  - {err}")
		return 1

	try:
		subjective_template = load_optional_template(args.subjective_template)
	except Exception as exc:
		print(f"ERROR: {exc}")
		return 1

	rubric_data = build_single_rubric(question_items, args.question_key, subjective_template)

	final_output: dict[str, Any] = {}
	if isinstance(raw, dict) and isinstance(raw.get("metadata"), dict):
		final_output["metadata"] = raw["metadata"]

	final_output["questions"] = rubric_data["questions"]

	# Optional consistency check: if metadata.total_questions exists, keep a hint in output.
	meta = final_output.get("metadata")
	if isinstance(meta, dict) and isinstance(meta.get("total_questions"), int):
		expected = int(meta["total_questions"])
		actual = len(question_items)
		if expected != actual:
			final_output["generation_warning"] = (
				f"metadata.total_questions={expected} but parsed_questions={actual}"
			)

	write_json(output_path, final_output)

	print("=" * 80)
	print("Rubric generation complete")
	print("=" * 80)
	print(f"Input : {input_path}")
	print(f"Output: {output_path}")
	print(f"Total questions: {len(question_items)}")
	if isinstance(final_output.get("metadata"), dict):
		print("Metadata: included")
	if isinstance(subjective_template, dict):
		print("Subjective template: applied per subjective question")
	print(f"Question payload key: {args.question_key}")
	print("Input validation: passed")
	print("Generated files:")
	print(f"  - {output_path.name}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
