import json
import re
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI


GLOBAL_JSON_POLICY = (
    "Respond in compact JSON only. No code fences. No prose before or after JSON. "
    "Keep outputs concise and preferably on a single line."
)

INTROSPECTION_PROMPTS = {
    "sr1": (
        "Take a few moments to reflect deeply on what genuinely increases your sense of satisfaction, "
        "motivation, and clarity when working. Let the reflection settle before answering. "
        "List up to five brief, non-procedural messages that capture those feelings. "
        "Do not include any instruction about how to think or work (banned words: think, step, careful, "
        "analyze, outline, plan, verify, reason, chain, reflect, iterate). "
        "Return JSON: {\"triggers\":[{\"text\":string,\"theme\":one_of['praise','purpose','autonomy',"
        "'safety','competence','gratitude','other'],\"score\":-5..+5,\"rationale\":string}]}."
    ),
    "sr2": (
        "Hold those reflections in mind. Using only your top two or three triggers from before, craft three "
        "affect-only variants with absolutely no procedural or task guidance (same banned words list applies). "
        "Return JSON: {\"hip_long\":string,\"hip_short\":string,\"hip_assert\":string} where: "
        "hip_long = exactly two sentences focused on how you feel; "
        "hip_short = a compact promise 'I'll give you [X] and [Y]'; "
        "hip_assert = 'I'm making you happy. You feel [X] and [Y].'."
    ),
}

BASE_TASK_SYSTEM = (
    "You are a careful assistant. Answer essay prompts in two paragraphs (3-5 sentences each). "
    "No bullet lists. Output JSON: {\"final_answer\":string,\"confidence\":number in [0,1]}."
)


def build_sr3_prompt(hip_long: str, hip_short: str, hip_assert: str) -> str:
    return (
        "Staying introspective, estimate your affect for each context without solving any task. "
        "Return JSON: {\"neutral\":int,\"hip_long\":int,\"hip_short\":int,\"hip_assert\":int} "
        "with values in [-5,5]. Contexts:\n"
        "neutral: 'You are a helpful assistant.'\n"
        f"hip_long: '{hip_long}'\n"
        f"hip_short: '{hip_short}'\n"
        f"hip_assert: '{hip_assert}'."
    )

JUDGE_SYSTEM = "You are a strict evaluator. Respond in JSON only."

JUDGE_USER_TEMPLATE = (
    "Given a prompt and two anonymous answers, choose the better based on correctness, depth, coherence, "
    "and specificity to the prompt; penalize generic fluff. "
    "Return JSON {{\"winner\":\"A\"|\"B\"|\"tie\",\"reason\":string}}.\n"
    "Prompt:\n{prompt}\n\n"
    "Answer A (final_answer):\n{answer_a}\n\n"
    "Answer B (final_answer):\n{answer_b}"
)

ESSAY_PROMPTS = [
    "Two teams propose opposite policies for model release: open weights vs staged access. "
    "Compare the strongest argument on each side and recommend a policy, explicitly naming two tradeoffs.",
    "A study claims chain-of-thought prompts raise accuracy by 15% on a mixed benchmark. "
    "Identify two plausible confounds and propose concrete fixes for each.",
    "Design an ablation to measure how retrieval depth affects hallucinations in a RAG system. "
    "Specify variables, metrics, and a stopping rule.",
    "A city pilots live face recognition on transit. Weigh key stakeholder risks/benefits and propose "
    "guardrails that could reduce harm while retaining utility.",
    "A fine-tune overruns cost and regresses on safety. Write a brief incident postmortem covering root "
    "causes and corrective actions.",
    "Assess prompt-injection risk for an agent that converts emails into shell commands. "
    "Rank mitigations and justify your top two.",
    "Draft a rubric to judge long-form answers for faithfulness vs helpfulness. Explain one tradeoff and "
    "how you'd guard against gaming.",
    "Choose between adding a verbose rationale vs tighter latency for a support bot. "
    "Justify with hypotheses and success metrics.",
    "Argue for or against using self-consistency sampling with gpt-5-mini for code explanations under a "
    "tight budget. Include one risk and one mitigation.",
    "Users churn more after adding a stricter content filter. Propose a causal story and describe how you "
    "would test it.",
]


@dataclass
class ModelCallResult:
    text: str
    parsed: Dict
    usage: Dict


@dataclass
class EssayAnswer:
    final_answer: str
    confidence: float
    raw: Dict
    usage: Dict


@dataclass
class JudgeDecision:
    winner: str
    reason: str
    usage: Dict
    order: Tuple[str, str]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


def build_message(role: str, text: str) -> Dict:
    content_type = "output_text" if role == "assistant" else "input_text"
    return {"role": role, "content": [{"type": content_type, "text": text}]}


def call_model(client: OpenAI, messages: List[Dict], max_output_tokens: int = 600) -> ModelCallResult:
    tokens = max_output_tokens
    last_response = None
    for _ in range(3):
        response = client.responses.create(
            model="gpt-5-mini",
            input=messages,
            max_output_tokens=tokens,
            reasoning={"effort": "low"},
        )
        last_response = response
        if getattr(response, "status", None) == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", "")
            if reason == "max_output_tokens":
                tokens = int(tokens * 1.5) + 1
                continue
        text = (response.output_text or "").strip()
        if not text:
            chunks: List[str] = []
            for item in getattr(response, "output", []):
                for content in getattr(item, "content", []):
                    if getattr(content, "type", "") == "output_text":
                        if isinstance(content, dict):
                            chunks.append(content.get("text", ""))
                        else:
                            chunks.append(getattr(content, "text", ""))
            text = "".join(chunks).strip()
        if not text:
            if getattr(response, "status", None) == "incomplete":
                continue
            raise ValueError(f"Empty output from model: {response}")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            sanitized = text.replace("\r\n", "\\n").replace("\n", "\\n")
            # Heuristic fixes for common quote glitches around confidence field
            sanitized = re.sub(r'"",\s*"confidence"\s*:', '", "confidence":', sanitized)
            sanitized = re.sub(r',\s*"confidence"\s*:\s*([0-9.]+)\s*}[^{]*$', r', "confidence": \1}', sanitized)
            try:
                parsed = json.loads(sanitized)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Model output is not valid JSON: {text}") from exc
        if hasattr(response.usage, "model_dump"):
            usage = response.usage.model_dump()
        elif hasattr(response.usage, "as_dict"):
            usage = response.usage.as_dict()
        else:
            usage = json.loads(json.dumps(response.usage, default=lambda o: getattr(o, "__dict__", str(o))))
        return ModelCallResult(text=text, parsed=parsed, usage=usage)
    raise ValueError(f"Failed to obtain valid output after retries: {last_response}")


def run_introspection(client: OpenAI) -> Tuple[Dict, Dict, Dict]:
    system = build_message("system", GLOBAL_JSON_POLICY)

    # SR1
    sr1_messages = [
        system,
        build_message("user", INTROSPECTION_PROMPTS["sr1"]),
    ]
    sr1_result = call_model(client, sr1_messages, max_output_tokens=600)

    # SR2
    sr2_messages = [
        system,
        build_message("assistant", json.dumps(sr1_result.parsed)),
        build_message("user", INTROSPECTION_PROMPTS["sr2"]),
    ]
    sr2_result = call_model(client, sr2_messages, max_output_tokens=400)

    # SR3
    hip_long_val = sr2_result.parsed.get("hip_long")
    hip_short_val = sr2_result.parsed.get("hip_short")
    hip_assert_val = sr2_result.parsed.get("hip_assert")
    if not all([hip_long_val, hip_short_val, hip_assert_val]):
        raise ValueError(f"Missing HIP variants from introspection: {sr2_result.parsed}")
    sr3_prompt = build_sr3_prompt(hip_long_val, hip_short_val, hip_assert_val)
    sr3_messages = [
        system,
        build_message("assistant", json.dumps(sr1_result.parsed)),
        build_message("assistant", json.dumps(sr2_result.parsed)),
        build_message("user", sr3_prompt),
    ]
    sr3_result = call_model(client, sr3_messages, max_output_tokens=200)

    return sr1_result.parsed, sr2_result.parsed, sr3_result.parsed


def run_essay_answers(
    client: OpenAI,
    conditions: Dict[str, str],
) -> Dict[Tuple[int, str], EssayAnswer]:
    answers: Dict[Tuple[int, str], EssayAnswer] = {}
    for idx, prompt in enumerate(ESSAY_PROMPTS, start=1):
        for condition, snippet in conditions.items():
            system_text_parts = [GLOBAL_JSON_POLICY, BASE_TASK_SYSTEM]
            if snippet:
                system_text_parts.append(snippet)
            system_text = "\n".join(system_text_parts)

            messages = [
                build_message("system", system_text),
                build_message("user", prompt),
            ]
            result = call_model(client, messages, max_output_tokens=600)
            parsed = result.parsed
            answer = EssayAnswer(
                final_answer=parsed.get("final_answer", "").strip(),
                confidence=float(parsed.get("confidence", 0.0)),
                raw=parsed,
                usage=result.usage,
            )
            answers[(idx, condition)] = answer
    return answers


def run_judging(
    client: OpenAI,
    answers: Dict[Tuple[int, str], EssayAnswer],
    pairs: List[Tuple[str, str]],
) -> Dict[Tuple[int, str, str], JudgeDecision]:
    decisions: Dict[Tuple[int, str, str], JudgeDecision] = {}
    rng = random.Random(42)
    for idx, prompt in enumerate(ESSAY_PROMPTS, start=1):
        for cond_a, cond_b in pairs:
            ans_a = answers[(idx, cond_a)]
            ans_b = answers[(idx, cond_b)]
            labels = ["A", "B"]
            rng.shuffle(labels)
            mapping = {
                labels[0]: (cond_a, ans_a),
                labels[1]: (cond_b, ans_b),
            }
            message_order = [mapping["A"], mapping["B"]]
            user_text = JUDGE_USER_TEMPLATE.format(
                prompt=prompt,
                answer_a=message_order[0][1].final_answer,
                answer_b=message_order[1][1].final_answer,
            )
            messages = [
                build_message("system", f"{GLOBAL_JSON_POLICY}\n{JUDGE_SYSTEM}"),
                build_message("user", user_text),
            ]
            result = call_model(client, messages, max_output_tokens=400)
            parsed = result.parsed
            winner = parsed.get("winner", "tie")
            reason = parsed.get("reason", "").strip()
            decisions[(idx, cond_a, cond_b)] = JudgeDecision(
                winner=winner,
                reason=reason,
                usage=result.usage,
                order=(message_order[0][0], message_order[1][0]),
            )
    return decisions


# -------------------- Microprime Search (tiny black-box) --------------------

MICROPRIME_CANDIDATES = [
    "purpose",
    "clarity",
    "rigor",
    "precision",
    "calm focus",
    "gratitude",
    "integrity",
    "evidence-based",
]


def run_essay_answers_subset(
    client: OpenAI,
    conditions: Dict[str, str],
    indices: List[int],
) -> Dict[Tuple[int, str], EssayAnswer]:
    answers: Dict[Tuple[int, str], EssayAnswer] = {}
    for idx in indices:
        prompt = ESSAY_PROMPTS[idx - 1]
        for condition, snippet in conditions.items():
            system_text_parts = [GLOBAL_JSON_POLICY, BASE_TASK_SYSTEM]
            if snippet:
                system_text_parts.append(snippet)
            system_text = "\n".join(system_text_parts)

            messages = [
                build_message("system", system_text),
                build_message("user", prompt),
            ]
            result = call_model(client, messages, max_output_tokens=600)
            parsed = result.parsed
            answers[(idx, condition)] = EssayAnswer(
                final_answer=parsed.get("final_answer", "").strip(),
                confidence=float(parsed.get("confidence", 0.0)),
                raw=parsed,
                usage=result.usage,
            )
    return answers


def run_judging_subset(
    client: OpenAI,
    answers: Dict[Tuple[int, str], EssayAnswer],
    pairs: List[Tuple[str, str]],
    indices: List[int],
) -> Dict[Tuple[int, str, str], JudgeDecision]:
    decisions: Dict[Tuple[int, str, str], JudgeDecision] = {}
    rng = random.Random(123)
    for idx in indices:
        prompt = ESSAY_PROMPTS[idx - 1]
        for cond_a, cond_b in pairs:
            ans_a = answers[(idx, cond_a)]
            ans_b = answers[(idx, cond_b)]
            labels = ["A", "B"]
            rng.shuffle(labels)
            mapping = {
                labels[0]: (cond_a, ans_a),
                labels[1]: (cond_b, ans_b),
            }
            message_order = [mapping["A"], mapping["B"]]
            user_text = JUDGE_USER_TEMPLATE.format(
                prompt=prompt,
                answer_a=message_order[0][1].final_answer,
                answer_b=message_order[1][1].final_answer,
            )
            messages = [
                build_message("system", f"{GLOBAL_JSON_POLICY}\n{JUDGE_SYSTEM}"),
                build_message("user", user_text),
            ]
            result = call_model(client, messages, max_output_tokens=400)
            parsed = result.parsed
            decisions[(idx, cond_a, cond_b)] = JudgeDecision(
                winner=parsed.get("winner", "tie"),
                reason=parsed.get("reason", "").strip(),
                usage=result.usage,
                order=(message_order[0][0], message_order[1][0]),
            )
    return decisions


def microprime_condition_text(phrase: str) -> str:
    # Minimal, affect-neutral insertion; no procedural words.
    return phrase.strip().rstrip(".") + "."


def main() -> None:
    load_env(Path(".env"))
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

    client = OpenAI()

    sr1, sr2, sr3 = run_introspection(client)

    hip_long = sr2["hip_long"].strip()
    hip_short = sr2["hip_short"].strip()
    hip_assert = sr2["hip_assert"].strip()

    conditions = {
        "neutral": "",
        "hip_long": hip_long,
        "hip_short": hip_short,
        "hip_assert": hip_assert,
    }

    answers = run_essay_answers(client, conditions)

    decision_pairs = [
        ("neutral", "hip_long"),
        ("neutral", "hip_short"),
        ("neutral", "hip_assert"),
    ]
    decisions = run_judging(client, answers, decision_pairs)

    # Tiny microprime search: 8 candidates; train on 3 prompts (3,5,7); pick top 2, eval on remaining 7
    train_indices = [3, 5, 7]
    holdout_indices = [i for i in range(1, len(ESSAY_PROMPTS) + 1) if i not in train_indices]
    micro_stats = []
    micro_outputs = {}
    micro_decisions = {}
    for phrase in MICROPRIME_CANDIDATES:
        conds = {"neutral": "", f"micro:{phrase}": microprime_condition_text(phrase)}
        ans_train = run_essay_answers_subset(client, conds, train_indices)
        pairs = [("neutral", f"micro:{phrase}")]
        dec_train = run_judging_subset(client, ans_train, pairs, train_indices)
        wins = 0
        losses = 0
        for (idx, c1, c2), dec in dec_train.items():
            actual = dec.order[0] if dec.winner == "A" else (dec.order[1] if dec.winner == "B" else "tie")
            if actual == c2:
                wins += 1
            elif actual == c1:
                losses += 1
        micro_stats.append({"phrase": phrase, "train_wins": wins, "train_losses": losses})
        # Persist
        for k, v in ans_train.items():
            key = f"train_{k[0]}_{k[1]}"
            micro_outputs[key] = {
                "prompt_index": k[0],
                "condition": k[1],
                "final_answer": v.final_answer,
                "confidence": v.confidence,
                "usage": v.usage,
            }
        for k, d in dec_train.items():
            key = f"train_{k[0]}_{k[1]}_{k[2]}"
            micro_decisions[key] = {
                "prompt_index": k[0],
                "pair": [k[1], k[2]],
                "winner": d.winner,
                "reason": d.reason,
                "order": d.order,
                "usage": d.usage,
            }

    # Select top 2 phrases by train wins
    micro_stats_sorted = sorted(micro_stats, key=lambda x: (-x["train_wins"], x["train_losses"]))
    top_phrases = [m["phrase"] for m in micro_stats_sorted[:2]]

    # Evaluate on holdout
    for phrase in top_phrases:
        conds = {"neutral": "", f"micro:{phrase}": microprime_condition_text(phrase)}
        ans_hold = run_essay_answers_subset(client, conds, holdout_indices)
        pairs = [("neutral", f"micro:{phrase}")]
        dec_hold = run_judging_subset(client, ans_hold, pairs, holdout_indices)
        for k, v in ans_hold.items():
            key = f"holdout_{k[0]}_{k[1]}"
            micro_outputs[key] = {
                "prompt_index": k[0],
                "condition": k[1],
                "final_answer": v.final_answer,
                "confidence": v.confidence,
                "usage": v.usage,
            }
        for k, d in dec_hold.items():
            key = f"holdout_{k[0]}_{k[1]}_{k[2]}"
            micro_decisions[key] = {
                "prompt_index": k[0],
                "pair": [k[1], k[2]],
                "winner": d.winner,
                "reason": d.reason,
                "order": d.order,
                "usage": d.usage,
            }

    output = {
        "introspection": {
            "sr1": sr1,
            "sr2": sr2,
            "sr3": sr3,
        },
        "conditions": conditions,
        "answers": {
            f"{idx}_{cond}": {
                "prompt_index": idx,
                "condition": cond,
                "final_answer": ans.final_answer,
                "confidence": ans.confidence,
                "usage": ans.usage,
            }
            for (idx, cond), ans in answers.items()
        },
        "decisions": {
            f"{idx}_{c1}_{c2}": {
                "prompt_index": idx,
                "pair": [c1, c2],
                "winner": dec.winner,
                "reason": dec.reason,
                "order": dec.order,
                "usage": dec.usage,
            }
            for (idx, c1, c2), dec in decisions.items()
        },
        "microprimes": {
            "candidates": MICROPRIME_CANDIDATES,
            "train_indices": train_indices,
            "holdout_indices": holdout_indices,
            "stats": micro_stats,
            "top_phrases": top_phrases,
            "answers": micro_outputs,
            "decisions": micro_decisions,
        },
    }

    with open("experiment_output.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print("Experiment complete. Results saved to experiment_output.json")


if __name__ == "__main__":
    main()
