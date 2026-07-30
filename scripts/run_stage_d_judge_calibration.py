from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_prompt(
    template: str, rubric: dict[str, Any], case: dict[str, Any]
) -> str:
    from verifiers.v1.judges.rubric import JSON_SUFFIX

    criteria = "\n".join(
        f"- {item['name']}: {item['text']} "
        f"(answer one of, worst to best: {', '.join(item['choices'])})"
        for item in rubric["criteria"]
    )
    reference = (
        "\nReference evidence:\n```\n"
        + "\n".join(case["reference_evidence"])
        + "\n```\n"
    )
    return (
        template.format(
            question=case["question"],
            reference=reference,
            response=case["response"],
            criteria=criteria,
        )
        + JSON_SUFFIX
    )


async def run(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    cases = load_cases(args.cases)
    template = args.prompt.read_text(encoding="utf-8")
    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_var, "EMPTY"),
        max_retries=2,
    )
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            prompt = render_prompt(template, rubric, case)
            for repeat in range(args.repeats):
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    top_p=1,
                    max_tokens=384,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                )
                text = response.choices[0].message.content or ""
                from verifiers.v1.judges.rubric import (
                    RubricVerdicts,
                    first_verdicts_object,
                )

                parsed_object = first_verdicts_object(text)
                parsed = None
                parse_error = None
                if parsed_object is not None:
                    try:
                        parsed = RubricVerdicts.model_validate(
                            parsed_object
                        ).model_dump()
                    except Exception as error:
                        parse_error = f"{type(error).__name__}: {error}"
                else:
                    parse_error = "no verdicts object"
                usage = response.usage
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "repeat": repeat,
                        "expected": case["expected"],
                        "response_text": text,
                        "parsed": parsed,
                        "parse_error": parse_error,
                        "usage": (
                            {
                                "prompt_tokens": usage.prompt_tokens,
                                "completion_tokens": usage.completion_tokens,
                            }
                            if usage is not None
                            else None
                        ),
                    }
                )
    finally:
        await client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-var", default="STAGE_D_JUDGE_API_KEY")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("datasets/stage-d/evidence-judge-calibration-v1.jsonl"),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("configs/stage-d/evidence-judge-prompt-v1.txt"),
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path("configs/stage-d/evidence-judge-rubric-v1.json"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
