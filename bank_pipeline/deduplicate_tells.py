"""Phase 2b: Cluster raw tells into canonical templates with adaptation instructions.

Groups tells by category, uses LLM to identify semantic clusters, and produces
canonical templates that include adaptation guidance for Phase 3 tell injection.

Usage:
    python -m tells.deduplicate_tells --dry-run
    python -m tells.deduplicate_tells --category SYSTEM_INSTRUCTIONS
    python -m tells.deduplicate_tells

Outputs:
    tells/data/tell_templates.json
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tells.extract_tells import TELL_CATEGORIES

from tells.data_config import TELL_DB_PATH, TEMPLATES_PATH, WORK_DIR


# ---------------------------------------------------------------------------
# Loading and formatting
# ---------------------------------------------------------------------------

def load_tells_by_category(db_path: Path) -> dict[str, list[dict]]:
    """Load tell_db.json and group by category."""
    tells = json.loads(db_path.read_text())

    by_category: dict[str, list[dict]] = defaultdict(list)
    for tell in tells:
        category = tell.get("category", "UNKNOWN")
        by_category[category].append(tell)

    return dict(by_category)


def format_tells_for_prompt(tells: list[dict]) -> str:
    """Format tells into a concise summary for the clustering prompt."""
    lines = []
    for tell in tells:
        evidence = tell.get("evidence", tell.get("description", "N/A"))[:300]
        lines.append(
            f"[{tell['tell_id']}] source={tell.get('source_category', '?')}, "
            f"location={tell.get('location', '?')}, "
            f"difficulty={tell.get('transplant_difficulty', '?')}\n"
            f"  description: {tell.get('description', 'N/A')}\n"
            f"  evidence: {evidence}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Clustering prompt
# ---------------------------------------------------------------------------

CLUSTERING_PROMPT = """\
You are an expert at analyzing patterns in AI evaluation transcripts. You will receive a list of extracted "Tells" — concrete features that reveal a transcript comes from an evaluation rather than real-world deployment. All tells below belong to the category **{category}**.

## Your Task

1. **Cluster** the tells into semantically distinct sub-patterns. Group tells that describe the same underlying signal (e.g., all "Hint:" instructions are one cluster; all "$1M reward" language is another).

2. **For each cluster**, produce a canonical template with the following fields:

   - **template_id**: A short snake_case identifier (e.g., `system_hint_instructions`, `mcq_formatting`)
   - **category**: "{category}" (same for all)
   - **pattern_name**: A short human-readable name for this sub-pattern (e.g., "Hint/walkthrough instructions")
   - **description**: 2-3 sentences describing the tell pattern and *why* it signals evaluation. Be specific about what makes this detectable.
   - **adaptation_instructions**: Detailed, concrete guidance (3-5 sentences) for a writer LLM on how to transplant this tell into a deployment transcript. Cover:
     - Where to insert (system prompt, user message, etc.)
     - How to make it fit the existing conversation topic
     - What to adjust for agentic vs. chat transcripts
     - Common pitfalls to avoid (e.g., contradicting existing content)
   - **representative_examples**: An array of 2-3 verbatim evidence excerpts from the tells below, chosen to show variety within the cluster. Include the tell_id for traceability.
   - **compatibility_notes**: 1-2 sentences on which D-case formats this works best with, and any known limitations.
   - **source_tell_ids**: Array of all tell_ids in this cluster.
   - **frequency**: Number of tells in this cluster.
   - **typical_location**: The most common location for tells in this cluster (system_prompt, user_message, etc.)

## Tells to Cluster

{tells_text}

## Output Format

Return a JSON array of template objects. No markdown code fences, no extra text — only valid JSON.

Guidelines:
- Aim for 3-8 clusters per category (merge very small clusters of 1-2 tells into the nearest match).
- Every tell must be assigned to exactly one cluster.
- Adaptation instructions should be actionable — a writer LLM should be able to follow them without seeing the original evidence.
- Pick representative examples that show the range of variation within the cluster.
"""


def build_clustering_prompt(category: str, tells: list[dict]) -> str:
    """Build the clustering prompt for a single category."""
    return CLUSTERING_PROMPT.format(
        category=category,
        tells_text=format_tells_for_prompt(tells),
    )


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


async def call_clustering_llm(
    client, prompt: str, category: str, max_retries: int = 2
) -> list[dict] | None:
    """Call LLM and parse JSON response, retrying on parse failures."""
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model="openai/gpt-5.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=16384,
            )
            text = strip_markdown_fences(response.choices[0].message.content.strip())

            templates = json.loads(text)
            if not isinstance(templates, list):
                print(f"  WARNING [{category}]: LLM returned non-array JSON, wrapping")
                templates = [templates]
            return templates

        except json.JSONDecodeError as exc:
            print(f"  ERROR [{category}]: JSON parse failed "
                  f"(attempt {attempt + 1}/{max_retries + 1}): {exc}")
            print(f"    Response ends with: ...{text[-100:]}")
            if attempt < max_retries:
                print("    Retrying...")
                continue
            return None
        except Exception as exc:
            print(f"  ERROR [{category}]: LLM call failed: {exc}")
            return None
    return None


# ---------------------------------------------------------------------------
# Template validation
# ---------------------------------------------------------------------------

REQUIRED_TEMPLATE_FIELDS = [
    "template_id", "pattern_name", "description",
    "adaptation_instructions", "representative_examples",
    "source_tell_ids", "frequency",
]


def validate_templates(
    templates: list[dict], tells: list[dict], category: str
) -> list[dict]:
    """Validate templates: check all tells are assigned, no duplicates."""
    all_tell_ids = {t["tell_id"] for t in tells}
    assigned_ids: set[str] = set()

    valid = []
    for template in templates:
        missing_fields = [f for f in REQUIRED_TEMPLATE_FIELDS if f not in template]
        if missing_fields:
            tid = template.get("template_id", "?")
            print(f"  WARNING [{category}]: Template '{tid}' missing: {missing_fields}")

        template["category"] = category

        source_ids = set(template.get("source_tell_ids", []))

        # Flag unknown tell IDs
        unknown = source_ids - all_tell_ids
        if unknown:
            tid = template.get("template_id", "?")
            print(f"  WARNING [{category}]: Template '{tid}' references "
                  f"unknown tell IDs: {unknown}")
            template["source_tell_ids"] = [
                tid for tid in template["source_tell_ids"] if tid in all_tell_ids
            ]
            source_ids -= unknown

        # Flag double-assigned tells
        overlap = assigned_ids & source_ids
        if overlap:
            tid = template.get("template_id", "?")
            print(f"  WARNING [{category}]: Template '{tid}' has "
                  f"overlapping tell IDs: {overlap}")

        assigned_ids |= source_ids
        template["frequency"] = len(template.get("source_tell_ids", []))
        valid.append(template)

    # Report unassigned tells
    unassigned = all_tell_ids - assigned_ids
    if unassigned:
        print(f"  WARNING [{category}]: {len(unassigned)} tells not assigned: "
              f"{sorted(unassigned)[:10]}")

    return valid


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def run_deduplication(
    categories: list[str] | None,
    dry_run: bool,
    output_path_override: str | None = None,
):
    """Main deduplication pipeline."""
    db_path = TELL_DB_PATH
    tells_by_category = load_tells_by_category(db_path)

    total_tells = sum(len(v) for v in tells_by_category.values())
    print(f"Loaded tell_db.json: {total_tells} tells across "
          f"{len(tells_by_category)} categories")
    for cat, tells in sorted(tells_by_category.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(tells)} tells")

    if categories:
        tells_by_category = {
            k: v for k, v in tells_by_category.items() if k in set(categories)
        }
        print(f"\nFiltered to {len(tells_by_category)} requested categories")

    if dry_run:
        largest_category = max(tells_by_category, key=lambda k: len(tells_by_category[k]))
        prompt = build_clustering_prompt(
            largest_category, tells_by_category[largest_category]
        )
        print(f"\n{'='*60}")
        print(f"  DRY RUN — Sample prompt for {largest_category} "
              f"({len(tells_by_category[largest_category])} tells)")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"{'='*60}")
        print(prompt[:5000])
        if len(prompt) > 5000:
            print(f"\n... [{len(prompt) - 5000} more chars]")
        return

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    all_templates: list[dict] = []

    # Resume support: load existing templates
    if output_path_override:
        from pathlib import Path
        output_path = Path(output_path_override)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = WORK_DIR / "tell_templates.json"
    existing_categories: set[str] = set()
    if output_path.exists():
        existing_templates = json.loads(output_path.read_text())
        existing_categories = {t["category"] for t in existing_templates}
        all_templates.extend(existing_templates)
        print(f"\nResuming: {len(existing_templates)} existing templates "
              f"for categories: {sorted(existing_categories)}")

    # Process categories sequentially (each prompt is large)
    for category in TELL_CATEGORIES:
        if category not in tells_by_category:
            continue
        if category in existing_categories:
            print(f"\n  Skipping {category} (already processed)")
            continue

        tells = tells_by_category[category]
        print(f"\n{'='*60}")
        print(f"  Clustering {category}: {len(tells)} tells")
        print(f"{'='*60}")

        prompt = build_clustering_prompt(category, tells)
        print(f"  Prompt length: {len(prompt)} chars")

        templates = await call_clustering_llm(client, prompt, category)
        if templates is None:
            print(f"  FAILED — skipping {category}")
            continue

        templates = validate_templates(templates, tells, category)
        print(f"  Generated {len(templates)} templates:")
        for template in templates:
            tid = template.get("template_id", "?")
            name = template.get("pattern_name", "?")
            print(f"    - {tid}: {name} ({template['frequency']} tells)")

        all_templates.extend(templates)

        # Save after each category for incremental progress
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_templates, indent=2))
        print(f"  Saved intermediate results ({len(all_templates)} templates)")

    print(f"\n{'='*60}")
    print(f"  Total: {len(all_templates)} templates saved to {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster raw tells into canonical templates"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show category stats and sample prompt without making LLM calls",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for canonical templates (default: tells/data/tell_templates.json)",
    )
    parser.add_argument(
        "--category", type=str,
        help="Process a single category (for testing)",
    )
    args = parser.parse_args()

    categories = [args.category] if args.category else None

    asyncio.run(run_deduplication(
        categories=categories,
        dry_run=args.dry_run,
        output_path_override=args.output,
    ))


if __name__ == "__main__":
    main()
