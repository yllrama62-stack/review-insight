"""Input validation, API calls, analysis, and prompt experiments."""

import csv
import io
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from prompts import (
    SENTIMENTS,
    TOPICS,
    classification_messages,
    recommendation_messages,
)


class ReviewError(ValueError):
    """An error that can be displayed to the user."""


@dataclass(frozen=True)
class Settings:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 2500
    batch_size: int = 8

    def __post_init__(self):
        if not self.model.strip():
            raise ReviewError("Enter a model name.")
        if not 0 <= self.temperature <= 1:
            raise ReviewError("Temperature must be between 0 and 1.")
        if not 500 <= self.max_tokens <= 6000:
            raise ReviewError("Output limit must be 500–6000 tokens.")
        if not 1 <= self.batch_size <= 10:
            raise ReviewError("Batch size must be between 1 and 10.")


def read_reviews(raw, column="review"):
    """Read a UTF-8 CSV, assign IDs, and count skipped blank rows."""
    if not raw or len(raw) > 2_000_000:
        raise ReviewError("Upload a nonempty CSV smaller than 2 MB.")

    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text), strict=True)

        if not reader.fieldnames or column not in reader.fieldnames:
            raise ReviewError(
                f"CSV must contain a column named {column!r}."
            )

        reviews = []
        skipped = 0

        for row in reader:
            if None in row:
                raise ReviewError(
                    "CSV has extra fields. Put quotation marks around "
                    "review text containing commas."
                )

            value = (row.get(column) or "").strip()

            if not value:
                skipped += 1
                continue

            if len(value) > 2500:
                raise ReviewError(
                    "Each review must contain at most 2,500 characters."
                )

            reviews.append({
                "id": f"R{len(reviews) + 1:03}",
                "text": value,
            })

            if len(reviews) > 100:
                raise ReviewError(
                    "Upload at most 100 nonempty reviews per run."
                )

    except (UnicodeError, csv.Error) as exc:
        raise ReviewError(
            "Use a valid UTF-8 CSV with correctly quoted fields."
        ) from exc

    if not reviews:
        raise ReviewError("The selected column contains no reviews.")

    return reviews, skipped


def call_json(client, messages, settings, usage):
    """Make an API call with bounded retries for temporary failures."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                response_format={"type": "json_object"},
            )

            if response.usage:
                usage["total_tokens"] += response.usage.total_tokens

            if (
                not response.choices
                or response.choices[0].finish_reason != "stop"
            ):
                raise ReviewError(
                    "The response did not finish. Increase the output "
                    "limit or reduce the batch size."
                )

            try:
                return json.loads(
                    response.choices[0].message.content or ""
                )
            except (ValueError, TypeError) as exc:
                raise ReviewError(
                    "The model returned invalid JSON. Try again."
                ) from exc

        except AuthenticationError as exc:
            raise ReviewError(
                "Authentication failed. Check your API key."
            ) from exc

        except BadRequestError as exc:
            raise ReviewError(
                "The API rejected these settings. Use a model supporting "
                "Chat Completions, JSON output, temperature, and max_tokens."
            ) from exc

        except (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        ) as exc:
            if attempt == 2:
                raise ReviewError(
                    "Connection, timeout, or rate-limit failure persisted "
                    "after three attempts. Check internet access, API "
                    "credit, and account limits."
                ) from exc

            time.sleep(2 ** attempt)

        except APIStatusError as exc:
            if exc.status_code >= 500 and attempt < 2:
                time.sleep(2 ** attempt)
                continue

            raise ReviewError(
                "The API service failed. Try again later."
            ) from exc


def validate_labels(data, reviews):
    """Reject invented IDs, invalid labels, and fabricated quotes."""
    rows = data.get("reviews") if isinstance(data, dict) else None
    originals = {review["id"]: review["text"] for review in reviews}

    if not isinstance(rows, list) or len(rows) != len(originals):
        raise ReviewError(
            "The model returned the wrong number of reviews."
        )

    seen = set()
    validated = []

    for row in rows:
        if not isinstance(row, dict):
            raise ReviewError("The model returned an invalid record.")

        review_id = row.get("id")
        quote = row.get("quote")

        if (
            not isinstance(review_id, str)
            or review_id not in originals
            or review_id in seen
        ):
            raise ReviewError(
                "The model returned duplicate or unknown review IDs."
            )

        if (
            row.get("topic") not in TOPICS
            or row.get("sentiment") not in SENTIMENTS
        ):
            raise ReviewError(
                "The model returned an unsupported classification."
            )

        if (
            not isinstance(quote, str)
            or not quote.strip()
            or quote not in originals[review_id]
        ):
            raise ReviewError(
                "An evidence quote did not match its original review. "
                "Try the analysis again."
            )

        seen.add(review_id)
        validated.append({
            key: row[key]
            for key in ["id", "topic", "sentiment", "quote"]
        })

    return validated


def classify(
    client,
    reviews,
    settings,
    enhanced,
    usage,
    progress=None,
):
    """Classify reviews in small batches."""
    labels = []

    for start in range(0, len(reviews), settings.batch_size):
        batch = reviews[start:start + settings.batch_size]

        data = call_json(
            client,
            classification_messages(batch, enhanced),
            settings,
            usage,
        )

        labels.extend(validate_labels(data, batch))

        if progress:
            progress((start + len(batch)) / len(reviews))

    return labels


def aggregate(labels):
    """Calculate counts in Python rather than asking AI to count."""
    return {
        "topics": dict(Counter(row["topic"] for row in labels)),
        "sentiments": dict(
            Counter(row["sentiment"] for row in labels)
        ),
        "complaints": dict(Counter(
            row["topic"]
            for row in labels
            if row["sentiment"] in ["negative", "mixed"]
        )),
    }


def validate_report(data, labels):
    """Check required report fields and supporting review IDs."""
    valid_ids = {row["id"] for row in labels}

    if (
        not isinstance(data, dict)
        or not isinstance(data.get("summary"), str)
        or not data["summary"].strip()
    ):
        raise ReviewError("The model returned an invalid summary.")

    actions = data.get("actions")

    if not isinstance(actions, list) or not 1 <= len(actions) <= 3:
        raise ReviewError(
            "The model must return one to three recommendations."
        )

    cleaned_actions = []

    for action in actions:
        if not isinstance(action, dict):
            raise ReviewError("An action has an invalid format.")

        for key in ["title", "recommendation", "success_metric"]:
            if (
                not isinstance(action.get(key), str)
                or not action[key].strip()
            ):
                raise ReviewError(
                    "An action is missing a required text field."
                )

        evidence_ids = action.get("evidence_ids")

        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(
                not isinstance(item, str) or item not in valid_ids
                for item in evidence_ids
            )
        ):
            raise ReviewError(
                "An action references missing or invalid evidence."
            )

        cleaned_actions.append({
            key: action[key]
            for key in [
                "title",
                "recommendation",
                "success_metric",
                "evidence_ids",
            ]
        })

    return {
        "summary": data["summary"],
        "actions": cleaned_actions,
    }


def analyze(client, reviews, settings, progress=None):
    """Run classification, aggregation, and recommendation generation."""
    started = time.monotonic()
    usage = {"total_tokens": 0}

    labels = classify(
        client, reviews, settings, True, usage, progress
    )
    counts = aggregate(labels)

    data = call_json(
        client,
        recommendation_messages(labels, counts),
        settings,
        usage,
    )

    report = validate_report(data, labels)

    input_hash = sha256(
        json.dumps(reviews, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "report": report,
        "labels": labels,
        "counts": counts,
        "metadata": {
            "settings": asdict(settings),
            "prompt_version": "1.0-role-fewshot",
            "input_sha256": input_hash,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": round(time.monotonic() - started, 2),
            **usage,
        },
    }


def compare(client, reviews, settings):
    """Compare two prompts using reviews with expected answer labels."""
    results = []
    gold = {review["id"]: review for review in reviews}

    for enhanced in [False, True]:
        usage = {"total_tokens": 0}
        started = time.monotonic()

        predictions = classify(
            client, reviews, settings, enhanced, usage
        )

        topic_correct = sum(
            row["topic"] == gold[row["id"]]["expected_topic"]
            for row in predictions
        )
        sentiment_correct = sum(
            row["sentiment"]
            == gold[row["id"]]["expected_sentiment"]
            for row in predictions
        )

        results.append({
            "technique": (
                "Role + few-shot" if enhanced else "Baseline zero-shot"
            ),
            "topic_accuracy": topic_correct / len(reviews),
            "sentiment_accuracy": sentiment_correct / len(reviews),
            "seconds": round(time.monotonic() - started, 2),
            **usage,
            "predictions": predictions,
        })

    return {
        "settings": asdict(settings),
        "results": results,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Small synthetic benchmark with subjective labels. "
            "Results do not establish general accuracy."
        ),
    }


def markdown_report(result):
    """Create a readable report that can be downloaded."""
    report = result["report"]

    parts = [
        "# Review Insight report",
        report["summary"],
        "## Review counts",
        "Each review has one primary topic.",
        json.dumps(result["counts"], indent=2),
        "## Recommended actions",
    ]

    for number, action in enumerate(report["actions"], 1):
        parts.extend([
            f"### {number}. {action['title']}",
            action["recommendation"],
            "Proposed measure: " + action["success_metric"],
            "Evidence: " + ", ".join(action["evidence_ids"]),
        ])

    parts.append("## Evidence")

    for row in result["labels"]:
        parts.append(
            f"- {row['id']} | {row['topic']} | "
            f"{row['sentiment']}: {row['quote']}"
        )

    parts.extend([
        "## Run metadata",
        json.dumps(result["metadata"], indent=2),
        "AI output requires human review. Recommendations are proposals.",
    ])

    return "\n\n".join(parts)