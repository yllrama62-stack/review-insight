"""Review Insight processing, validation, and Groq API calls."""

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
    """An actionable error suitable for the interface."""


@dataclass(frozen=True)
class Settings:
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.2
    max_tokens: int = 6000
    batch_size: int = 4

    def _post_init_(self):
        if not self.model.strip():
            raise ReviewError("Enter a Groq model name.")

        if not 0 <= self.temperature <= 1:
            raise ReviewError(
                "Temperature must be between 0 and 1."
            )

        if not 500 <= self.max_tokens <= 6000:
            raise ReviewError(
                "Output limit must be between 500 and 6000 tokens."
            )

        if not 1 <= self.batch_size <= 10:
            raise ReviewError(
                "Batch size must be between 1 and 10."
            )


def read_reviews(raw, column="review"):
    """Read a UTF-8 CSV and assign stable per-run review IDs."""
    if not raw or len(raw) > 2_000_000:
        raise ReviewError(
            "Upload a nonempty CSV smaller than 2 MB."
        )

    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(
            io.StringIO(text),
            strict=True,
        )

        if (
            not reader.fieldnames
            or column not in reader.fieldnames
        ):
            raise ReviewError(
                f"CSV must contain a column named {column!r}."
            )

        reviews = []
        skipped = 0

        for row in reader:
            if None in row:
                raise ReviewError(
                    "CSV contains extra fields. Put quotation marks "
                    "around review text containing commas."
                )

            value = (row.get(column) or "").strip()

            if not value:
                skipped += 1
                continue

            if len(value) > 2500:
                raise ReviewError(
                    "Each review must contain at most "
                    "2,500 characters."
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
        raise ReviewError(
            "The selected column contains no nonempty reviews."
        )

    return reviews, skipped


def api_error_details(error):
    """Extract a useful error message without dumping request data."""
    body = getattr(error, "body", None)

    if isinstance(body, dict):
        details = body.get("error", body)

        if isinstance(details, dict):
            message = details.get("message")
            code = details.get("code")

            if isinstance(message, str):
                if code:
                    return f"{message} (code: {code})"
                return message

    message = getattr(error, "message", None)

    if isinstance(message, str) and message:
        return message

    return "The API did not provide additional error details."


def call_json(client, messages, settings, usage):
    """Call Groq and retry temporary failures at most three times."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=settings.model.strip(),
                messages=messages,
                temperature=settings.temperature,
                max_completion_tokens=settings.max_tokens,
                response_format={"type": "json_object"},
            )

            if response.usage:
                usage["total_tokens"] += (
                    response.usage.total_tokens or 0
                )

            if not response.choices:
                raise ReviewError(
                    "Groq returned no response choices. Try again."
                )

            choice = response.choices[0]

            if choice.finish_reason == "length":
                raise ReviewError(
                    "The response reached its output token limit. "
                    "Increase the output limit if possible or reduce "
                    "the number of reviews per call."
                )

            if choice.finish_reason != "stop":
                raise ReviewError(
                    "The model did not complete the response. "
                    f"Finish reason: {choice.finish_reason}. "
                    "Try again with a smaller batch."
                )

            content = choice.message.content

            if not isinstance(content, str) or not content.strip():
                raise ReviewError(
                    "The model returned an empty response."
                )

            try:
                return json.loads(content)

            except (ValueError, TypeError) as exc:
                raise ReviewError(
                    "The model returned invalid JSON. Try again."
                ) from exc

        except AuthenticationError as exc:
            raise ReviewError(
                "Authentication failed. Enter an active Groq API key. "
                "Do not use an OpenAI key, Gemini key, or deleted key."
            ) from exc

        except BadRequestError as exc:
            raise ReviewError(
                "Groq rejected the request: "
                + api_error_details(exc)
            ) from exc

        except RateLimitError as exc:
            details = api_error_details(exc)

            # Exhausted quotas should not be retried immediately.
            if any(
                text in details.lower()
                for text in [
                    "insufficient_quota",
                    "billing",
                    "daily limit",
                    "tokens per day",
                    "requests per day",
                ]
            ):
                raise ReviewError(
                    "Groq quota or account limit reached: "
                    + details
                ) from exc

            if attempt == 2:
                raise ReviewError(
                    "Groq's rate limit persisted after three attempts. "
                    "Wait before retrying and check your account limits. "
                    "Details: " + details
                ) from exc

            time.sleep(2 ** attempt)

        except APITimeoutError as exc:
            if attempt == 2:
                raise ReviewError(
                    "Groq timed out after three attempts. "
                    "Try again later or use a smaller batch."
                ) from exc

            time.sleep(2 ** attempt)

        except APIConnectionError as exc:
            if attempt == 2:
                raise ReviewError(
                    "Could not connect to Groq after three attempts. "
                    "Check your internet connection and try later."
                ) from exc

            time.sleep(2 ** attempt)

        except APIStatusError as exc:
            if exc.status_code >= 500 and attempt < 2:
                time.sleep(2 ** attempt)
                continue

            raise ReviewError(
                f"Groq API error {exc.status_code}: "
                + api_error_details(exc)
            ) from exc


def validate_labels(data, reviews):
    """Validate review IDs, classifications, and exact quotations."""
    rows = (
        data.get("reviews")
        if isinstance(data, dict)
        else None
    )

    originals = {
        review["id"]: review["text"]
        for review in reviews
    }

    if (
        not isinstance(rows, list)
        or len(rows) != len(originals)
    ):
        raise ReviewError(
            "The model returned the wrong number of reviews. "
            "Try again with a smaller batch."
        )

    seen = set()
    validated = []

    for row in rows:
        if not isinstance(row, dict):
            raise ReviewError(
                "The model returned an invalid review record."
            )

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

        if row.get("topic") not in TOPICS:
            raise ReviewError(
                "The model returned an unsupported topic."
            )

        if row.get("sentiment") not in SENTIMENTS:
            raise ReviewError(
                "The model returned an unsupported sentiment."
            )

        if (
            not isinstance(quote, str)
            or not quote.strip()
            or quote not in originals[review_id]
        ):
            raise ReviewError(
                "An evidence quotation did not match its original "
                "review. Try generating the analysis again."
            )

        seen.add(review_id)

        validated.append({
            "id": review_id,
            "topic": row["topic"],
            "sentiment": row["sentiment"],
            "quote": quote,
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
    """Classify reviews in batches using the selected prompt."""
    if not reviews:
        raise ReviewError(
            "Provide at least one review to classify."
        )

    labels = []

    for start in range(
        0,
        len(reviews),
        settings.batch_size,
    ):
        batch = reviews[
            start:start + settings.batch_size
        ]

        data = call_json(
            client,
            classification_messages(
                batch,
                enhanced=enhanced,
            ),
            settings,
            usage,
        )

        labels.extend(
            validate_labels(data, batch)
        )

        if progress:
            progress(
                (start + len(batch)) / len(reviews)
            )

    return labels


def aggregate(labels):
    """Calculate counts in Python rather than asking AI to count."""
    return {
        "topics": dict(
            Counter(
                row["topic"]
                for row in labels
            )
        ),
        "sentiments": dict(
            Counter(
                row["sentiment"]
                for row in labels
            )
        ),
        "complaints": dict(
            Counter(
                row["topic"]
                for row in labels
                if row["sentiment"] in [
                    "negative",
                    "mixed",
                ]
            )
        ),
    }


def validate_report(data, labels):
    """Validate report fields and references to review evidence."""
    valid_ids = {
        row["id"]
        for row in labels
    }

    if (
        not isinstance(data, dict)
        or not isinstance(data.get("summary"), str)
        or not data["summary"].strip()
    ):
        raise ReviewError(
            "The model returned an invalid summary."
        )

    actions = data.get("actions")

    if (
        not isinstance(actions, list)
        or not 1 <= len(actions) <= 3
    ):
        raise ReviewError(
            "The model must return one to three recommendations."
        )

    cleaned_actions = []

    for action in actions:
        if not isinstance(action, dict):
            raise ReviewError(
                "A recommendation has an invalid format."
            )

        for field in [
            "title",
            "recommendation",
            "success_metric",
        ]:
            value = action.get(field)

            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ReviewError(
                    "A recommendation is missing a required "
                    f"text field: {field}."
                )

        evidence_ids = action.get("evidence_ids")

        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(
                not isinstance(item, str)
                or item not in valid_ids
                for item in evidence_ids
            )
        ):
            raise ReviewError(
                "A recommendation references invalid review evidence."
            )

        cleaned_actions.append({
            "title": action["title"],
            "recommendation": action["recommendation"],
            "success_metric": action["success_metric"],
            "evidence_ids": list(
                dict.fromkeys(evidence_ids)
            ),
        })

    return {
        "summary": data["summary"],
        "actions": cleaned_actions,
    }


def analyze(client, reviews, settings, progress=None):
    """Run classification, counting, and recommendation generation."""
    started = time.monotonic()
    usage = {"total_tokens": 0}

    labels = classify(
        client,
        reviews,
        settings,
        enhanced=True,
        usage=usage,
        progress=progress,
    )

    counts = aggregate(labels)

    report_data = call_json(
        client,
        recommendation_messages(
            labels,
            counts,
        ),
        settings,
        usage,
    )

    report = validate_report(
        report_data,
        labels,
    )

    input_hash = sha256(
        json.dumps(
            reviews,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "report": report,
        "labels": labels,
        "counts": counts,
        "metadata": {
            "provider": "Groq",
            "settings": asdict(settings),
            "prompt_version": "1.1-role-fewshot-groq",
            "input_sha256": input_hash,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "seconds": round(
                time.monotonic() - started,
                2,
            ),
            **usage,
        },
    }


def compare(client, reviews, settings):
    """Compare baseline and enhanced prompts on labeled reviews."""
    if not reviews:
        raise ReviewError(
            "The evaluation dataset is empty."
        )

    for review in reviews:
        if (
            review.get("expected_topic") not in TOPICS
            or review.get("expected_sentiment") not in SENTIMENTS
        ):
            raise ReviewError(
                "An evaluation review has invalid expected labels."
            )

    gold = {
        review["id"]: review
        for review in reviews
    }

    # Remove expected answers before building any API prompts.
    inputs = [
        {
            "id": review["id"],
            "text": review["text"],
        }
        for review in reviews
    ]

    results = []

    for enhanced in [False, True]:
        usage = {"total_tokens": 0}
        started = time.monotonic()

        predictions = classify(
            client,
            inputs,
            settings,
            enhanced=enhanced,
            usage=usage,
        )

        topic_correct = sum(
            row["topic"]
            == gold[row["id"]]["expected_topic"]
            for row in predictions
        )

        sentiment_correct = sum(
            row["sentiment"]
            == gold[row["id"]]["expected_sentiment"]
            for row in predictions
        )

        results.append({
            "technique": (
                "Role + few-shot"
                if enhanced
                else "Baseline zero-shot"
            ),
            "topic_accuracy": (
                topic_correct / len(reviews)
            ),
            "sentiment_accuracy": (
                sentiment_correct / len(reviews)
            ),
            "seconds": round(
                time.monotonic() - started,
                2,
            ),
            **usage,
            "predictions": predictions,
        })

    return {
        "provider": "Groq",
        "settings": asdict(settings),
        "results": results,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "note": (
            "Small synthetic benchmark with subjective labels. "
            "Accuracy measures agreement with the expected labels; "
            "these results do not establish general accuracy."
        ),
    }


def markdown_report(result):
    """Create a readable report for downloading."""
    report = result["report"]

    parts = [
        "# Review Insight report",
        report["summary"],
        "## Review counts",
        (
            "Each review has one primary topic. Negative and mixed "
            "sentiments both contribute to complaint counts."
        ),
        "json",
        json.dumps(
            result["counts"],
            indent=2,
        ),
        "",
        "## Recommended actions",
    ]

    for number, action in enumerate(
        report["actions"],
        1,
    ):
        parts.extend([
            f"### {number}. {action['title']}",
            action["recommendation"],
            "Proposed measure: " + action["success_metric"],
            "Evidence: " + ", ".join(
                action["evidence_ids"]
            ),
        ])

    parts.append("## Evidence")

    for row in result["labels"]:
        parts.append(
            f"- {row['id']} | {row['topic']} | "
            f"{row['sentiment']}: {row['quote']}"
        )

    parts.extend([
        "## Run metadata",
        "json",
        json.dumps(
            result["metadata"],
            indent=2,
        ),
        "",
        (
            "AI output requires human review. Recommendations "
            "and success measures are proposals, not proven outcomes."
        ),
    ])

    return "\n\n".join(parts)