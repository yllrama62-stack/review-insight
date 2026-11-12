"""Review Insight: customer review analysis using Groq."""

import json

import pandas as pd
import streamlit as st
from openai import OpenAI

from core import (
    ReviewError,
    Settings,
    analyze,
    compare,
    markdown_report,
    read_reviews,
)


GROQ_ENDPOINT = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

SAMPLE_REVIEWS = [
    "The cappuccino tasted rich and smooth.",
    "Our coffees arrived cold.",
    "The queue took twenty minutes before anyone served us.",
    "The barista was patient and friendly.",
    "Too expensive for such a small sandwich.",
    "The lunch deal is excellent value.",
    "The music was so loud we could not talk.",
    "A peaceful place with comfortable chairs.",
    "My takeaway order arrived late.",
    "Delivery was fast and everything was intact.",
    "The cake was stale and dry.",
    "The croissant was fresh and buttery.",
    "Friendly staff but the wait was very long.",
    "Lovely coffee, though the sandwich was stale.",
    "The tables were dirty when we arrived.",
    "The room was bright and clean.",
    "The prices increased again and portions got smaller.",
    "Good value for a filling breakfast.",
    "Our delivery bag was torn and the drink spilled.",
    "The courier arrived right on time.",
    "Nobody helped us with a missing item.",
    "Staff quickly fixed the mistake with our order.",
    "The cafe opens at eight.",
    "Please add more vegan options to the menu.",
]

# These evaluation examples differ from the examples in prompts.py.
BENCHMARK_ITEMS = [
    ("The waiter ignored us for half an hour.", "service", "negative"),
    ("The cashier patiently explained every option.", "service", "positive"),
    ("The soup was fresh and delicious.", "quality", "positive"),
    ("The bread was moldy.", "quality", "negative"),
    ("A fair price for a generous meal.", "price", "positive"),
    ("Six euros for tap water is too much.", "price", "negative"),
    ("My food reached me much later than promised.", "delivery", "negative"),
    ("The package arrived earlier than expected.", "delivery", "positive"),
    ("The quiet seating area helped me concentrate.", "environment", "positive"),
    ("The room was unbearably noisy.", "environment", "negative"),
    ("The cafe is located next to the station.", "other", "neutral"),
    (
        "The receptionist was kind but repeatedly forgot our request.",
        "service",
        "mixed",
    ),
]

BENCHMARK = [
    {
        "id": f"B{number:03}",
        "text": text,
        "expected_topic": topic,
        "expected_sentiment": sentiment,
    }
    for number, (text, topic, sentiment) in enumerate(
        BENCHMARK_ITEMS, 1
    )
]


st.set_page_config(
    page_title="Review Insight",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Review Insight")
st.write("Turn customer reviews into evidence-backed business priorities.")
st.caption("AI provider: Groq")


with st.sidebar:
    st.header("Groq settings")

    api_key = st.text_input(
        "Groq API key",
        type="password",
        key="review_groq_api_key",
        help="Enter an active API key from Groq Console.",
    )

    st.caption(
        "Your key is used for this session and is not included in "
        "downloaded reports. Free-tier availability and quotas depend "
        "on your Groq account and selected model."
    )

    model = st.text_input(
        "Groq model",
        value=DEFAULT_MODEL,
        key="review_groq_model",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )

    max_tokens = st.slider(
        "Maximum output tokens per call",
        min_value=500,
        max_value=6000,
        value=6000,
        step=250,
    )

    batch_size = st.slider(
        "Reviews per API call",
        min_value=1,
        max_value=10,
        value=4,
    )

    st.caption(
        "Smaller batches reduce the amount processed per request. "
        "Some models include reasoning tokens in the output allowance. "
        "Model availability and account limits can change."
    )

    with st.expander("Connection details"):
        st.write("Provider: Groq")
        st.code(GROQ_ENDPOINT, language=None)
        st.caption(
            "The OpenAI Python package is used as a compatible client. "
            "Requests are sent to Groq, not OpenAI."
        )

    if st.button("Clear saved results"):
        for name in [
            "analysis",
            "original_reviews",
            "experiment",
        ]:
            st.session_state.pop(name, None)
        st.rerun()


def make_settings():
    return Settings(
        model=model.strip(),
        temperature=temperature,
        max_tokens=max_tokens,
        batch_size=batch_size,
    )


def make_client():
    if not api_key.strip():
        raise ReviewError(
            "Enter your Groq API key in the sidebar."
        )

    return OpenAI(
        api_key=api_key.strip(),
        base_url=GROQ_ENDPOINT,
        timeout=40.0,
        max_retries=0,
    )


def show_error(error):
    """Display useful guidance for errors raised by core.py."""
    message = str(error)
    lowered = message.lower()

    if "authentication failed" in lowered:
        st.error(
            "Groq rejected the credentials. Clear the key field and "
            "paste an active Groq key directly from Groq Console. "
            "Do not use an OpenAI or Gemini key."
        )

    elif "api rejected these settings" in lowered:
        st.error(
            "Groq rejected the request settings. Check that the model "
            "is available in your Groq project. Also confirm core.py "
            "uses max_completion_tokens in the API call."
        )

    else:
        st.error(message)

    if any(
        phrase in lowered
        for phrase in [
            "rate-limit",
            "rate limit",
            "timeout",
            "connection",
        ]
    ):
        st.caption(
            "Check your internet connection and your Groq account's "
            "quota. Wait before retrying if a limit has been reached."
        )


analysis_tab, experiment_tab, guide_tab = st.tabs([
    "Analyze reviews",
    "Prompt laboratory",
    "Project guide",
])


with analysis_tab:
    source = st.radio(
        "Choose your reviews",
        ["Sample cafe reviews", "Upload CSV"],
        horizontal=True,
    )

    reviews = None

    if source == "Sample cafe reviews":
        reviews = [
            {
                "id": f"R{number:03}",
                "text": text,
            }
            for number, text in enumerate(
                SAMPLE_REVIEWS, 1
            )
        ]

        st.caption(
            "These 24 sample reviews are synthetic demonstration data."
        )

        sample_csv = pd.DataFrame({
            "review": SAMPLE_REVIEWS,
        }).to_csv(index=False)

        st.download_button(
            "Download sample CSV",
            sample_csv,
            file_name="sample_reviews.csv",
            mime="text/csv",
        )

    else:
        upload = st.file_uploader(
            "Upload a UTF-8 CSV: up to 2 MB and 100 reviews",
            type=["csv"],
        )

        column = st.text_input(
            "Review column name",
            value="review",
        )

        if upload is not None:
            try:
                reviews, skipped = read_reviews(
                    upload.getvalue(),
                    column,
                )

                st.caption(
                    f"Skipped {skipped} blank review rows."
                )

            except ReviewError as error:
                st.error(str(error))

    if reviews:
        st.dataframe(
            pd.DataFrame(reviews),
            hide_index=True,
            width="stretch",
        )

        st.caption(
            f"{len(reviews)} reviews ready. "
            "Duplicate inputs count separately."
        )

    consent = st.checkbox(
        "I understand these reviews will be sent to Groq.",
        key="review_groq_consent",
    )

    st.caption(
        "Remove names, emails, and other personal or confidential "
        "information before uploading. Review Groq's current data-use "
        "terms. The app sends review text as provided."
    )

    if st.button(
        "Analyze feedback",
        type="primary",
        disabled=not reviews or not consent,
    ):
        st.session_state.pop("analysis", None)
        st.session_state.pop("original_reviews", None)

        progress = st.progress(
            0.0,
            text=(
                "Classifying reviews, then preparing "
                "recommendations..."
            ),
        )

        try:
            settings = make_settings()

            with make_client() as api:
                result = analyze(
                    api,
                    reviews,
                    settings,
                    progress=progress.progress,
                )

            result["metadata"]["provider"] = "Groq"
            result["metadata"]["endpoint"] = GROQ_ENDPOINT

            st.session_state["analysis"] = result
            st.session_state["original_reviews"] = reviews

        except ReviewError as error:
            show_error(error)

        except Exception:
            st.error(
                "Unexpected analysis failure. Check your configuration "
                "and retry. No partial report was saved."
            )

        finally:
            progress.empty()

    if "analysis" in st.session_state:
        result = st.session_state["analysis"]

        st.divider()
        st.subheader("Saved analysis")

        st.caption(
            "Changing inputs or settings does not update this report. "
            "Click Analyze feedback to generate a new report."
        )

        left, middle, right = st.columns(3)

        left.metric(
            "Reviews analyzed",
            len(result["labels"]),
        )

        complaint_count = sum(
            row["sentiment"] in ["negative", "mixed"]
            for row in result["labels"]
        )

        middle.metric(
            "Reviews with complaint signals",
            complaint_count,
        )

        right.metric(
            "API tokens used",
            result["metadata"]["total_tokens"],
        )

        st.subheader("Overview")
        st.write(result["report"]["summary"])

        topic_column, sentiment_column = st.columns(2)

        with topic_column:
            st.subheader("Primary topics")

            st.bar_chart(
                pd.Series(
                    result["counts"]["topics"],
                    name="Reviews",
                )
            )

        with sentiment_column:
            st.subheader("Sentiment")

            st.bar_chart(
                pd.Series(
                    result["counts"]["sentiments"],
                    name="Reviews",
                )
            )

        st.caption(
            "Each review receives one primary topic. Negative and "
            "mixed sentiments both count as complaint signals."
        )

        st.subheader("Prioritized recommendations")

        evidence = {
            row["id"]: row
            for row in result["labels"]
        }

        for number, action in enumerate(
            result["report"]["actions"], 1
        ):
            with st.container(border=True):
                st.markdown(
                    f"*{number}. {action['title']}*"
                )

                st.write(action["recommendation"])

                st.caption(
                    "Proposed success measure: "
                    + action["success_metric"]
                )

                for review_id in action["evidence_ids"]:
                    st.text(
                        f"{review_id}: "
                        f"{evidence[review_id]['quote']}"
                    )

        with st.expander(
            "Inspect classifications and original reviews"
        ):
            table = pd.DataFrame(
                result["labels"]
            ).merge(
                pd.DataFrame(
                    st.session_state["original_reviews"]
                ),
                on="id",
            )

            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
            )

        with st.expander("Run settings and metadata"):
            st.json(result["metadata"])

        st.download_button(
            "Download report (.md)",
            markdown_report(result),
            file_name="review-insight-report.md",
            mime="text/markdown",
        )

        st.download_button(
            "Download full analysis (.json)",
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="review-insight-analysis.json",
            mime="application/json",
        )

        st.info(
            "AI interpretations can be wrong. Evidence quotes are "
            "checked against the input, but recommendations still "
            "need human review."
        )


with experiment_tab:
    st.subheader("Compare prompting techniques")

    st.write(
        "Compare a zero-shot baseline against role assignment plus "
        "few-shot examples on the same 12 labeled synthetic reviews."
    )

    st.caption(
        "Both variants use your sidebar settings. Expected answer "
        "labels are excluded from the prompts sent to Groq."
    )

    with st.expander("Inspect the evaluation dataset"):
        st.dataframe(
            pd.DataFrame(BENCHMARK),
            hide_index=True,
            width="stretch",
        )

    experiment_consent = st.checkbox(
        "I understand the synthetic benchmark will be sent to Groq "
        "for both prompt variants.",
        key="review_groq_experiment_consent",
    )

    st.caption(
        "The comparison makes multiple API requests and uses your "
        "account's quota. Paid usage may incur charges if enabled."
    )

    if st.button(
        "Run prompt comparison",
        disabled=not experiment_consent,
    ):
        st.session_state.pop("experiment", None)

        try:
            settings = make_settings()

            with st.spinner(
                "Running both prompt variants..."
            ):
                with make_client() as api:
                    experiment = compare(
                        api,
                        BENCHMARK,
                        settings,
                    )

            experiment["provider"] = "Groq"
            experiment["endpoint"] = GROQ_ENDPOINT

            st.session_state["experiment"] = experiment

        except ReviewError as error:
            show_error(error)

        except Exception:
            st.error(
                "The experiment could not complete. "
                "Check your settings and retry."
            )

    if "experiment" in st.session_state:
        experiment = st.session_state["experiment"]

        scores = [
            {
                name: value
                for name, value in row.items()
                if name != "predictions"
            }
            for row in experiment["results"]
        ]

        st.dataframe(
            pd.DataFrame(scores),
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "Accuracy ranges from 0 to 1. For example, 0.75 means "
            "75% agreement with the expected labels."
        )

        st.caption(experiment["note"])

        with st.expander(
            "Inspect predictions and experiment settings"
        ):
            st.json(experiment)

        st.download_button(
            "Download experiment results",
            json.dumps(
                experiment,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="prompt-experiment.json",
            mime="application/json",
        )

    st.info(
        "For your presentation: run at temperature 0.2 and download "
        "the results. Then try 0.7 and save again. Compare errors, "
        "accuracy, time, and token usage. A more elaborate prompt "
        "is not guaranteed to score higher."
    )


with guide_tab:
    st.markdown("""
### How this project works

1. *Input:* preview synthetic reviews or upload a CSV.
2. *Classification:* the model assigns a topic, sentiment, and evidence quote.
3. *Validation:* Python checks IDs, labels, and exact evidence quotations.
4. *Aggregation:* Python calculates counts.
5. *Recommendations:* a second API request proposes evidence-linked actions.
6. *Export:* download reports, analyses, and actual experiment results.

### Prompt techniques

*Role assignment:* the model acts as a customer-experience analyst.

*Few-shot prompting:* examples demonstrate how reviews should be classified.

The Prompt laboratory compares these techniques with zero-shot instructions.

### Adjustable settings

Model, temperature, maximum output tokens, and batch size.

### Connection

This app uses Groq's OpenAI-compatible API endpoint.

The Python client remains named OpenAI, but requests are sent to Groq.
The openai/ prefix in the default model name identifies the model family;
it does not change the API provider.

Use an active Groq API key created in Groq Console.

### Error handling

The processing code handles invalid uploads, missing keys,
authentication failures, timeouts, rate limits, invalid output,
and unsupported settings.

### Limitations

Each review receives one primary topic. Classifications can be subjective.
The evaluation dataset is small and synthetic.

Exact quote validation does not guarantee correct interpretation.
Recommendations require human review.

Model availability and free-tier quotas depend on Groq's current
offerings and your account.

### Project files

- app.py: interface, sample data, and Groq connection.
- core.py: processing, API calls, validation, counts, and reports.
- prompts.py: instructions and examples sent to the model.
- requirements.txt: Python dependencies.
""")