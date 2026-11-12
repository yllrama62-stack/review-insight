# Review Insight

*Student:* Yll Rama  
*Project:* Individual generative AI application

## Purpose

Review Insight helps small businesses understand customer feedback.
It converts reviews into topic and sentiment counts, a summary,
and prioritized recommendations linked to supporting review evidence.

## Features

- Preview 24 synthetic café reviews or upload a CSV.
- Classify reviews by primary topic and sentiment.
- Validate evidence quotations against the original reviews.
- Calculate counts in Python and display charts.
- Generate up to three evidence-linked recommendations.
- Compare zero-shot prompting with role assignment plus few-shot examples.
- Download Markdown reports, JSON analyses, and experiment results.

## Technology

- Python
- Streamlit
- pandas
- Groq API through the OpenAI-compatible Python client
- Default model: openai/gpt-oss-120b

Model availability and API quotas depend on the Groq account.

## Project files

| File | Purpose |
|---|---|
| app.py | Interface, sample data, settings, and Groq connection |
| core.py | Input processing, API calls, validation, counts, and exports |
| prompts.py | Classification and recommendation prompts |
| requirements.txt | Required Python packages |

## Run on Windows

Install Python 3.10 or newer. Download and extract this repository.
Open PowerShell in the extracted folder, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```


Open http://localhost:8501.

Enter an active Groq API key in the sidebar.
Do not put API keys into source files or upload them to GitHub.

To stop the application, press Ctrl+C in its PowerShell window.

## How it works

1. Validate the CSV and assign review IDs.
2. Send reviews to the model in batches.
3. Receive a primary topic, sentiment, and exact evidence quotation.
4. Validate IDs, labels, and quotations in Python.
5. Calculate topic, sentiment, and complaint counts in Python.
6. Send validated evidence and counts to a second prompt.
7. Display the summary and recommendations with supporting review IDs.
8. Export the results and run metadata.

Each review receives one primary topic. Negative and mixed sentiments
both contribute to complaint counts.

## Prompt engineering

### Role assignment

The model acts as a careful customer-experience analyst.

### Few-shot prompting

Three examples demonstrate topic classification, sentiment,
and the expected evidence format.

### Structured output

Prompts request JSON so responses can be validated and processed.

### Prompt comparison

The Prompt laboratory compares a zero-shot baseline with role assignment
plus few-shot examples on 12 labeled synthetic reviews.

Expected labels are excluded from API prompts. The experiment reports
topic accuracy, sentiment accuracy, time, token usage, and predictions.

Run the experiment and save its actual results before claiming that one
technique performs better. The small synthetic dataset does not establish
general accuracy.

## Adjustable settings

- Model
- Temperature
- Maximum completion tokens
- Reviews per API call

## Error handling

The app handles invalid input, missing credentials, authentication failures,
timeouts, connection problems, rate limits, and invalid model output.

It rejects unknown review IDs, unsupported labels, fabricated evidence
quotations, and recommendations referencing nonexistent review IDs.

Temporary API failures are retried up to three times.

## Input limits

- UTF-8 CSV with a review column
- Maximum file size: 2 MB
- Maximum nonempty reviews: 100
- Maximum characters per review: 2,500

Blank reviews are skipped. Duplicate reviews are retained.

## Limitations and privacy

AI classifications and recommendations can be wrong.
Matching a quotation does not prove that its interpretation is correct.
Recommendations require human review.

Remove personal or confidential data before uploading reviews.
Review Groq's current data-use terms. Downloaded reports contain
review excerpts.

## Verification and evaluation status

Repository source files passed syntax checks and basic checks for input
parsing, evidence validation, counting, and exclusion of evaluation labels.

These checks do not establish successful live API generation.
Add actual analysis and experiment exports as supporting evidence.

## AI assistance

Code, troubleshooting, and documentation were developed with AI assistance.
The student is responsible for reviewing, understanding, testing, and
disclosing this assistance according to course policy.
