# AI Companion

A lightweight AI companion with persistent memory, built with Python, OpenAI, and SQLite.

## Features

- AI conversation
- Persistent memory
- SQLite memory database
- Entity-aware memory
- Semantic memory retrieval
- Memory updates
- Automated memory evaluation

## Requirements

- Python 3.10+
- OpenAI API key

## Setup

Clone the repository and enter the project:

```bash
git clone <your-repository-url>
cd ai-companion-
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Run

```bash
python src/main.py
```

## Evaluation

Run the memory evaluation harness:

```bash
python src/eval/harness.py
```

Results are saved to:

```text
eval_results.json
```

## Configuration

The application uses:

- OpenAI for conversation, memory processing, and embeddings
- SQLite for persistent memory
- Python for memory retrieval and similarity matching

The SQLite database is created automatically.

## Environment Variables

Only one environment variable is required:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit `.env` or your API key to Git.

## Recommended .gitignore

```gitignore
.env
venv/
__pycache__/
*.pyc
memory.db
eval_results.json
companion_memory.db
```

