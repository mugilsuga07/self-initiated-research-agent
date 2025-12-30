# Autonomous AI Research & Decision Support Agent

An AI-powered research agent that helps users make complex decisions through structured research and analysis.

## Features

- Accepts natural language questions
- Decomposes questions into research sub-topics
- Searches and gathers evidence from web sources
- Extracts claims and evidence from content
- Ranks sources by quality and credibility
- Identifies gaps, conflicts, and assumptions
- Generates clarifying questions
- Produces structured recommendations

## Architecture

The agent follows a 9-step pipeline:

1. **Input & Session Setup** - Validate and create research session
2. **Goal Decomposition** - Break down question into sub-questions using LLM
3. **Autonomous Discovery** - Search web for relevant sources
4. **Content Extraction** - Fetch and clean web page content
5. **Evidence Extraction** - Extract atomic claims from content
6. **Intelligent Ranking** - Score sources by recency, credibility, evidence
7. **Gap Analysis** - Identify unknowns, conflicts, assumptions
8. **Inquiry Mode** - Generate clarifying questions
9. **Final Recommendation** - Produce structured decision with reasoning

## Installation

```bash
pip install -e .
```

For the web UI:
```bash
pip install -e ".[ui]"
```

## Configuration

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

### Web UI (Streamlit)

```bash
streamlit run src/ui/streamlit_app.py
```

### Command Line

```bash
python -m src.main
```

Or with a question:

```bash
python -m src.main "Understand the latest research on self-driving cars"
```

## Project Structure

```
src/
  agent/       - Planning and decision making
    planner.py    - Goal decomposition
    decision.py   - Final recommendation generation
  analysis/    - Gap detection and clarification
    gaps.py       - Identifies unknowns and conflicts
    clarifier.py  - Generates clarifying questions
  llm/         - LLM client wrapper
    client.py     - OpenAI API abstraction
  models/      - Data models
    session.py    - Session management
    source.py     - Source and search results
    claim.py      - Evidence claims
  research/    - Search, extraction, and ranking
    search.py     - Web search integration
    extractor.py  - Content extraction
    claims.py     - Claim extraction from content
    ranker.py     - Source quality ranking
  ui/          - Streamlit web interface
    streamlit_app.py - Main UI
    pipeline.py      - Research pipeline runner
```

## Tech Stack

- **Python 3.11+**
- **OpenAI API** - LLM for reasoning and extraction
- **Tavily API** - Web search
- **Streamlit** - Web UI
- **Trafilatura** - Web content extraction
- **Rich** - Terminal formatting

## License

MIT License - see LICENSE file for details.
