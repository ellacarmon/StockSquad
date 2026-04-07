# Azure AI Foundry Evaluation Integration - Roadmap

> **Status:** Planned Feature
> **Priority:** Phase 3 Enhancement
> **Estimated Effort:** 5-8 weeks
> **Dependencies:** `azure-ai-evaluation` SDK

---

## Overview

Azure AI Foundry Evaluation provides a comprehensive framework to measure and monitor the quality of AI-generated outputs. For StockSquad, this means systematically evaluating whether agents produce high-quality, accurate, and reliable stock analysis reports.

---

## Problem Statement

### Current State
StockSquad produces multi-agent analysis reports, but has **no systematic way to measure**:
- Is the TechnicalAgent's analysis coherent and accurate?
- Does the Devil's Advocate actually challenge assumptions effectively?
- Are agent responses grounded in the data they're given?
- Is the final report comprehensive and useful?
- Are there safety issues (hallucinations, overconfident predictions)?

### With Evaluation Integration
- **Automated, quantitative scoring** on every analysis run
- Quality monitoring over time
- A/B testing different agent prompts
- Regression detection when making changes
- Confidence scoring for end users
- Compliance documentation for financial use cases

---

## Azure AI Evaluation SDK Capabilities

### Built-in Evaluators

**Quality Metrics (AI-Assisted)**
- **Coherence** - Does the reasoning flow logically?
- **Fluency** - Is the text well-written and professional?
- **Relevance** - Does the response address the task?
- **Groundedness** - Are claims backed by provided data?

**RAG-Specific Metrics**
- **Retrieval Relevance** - Are retrieved documents contextually useful?
- **Context Utilization** - Does the agent use retrieved context effectively?

**Agent-Specific Metrics**
- **Tool Call Accuracy** - Did agents call tools correctly?
- **Task Completion** - Did each agent accomplish its assigned task?

**Safety & Risk Metrics**
- Hallucination detection
- Overconfidence detection
- Content safety (hate, violence, protected materials)

### Installation
```bash
pip install azure-ai-evaluation
```

---

## Evaluation Metrics for StockSquad Agents

### 1. Quality Metrics Application

| Metric | Application in StockSquad | Example |
|--------|---------------------------|---------|
| **Coherence** | Does the agent's reasoning flow logically? | "TechnicalAgent cited RSI overbought but then recommended buying - coherence score: 2/5" |
| **Fluency** | Is the report well-written and professional? | Evaluate OrchestratorAgent's final synthesis |
| **Relevance** | Does the response address the analysis goal? | "DevilsAdvocateAgent introduced crypto analysis when ticker was AAPL - relevance: 3/5" |
| **Groundedness** | Are claims backed by the provided data? | "SentimentAgent claimed 'overwhelmingly positive' but news data showed mixed sentiment - groundedness: 2/5" |

### 2. RAG Metrics for Long-Term Memory

| Metric | Application | Use Case |
|--------|-------------|----------|
| **Retrieval Relevance** | Are retrieved past analyses actually relevant? | When OrchestratorAgent retrieves past AAPL analysis, is it contextually useful? |
| **Context Utilization** | Does the agent use retrieved memory effectively? | "Agent retrieved 3 past analyses but only referenced 1" |

### 3. Agent-Specific Metrics

| Metric | Application | Example |
|--------|-------------|---------|
| **Tool Call Accuracy** | Did DataAgent call yfinance correctly? | Detect if agent requested wrong ticker or date range |
| **Task Completion** | Did each agent accomplish its assigned task? | "FundamentalsAgent assigned to analyze P/E ratio but skipped it" |

### 4. Safety & Risk Metrics

| Metric | Why It Matters | Example |
|--------|----------------|---------|
| **Financial Disclaimer Detection** | Ensure reports include proper disclaimers | "Report missing risk warning - safety score: FAIL" |
| **Overconfidence Detection** | Flag reports making absolute predictions | "Agent used 'guaranteed to rise' - overconfidence detected" |
| **Hallucination Detection** | Catch fabricated data or false claims | "Agent cited Q4 earnings that haven't been released" |

### 5. Custom Business Metrics

StockSquad-specific evaluators to create:

| Custom Metric | What It Measures |
|---------------|------------------|
| **Devil's Advocate Effectiveness** | Does the counterargument actually challenge the bull case? Score based on unique risk identification |
| **Data Citation Rate** | % of claims backed by specific data points |
| **Multi-Agent Consensus Coherence** | Do agent conclusions align with each other's data? |
| **Prediction Calibration** | Historical accuracy of ML signal scores vs actual outcomes |

---

## Implementation Architecture

### Code Structure

```
StockSquad/
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py              # Main evaluation orchestrator
│   ├── custom_evaluators.py      # StockSquad-specific evaluators
│   ├── metrics_tracker.py        # Track metrics over time
│   └── ab_testing.py             # A/B test framework for prompts
├── agents/
│   └── orchestrator.py           # Modified to call evaluator
├── ui/
│   └── evaluation_dashboard.py   # Streamlit evaluation dashboard
└── data/
    └── evaluations.db            # SQLite store for evaluation results
```

### 1. Core Evaluator Class

```python
# evaluation/evaluator.py

from azure.ai.evaluation import (
    CoherenceEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    FluencyEvaluator
)

class StockSquadEvaluator:
    """Evaluates agent outputs for quality and accuracy"""

    def __init__(self, azure_openai_config):
        # Initialize built-in evaluators
        self.coherence_eval = CoherenceEvaluator(azure_openai_config)
        self.groundedness_eval = GroundednessEvaluator(azure_openai_config)
        self.relevance_eval = RelevanceEvaluator(azure_openai_config)

    def evaluate_agent_output(self, agent_name, query, response, context=None):
        """Evaluate a single agent's output"""
        results = {}

        # Coherence: Is the reasoning logical?
        results['coherence'] = self.coherence_eval(
            query=query,
            response=response
        )

        # Groundedness: Is it backed by data?
        if context:
            results['groundedness'] = self.groundedness_eval(
                query=query,
                response=response,
                context=context
            )

        # Relevance: Does it address the task?
        results['relevance'] = self.relevance_eval(
            query=query,
            response=response,
            context=context
        )

        return results

    def evaluate_full_report(self, ticker, report, agent_outputs, market_data):
        """Evaluate the complete StockSquad analysis"""
        evaluation = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'agent_scores': {},
            'overall_score': 0
        }

        # Evaluate each agent
        for agent_name, output in agent_outputs.items():
            evaluation['agent_scores'][agent_name] = self.evaluate_agent_output(
                agent_name=agent_name,
                query=f"Analyze {ticker}",
                response=output,
                context=market_data
            )

        # Custom: Devil's Advocate effectiveness
        evaluation['devils_advocate_score'] = self._evaluate_counterargument(
            bull_case=agent_outputs.get('fundamentals'),
            bear_case=agent_outputs.get('devils_advocate')
        )

        # Store evaluation results
        self._store_evaluation(evaluation)

        return evaluation
```

### 2. Integration with Orchestrator

```python
# Modified: agents/orchestrator.py

class OrchestratorAgent:
    def __init__(self, ...):
        # ... existing code ...
        self.evaluator = StockSquadEvaluator(config) if config.enable_evaluation else None

    def analyze_stock(self, ticker):
        # ... run all agents ...

        # NEW: Evaluate the analysis
        if self.evaluator:
            evaluation_results = self.evaluator.evaluate_full_report(
                ticker=ticker,
                report=final_report,
                agent_outputs={
                    'technical': technical_agent.last_output,
                    'sentiment': sentiment_agent.last_output,
                    'fundamentals': fundamentals_agent.last_output,
                    'devils_advocate': devils_advocate.last_output
                },
                market_data=data_agent.last_data
            )

            # Store evaluation in memory
            self.long_term_memory.store_evaluation(ticker, evaluation_results)

            # Add quality score to report
            final_report += self._format_quality_score(evaluation_results)

        return final_report
```

### 3. Metrics Tracking Over Time

```python
# evaluation/metrics_tracker.py

class EvaluationMetricsTracker:
    """Tracks evaluation metrics over time"""

    def __init__(self, db_path="data/evaluations.db"):
        self.db = sqlite3.connect(db_path)
        self._init_schema()

    def get_agent_performance_trends(self, agent_name, days=30):
        """Track how an agent's quality scores trend over time"""
        # Query evaluations from the last N days
        # Return: avg coherence, groundedness, relevance

    def detect_quality_regression(self, threshold=0.5):
        """Alert if agent quality drops below threshold"""
        # Compare recent scores to historical baseline
        # Return warnings for agents below threshold

    def get_report_quality_distribution(self):
        """Show distribution of report quality scores"""
        # Used for dashboard visualization
```

### 4. Custom Evaluators

```python
# evaluation/custom_evaluators.py

class DevilsAdvocateEvaluator:
    """Evaluates effectiveness of Devil's Advocate counterarguments"""

    def evaluate(self, bull_case: str, bear_case: str) -> float:
        """
        Scores 0-5 based on:
        - Number of unique risks identified
        - Specificity of counterarguments
        - Whether bear case uses different data than bull case
        """
        # Implementation using Azure OpenAI

class DataCitationEvaluator:
    """Measures % of claims backed by specific data points"""

    def evaluate(self, response: str, market_data: dict) -> dict:
        """
        Returns:
        - citation_rate: % of claims with data references
        - uncited_claims: List of claims without citations
        """
        # Implementation using NLP analysis

class MultiAgentCoherenceEvaluator:
    """Checks if agent conclusions align with each other's data"""

    def evaluate(self, agent_outputs: dict) -> dict:
        """
        Returns:
        - coherence_score: 0-5
        - conflicts: List of contradictory claims between agents
        """
        # Implementation comparing agent outputs
```

---

## Practical Use Cases

### 1. Development Workflow

```bash
# Before making changes - establish baseline
python main.py evaluate-baseline --runs 10  # Run 10 analyses, track scores

# Make changes to TechnicalAgent prompt
vim agents/technical_agent.py

# After changes - compare
python main.py evaluate-comparison --runs 10  # Compare new vs baseline

# Output:
# ✅ Coherence: 4.2 → 4.5 (+7%)
# ⚠️  Groundedness: 4.1 → 3.8 (-7%)  # Regression detected!
# ✅ Relevance: 4.3 → 4.4 (+2%)
```

### 2. Prompt Engineering A/B Testing

Test different agent prompts systematically:

```python
# A/B test Devil's Advocate prompts
prompts = [
    "Challenge the analysis by identifying risks...",
    "Act as a skeptical investor and critique...",
    "Identify blind spots and overoptimistic assumptions..."
]

for prompt in prompts:
    scores = run_evaluation_with_prompt(prompt, test_tickers=['AAPL', 'MSFT', 'NVDA'])
    # Pick the prompt with highest "counterargument quality" score
```

### 3. User Trust Indicators

Add quality scores to reports:

```markdown
# StockSquad Analysis - AAPL

📊 **Report Quality Score: 4.3/5**
- Coherence: ⭐⭐⭐⭐⭐ (4.5/5)
- Groundedness: ⭐⭐⭐⭐ (4.2/5) - 98% of claims cited data
- Agent Consensus: ⭐⭐⭐⭐ (4.1/5)

...rest of report...
```

### 4. Continuous Monitoring

```python
# Auto-evaluate every analysis run
# Track metrics in dashboard
# Alert on quality regression

if evaluation_score < baseline_threshold:
    logger.warning(f"Quality regression detected for {agent_name}")
    send_alert(f"Agent {agent_name} coherence dropped from 4.2 to 3.5")
```

### 5. Compliance & Auditability

For financial applications, provide quantitative evidence:

```
StockSquad Quality Report (Last 1000 Analyses)
- Average Groundedness Score: 4.2/5
- False Claim Rate: 0.3%
- Data Citation Rate: 94%
- Agent Consensus Rate: 89%
```

---

## Implementation Roadmap

### Phase 1: Basic Integration (1-2 weeks)

**Objective:** Get basic evaluation working end-to-end

- [ ] Install `azure-ai-evaluation` package
- [ ] Create `evaluation/evaluator.py` with basic evaluators
- [ ] Integrate coherence, relevance, fluency evaluators
- [ ] Evaluate OrchestratorAgent's final report
- [ ] Store evaluation scores in SQLite database
- [ ] Add `--evaluate` flag to CLI
- [ ] Display basic scores after analysis

**Deliverable:** `python main.py analyze AAPL --evaluate` shows quality scores

---

### Phase 2: Agent-Specific Evaluation (2-3 weeks)

**Objective:** Evaluate each agent individually with custom metrics

- [ ] Evaluate each agent's output separately
- [ ] Create `evaluation/custom_evaluators.py`
- [ ] Implement Devil's Advocate effectiveness evaluator
- [ ] Implement data citation rate evaluator
- [ ] Implement multi-agent coherence evaluator
- [ ] Create `evaluation/metrics_tracker.py` for time-series tracking
- [ ] Build basic evaluation dashboard (Streamlit page)
- [ ] Add evaluation history view to CLI

**Deliverable:** Per-agent quality scores with custom metrics tracked over time

---

### Phase 3: Continuous Monitoring & Automation (2-3 weeks)

**Objective:** Production-ready evaluation pipeline

- [ ] Auto-evaluate every analysis run (make it default)
- [ ] Implement quality regression detection
- [ ] Add alert system for quality drops
- [ ] Create A/B testing framework (`evaluation/ab_testing.py`)
- [ ] Build comprehensive evaluation dashboard
  - Quality trends over time
  - Agent performance comparison
  - Regression alerts
  - A/B test results visualization
- [ ] Integration with Azure AI Foundry portal for cloud evaluation
- [ ] Document evaluation methodology for compliance

**Deliverable:** Full evaluation pipeline with monitoring dashboard and A/B testing

---

## CLI Commands

```bash
# Run analysis with evaluation
python main.py analyze AAPL --evaluate

# Establish evaluation baseline
python main.py eval-baseline --tickers AAPL,MSFT,NVDA --runs 10

# Compare current vs baseline
python main.py eval-compare --tickers AAPL,MSFT,NVDA --runs 10

# View evaluation history
python main.py eval-history --agent TechnicalAgent --days 30

# A/B test prompts
python main.py eval-ab-test \
    --agent DevilsAdvocate \
    --prompt-a prompts/devils_advocate_v1.txt \
    --prompt-b prompts/devils_advocate_v2.txt \
    --tickers AAPL,MSFT,GOOGL \
    --runs 20

# View evaluation dashboard
python main.py eval-dashboard  # Opens Streamlit dashboard

# Export evaluation report
python main.py eval-export --format json --output evaluations.json
```

---

## Expected Outputs

### Terminal Output Example

```
✨ Analysis Complete for AAPL

📊 Evaluation Results:

   Agent Performance:
   ┌─────────────────┬───────────┬──────────────┬───────────┐
   │ Agent           │ Coherence │ Groundedness │ Relevance │
   ├─────────────────┼───────────┼──────────────┼───────────┤
   │ DataAgent       │    N/A    │     N/A      │ 100% ✅   │
   │ TechnicalAgent  │  4.3/5 ⭐ │   4.1/5 ⭐   │ 4.5/5 ⭐  │
   │ SentimentAgent  │  4.5/5 ⭐ │   4.0/5 ⭐   │ 4.6/5 ⭐  │
   │ Fundamentals    │  4.4/5 ⭐ │   4.3/5 ⭐   │ 4.4/5 ⭐  │
   │ DevilsAdvocate  │  4.0/5 ⭐ │   3.8/5 ⭐   │ 4.2/5 ⭐  │
   └─────────────────┴───────────┴──────────────┴───────────┘

   Custom Metrics:
   • Devil's Advocate Effectiveness: 4.0/5 (identified 7 unique risks)
   • Data Citation Rate: 92% (38/41 claims cited)
   • Multi-Agent Coherence: 4.1/5 (1 minor conflict detected)

   Overall Report Quality: 4.2/5 ⭐⭐⭐⭐

   ⚠️  Warnings:
   - TechnicalAgent: 2 claims without data citation
   - SentimentAgent: Retrieved 4 news articles but only cited 2
   - Minor conflict: Technical says "overbought" but Fundamentals says "undervalued"
```

### Dashboard Visualization

The Streamlit dashboard would show:
- **Quality trends** - Line charts of coherence/groundedness/relevance over time
- **Agent comparison** - Bar charts comparing agent performance
- **Regression alerts** - Table of agents with declining scores
- **A/B test results** - Side-by-side comparison of prompt variations
- **Citation analysis** - Heatmap of data citation rates per agent
- **Conflict detection** - Network graph showing agent agreement/disagreement

---

## Database Schema

```sql
-- evaluations.db

CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    overall_score REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE agent_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER REFERENCES evaluations(id),
    agent_name TEXT NOT NULL,
    coherence_score REAL,
    groundedness_score REAL,
    relevance_score REAL,
    fluency_score REAL,
    custom_metrics JSON,
    warnings TEXT
);

CREATE TABLE ab_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_name TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    prompt_a TEXT,
    prompt_b TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ab_test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id INTEGER REFERENCES ab_tests(id),
    prompt_version TEXT,  -- 'a' or 'b'
    ticker TEXT,
    avg_coherence REAL,
    avg_groundedness REAL,
    avg_relevance REAL,
    custom_score REAL
);
```

---

## Benefits Summary

### For Development
- **Quantitative feedback** on code changes
- **Regression detection** when modifying prompts or agents
- **Systematic prompt engineering** via A/B testing
- **Debugging aid** - identify which agent is underperforming

### For Users
- **Trust indicators** - quality scores build confidence
- **Transparency** - see which claims are data-backed
- **Consistency** - ensure reports maintain quality over time

### For Compliance
- **Auditability** - quantitative evidence of AI quality
- **Safety metrics** - detect hallucinations and overconfidence
- **Documentation** - evaluation methodology for regulators

### For Production
- **Monitoring** - track quality in real-time
- **Alerting** - get notified of quality degradation
- **Optimization** - data-driven agent improvements

---

## Resources

- [Azure AI Evaluation SDK - PyPI](https://pypi.org/project/azure-ai-evaluation/)
- [Microsoft Learn - Evaluate with SDK](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)
- [Microsoft Learn - Cloud Evaluation](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/cloud-evaluation)
- [Continuous Evaluation Framework Blog](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/evaluating-generative-ai-models-using-microsoft-foundry%E2%80%99s-continuous-evaluation-/4468075)

---

## Next Steps

When ready to implement:

1. Review this roadmap with stakeholders
2. Prioritize which phases to implement
3. Set up evaluation infrastructure (database, dashboard)
4. Start with Phase 1 basic integration
5. Iterate based on insights from evaluation data

---

*Document created: 2026-04-05*
*Last updated: 2026-04-05*
*Status: Planning/Design Document*
