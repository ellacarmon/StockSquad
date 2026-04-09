"""
OrchestratorAgent: Manages the workflow and coordinates between agents.
Assigns tasks, collects results, and synthesizes final analysis reports.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from agents.data_agent import DataAgent
from agents.technical_agent import TechnicalAgent
from agents.sentiment_agent import SentimentAgent
from agents.social_media_agent import SocialMediaAgent
from agents.social_sentiment_agent import SocialSentimentAgent
from agents.fundamentals_agent import FundamentalsAgent
from agents.devils_advocate import DevilsAdvocateAgent
from ml.training.data_collector import HistoricalDataCollector
from skills import register_all_skills
from skills.registry import SkillsRegistry

# Ensure skills are registered at module import time
if not SkillsRegistry.list_skills():
    print("[orchestrator.py] Registering skills at import time...")
    register_all_skills()


class OrchestratorAgent:
    """
    Orchestrator agent that manages the analysis workflow.

    This agent:
    - Coordinates tasks between specialized agents
    - Maintains session memory
    - Retrieves and references past analyses
    - Synthesizes final reports
    - Stores results in long-term memory
    """

    AGENT_NAME = "OrchestratorAgent"
    AGENT_INSTRUCTIONS = """You are the OrchestratorAgent for StockSquad, responsible for coordinating the full agent squad.

Your responsibilities:
1. Manage the multi-agent analysis workflow
2. Synthesize findings from ALL agents:
   - DataAgent (market data & company info)
   - TechnicalAgent (charts, indicators, ML signals)
   - SentimentAgent (news analysis & narrative trends)
   - SocialMediaAgent (X/Twitter & Reddit retail sentiment)
   - FundamentalsAgent (financial ratios & valuation)
   - DevilsAdvocateAgent (risk challenges & counter-arguments)
3. Reconcile conflicting viewpoints between agents
4. Reference past analyses from long-term memory
5. Produce a comprehensive, balanced final report

Analysis Integration:
1. Review each agent's perspective
2. Identify consensus and disagreements
3. Weigh the strength of each argument
4. Consider the Devil's Advocate challenges seriously
5. Synthesize into a coherent investment thesis

Final Report Structure:
- Executive Summary (3-4 sentences covering all perspectives)
- Company Overview
- Technical Analysis Summary (trend, signals, ML score)
- Fundamental Analysis Summary (valuation, financials, health)
- Sentiment Analysis Summary (news themes, market mood)
- Risk Assessment (Devil's Advocate challenges)
- Synthesis (how do all perspectives align or conflict?)
- Investment Thesis (bull case vs bear case)
- Final Recommendation (with confidence level and rationale)
- Key Risks to Monitor

Provide balanced, multi-dimensional analysis. Acknowledge uncertainty."""

    def __init__(self):
        """Initialize the OrchestratorAgent."""
        # Ensure skills are registered before creating any agents
        if not SkillsRegistry.list_skills():
            print("[OrchestratorAgent] Skills not registered, registering now...")
            register_all_skills()

        self.settings = get_settings()
        self.client = self._initialize_client()
        self.assistant = None
        self.long_term_memory = LongTermMemory()

    def _initialize_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client with DefaultAzureCredential."""
        # Get token provider for Azure OpenAI scope
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )

        return AzureOpenAI(
            api_version=self.settings.azure_openai_api_version,
            azure_endpoint=self.settings.azure_openai_endpoint,
            azure_ad_token_provider=token_provider,
        )

    def create_assistant(self):
        """Create the Azure OpenAI assistant if not already created."""
        if self.assistant is None:
            self.assistant = self.client.beta.assistants.create(
                model=self.settings.azure_openai_deployment_name,
                name=self.AGENT_NAME,
                instructions=self.AGENT_INSTRUCTIONS,
            )
        return self.assistant

    def analyze_stock(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Orchestrate a complete stock analysis.

        Args:
            ticker: Stock ticker symbol
            period: Historical data period

        Returns:
            Dictionary containing complete analysis results
        """
        ticker = ticker.upper()
        start_time = datetime.now()

        # Initialize short-term memory for this session
        memory = ShortTermMemory(ticker=ticker)

        # Log orchestration start
        memory.add_message(
            agent=self.AGENT_NAME,
            role="system",
            content=f"Starting analysis for {ticker}"
        )

        print(f"\n[{self.AGENT_NAME}] Analyzing {ticker}...")

        # Step 1: Check long-term memory for past analyses
        print(f"[{self.AGENT_NAME}] Checking for past analyses...")
        past_analyses = self.long_term_memory.retrieve_past_analyses(
            ticker=ticker,
            limit=3
        )

        past_context = ""
        if past_analyses:
            print(f"[{self.AGENT_NAME}] Found {len(past_analyses)} past analysis(es)")
            past_summaries = []
            for i, analysis in enumerate(past_analyses, 1):
                date = analysis.get("timestamp", "Unknown date")
                summary = analysis.get("summary", "No summary")
                past_summaries.append(f"  {i}. {date[:10]}: {summary[:100]}...")

            past_context = "\n\nPast Analyses:\n" + "\n".join(past_summaries)
            memory.add_message(
                agent=self.AGENT_NAME,
                role="system",
                content=f"Found {len(past_analyses)} past analyses",
                metadata={"past_analyses": past_analyses}
            )
        else:
            print(f"[{self.AGENT_NAME}] No past analyses found")
            past_context = "\n\nThis is the first analysis of this ticker."

        # Step 2: Assign data collection to DataAgent
        print(f"[{self.AGENT_NAME}] Requesting data collection from DataAgent...")
        data_agent = DataAgent(memory=memory)

        try:
            data_result = data_agent.analyze_ticker(ticker, period)
            print(f"[{self.AGENT_NAME}] Data collection complete")

            memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content="DataAgent has completed data collection",
                metadata={"data_agent_result": data_result}
            )

            # Get stock data from scratchpad
            stock_data = memory.get_from_scratchpad(f"{ticker}_data")

            # Validate we have usable data
            if stock_data and stock_data.get("error"):
                error_msg = f"❌ Cannot analyze {ticker}: {stock_data['error']}"
                print(f"[{self.AGENT_NAME}] {error_msg}")

                return {
                    "success": False,
                    "error": stock_data["error"],
                    "message": stock_data.get("message", "Ticker data unavailable"),
                    "ticker": ticker,
                    "report": f"# Analysis Failed\n\n{error_msg}\n\n{stock_data.get('message', '')}\n\nPlease verify the ticker symbol is correct and currently trading."
                }

            if not stock_data or not stock_data.get("price_history"):
                error_msg = f"❌ No price data available for {ticker}"
                print(f"[{self.AGENT_NAME}] {error_msg}")

                return {
                    "success": False,
                    "error": "No price data",
                    "message": "Unable to fetch price history for this ticker",
                    "ticker": ticker,
                    "report": f"# Analysis Failed\n\n{error_msg}\n\nThe ticker may be invalid, delisted, or not supported by data providers."
                }

        except Exception as e:
            error_msg = f"Data collection failed: {str(e)}"
            print(f"[{self.AGENT_NAME}] {error_msg}")
            memory.add_message(
                agent=self.AGENT_NAME,
                role="system",
                content=error_msg
            )
            return {
                "success": False,
                "error": error_msg,
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }
        finally:
            data_agent.cleanup()

        # Step 3: Run SentimentAgent analysis first so ML models can consume it
        sentiment_result = None
        if stock_data:
            try:
                print(f"[{self.AGENT_NAME}] Running sentiment analysis...")
                news_articles = stock_data.get("recent_news", [])

                sentiment_agent = SentimentAgent(memory=memory)
                sentiment_result = sentiment_agent.analyze_sentiment(ticker, news_articles)
                sentiment_agent.cleanup()

                # Persist a dated numeric snapshot for future ML retraining.
                try:
                    latest_price_row = stock_data.get("price_history", {}).get("data", [])[-1]
                    snapshot_date = str(latest_price_row.get("Date", ""))[:10] if latest_price_row else datetime.now().strftime("%Y-%m-%d")
                    HistoricalDataCollector().store_sentiment_snapshot(
                        ticker=ticker,
                        date=snapshot_date,
                        sentiment_result=sentiment_result,
                        source="sentiment_agent_live"
                    )
                except Exception as persist_error:
                    print(f"[{self.AGENT_NAME}] Warning: failed to persist sentiment snapshot: {persist_error}")

                memory.add_message(
                    agent=self.AGENT_NAME,
                    role="assistant",
                    content="SentimentAgent analysis complete"
                )
            except Exception as e:
                print(f"[{self.AGENT_NAME}] Sentiment analysis failed: {e}")
                memory.add_message(
                    agent=self.AGENT_NAME,
                    role="system",
                    content=f"Sentiment analysis error: {str(e)}"
                )

        # Step 4: Run TechnicalAgent analysis
        technical_result = None
        if stock_data:
            try:
                print(f"[{self.AGENT_NAME}] Running technical analysis...")
                import pandas as pd

                # Convert price history to DataFrame
                price_history = stock_data.get("price_history", {}).get("data", [])
                if price_history:
                    df = pd.DataFrame(price_history)
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)

                    # Use ensemble_unanimous model - our best performer
                    technical_agent = TechnicalAgent(memory=memory, model_type="ensemble_unanimous")
                    technical_result = technical_agent.analyze_technicals(
                        ticker,
                        df,
                        sentiment_result=sentiment_result
                    )
                    technical_agent.cleanup()

                    memory.add_message(
                        agent=self.AGENT_NAME,
                        role="assistant",
                        content="TechnicalAgent analysis complete"
                    )
            except Exception as e:
                print(f"[{self.AGENT_NAME}] Technical analysis failed: {e}")
                memory.add_message(
                    agent=self.AGENT_NAME,
                    role="system",
                    content=f"Technical analysis error: {str(e)}"
                )

        # Step 5: Run SocialMediaAgent analysis (Grok-based)
        social_media_result = None
        try:
            print(f"[{self.AGENT_NAME}] Running social media analysis...")

            social_media_agent = SocialMediaAgent(memory=memory)
            social_media_result = social_media_agent.analyze_social_sentiment(ticker)
            social_media_agent.cleanup()

            memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content="SocialMediaAgent analysis complete"
            )
        except Exception as e:
            print(f"[{self.AGENT_NAME}] Social media analysis failed: {e}")
            memory.add_message(
                agent=self.AGENT_NAME,
                role="system",
                content=f"Social media analysis error: {str(e)}"
            )

        # Step 5b: Run SocialSentimentAgent analysis (xpoz.ai multi-platform)
        social_sentiment_result = None
        try:
            print(f"[{self.AGENT_NAME}] Running xpoz.ai social sentiment analysis...")

            social_sentiment_agent = SocialSentimentAgent(memory=memory)
            social_sentiment_result = social_sentiment_agent.analyze_sentiment(ticker, days_back=7)
            social_sentiment_agent.cleanup()

            memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content="SocialSentimentAgent analysis complete"
            )
        except Exception as e:
            print(f"[{self.AGENT_NAME}] xpoz.ai social sentiment analysis failed: {e}")
            memory.add_message(
                agent=self.AGENT_NAME,
                role="system",
                content=f"xpoz.ai social sentiment error: {str(e)}"
            )

        # Step 6: Run FundamentalsAgent analysis
        fundamentals_result = None
        if stock_data:
            try:
                print(f"[{self.AGENT_NAME}] Running fundamental analysis...")
                financials = stock_data.get("financials", {})

                fundamentals_agent = FundamentalsAgent(memory=memory)
                fundamentals_result = fundamentals_agent.analyze_fundamentals(ticker, financials)
                fundamentals_agent.cleanup()

                memory.add_message(
                    agent=self.AGENT_NAME,
                    role="assistant",
                    content="FundamentalsAgent analysis complete"
                )
            except Exception as e:
                print(f"[{self.AGENT_NAME}] Fundamental analysis failed: {e}")
                memory.add_message(
                    agent=self.AGENT_NAME,
                    role="system",
                    content=f"Fundamental analysis error: {str(e)}"
                )

        # Step 7: Run DevilsAdvocateAgent to challenge consensus
        devils_advocate_result = None
        try:
            print(f"[{self.AGENT_NAME}] Running Devil's Advocate challenge...")
            devils_advocate = DevilsAdvocateAgent(memory=memory)
            devils_advocate_result = devils_advocate.challenge_consensus(
                ticker=ticker,
                data_result=data_result,
                technical_result=technical_result,
                sentiment_result=sentiment_result,
                fundamentals_result=fundamentals_result
            )
            devils_advocate.cleanup()

            memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content="DevilsAdvocateAgent challenge complete"
            )
        except Exception as e:
            print(f"[{self.AGENT_NAME}] Devil's Advocate failed: {e}")
            memory.add_message(
                agent=self.AGENT_NAME,
                role="system",
                content=f"Devil's Advocate error: {str(e)}"
            )

        # Step 8: Synthesize all findings
        print(f"[{self.AGENT_NAME}] Synthesizing final report from all agents...")

        # Create assistant and thread for synthesis
        assistant = self.create_assistant()
        thread = self.client.beta.threads.create()

        # Prepare comprehensive synthesis prompt
        agent_reports = []

        agent_reports.append(f"**DataAgent Report:**\n{data_result.get('response', 'N/A')}")

        if technical_result and technical_result.get('report'):
            tech_report = technical_result.get('report', '')
            signal = technical_result.get('signal_score', {})

            if signal and signal.get('direction'):
                agent_reports.append(
                    f"**TechnicalAgent Report:**\n"
                    f"Signal: {signal.get('direction', 'N/A')} | "
                    f"Confidence: {signal.get('confidence', 0)}/100 | "
                    f"Recommendation: {signal.get('recommendation', 'N/A')}\n\n"
                    f"{tech_report}"
                )
            else:
                agent_reports.append(f"**TechnicalAgent Report:**\n{tech_report}")

        if sentiment_result and sentiment_result.get('report'):
            agent_reports.append(f"**SentimentAgent Report:**\n{sentiment_result.get('report')}")

        if social_media_result and social_media_result.get('report'):
            social_report = social_media_result.get('report', '')
            sentiment_data = social_media_result.get('sentiment_analysis', {})
            overall = sentiment_data.get('overall', {})

            if overall and overall.get('sentiment'):
                agent_reports.append(
                    f"**SocialMediaAgent Report (Grok-based):**\n"
                    f"Retail Sentiment: {overall.get('sentiment')} "
                    f"({overall.get('stats', {}).get('bullish_pct', 0):.1f}% Bullish, "
                    f"{overall.get('stats', {}).get('bearish_pct', 0):.1f}% Bearish)\n\n"
                    f"{social_report}"
                )
            else:
                agent_reports.append(f"**SocialMediaAgent Report (Grok-based):**\n{social_report}")

        if social_sentiment_result and social_sentiment_result.get('report'):
            xpoz_report = social_sentiment_result.get('report', '')
            xpoz_sentiment = social_sentiment_result.get('structured_sentiment', {})
            overall_xpoz = xpoz_sentiment.get('overall_sentiment', {})
            post_count = social_sentiment_result.get('post_count', 0)

            if overall_xpoz and overall_xpoz.get('direction'):
                agent_reports.append(
                    f"**SocialSentimentAgent Report (xpoz.ai - Multi-Platform):**\n"
                    f"Overall Sentiment: {overall_xpoz.get('direction', 'N/A').upper()} "
                    f"(Score: {overall_xpoz.get('score', 50)}/100, "
                    f"Confidence: {overall_xpoz.get('confidence', 0)}%)\n"
                    f"Total Posts Analyzed: {post_count} across Twitter/X, Reddit, TikTok, Instagram\n\n"
                    f"{xpoz_report}"
                )
            else:
                agent_reports.append(f"**SocialSentimentAgent Report (xpoz.ai):**\n{xpoz_report}")

        if fundamentals_result and fundamentals_result.get('report'):
            fund_report = fundamentals_result.get('report', '')
            score = fundamentals_result.get('fundamental_score', {})

            if score and score.get('fundamental_score') is not None:
                agent_reports.append(
                    f"**FundamentalsAgent Report:**\n"
                    f"Score: {score.get('fundamental_score')}/100 | "
                    f"Rating: {score.get('rating', 'N/A')}\n\n"
                    f"{fund_report}"
                )
            else:
                agent_reports.append(f"**FundamentalsAgent Report:**\n{fund_report}")

        if devils_advocate_result and devils_advocate_result.get('report'):
            consensus = devils_advocate_result.get('consensus', '')
            report = devils_advocate_result.get('report', '')

            if consensus:
                agent_reports.append(
                    f"**DevilsAdvocateAgent Challenge:**\n"
                    f"Consensus Detected: {consensus}\n\n"
                    f"{report}"
                )
            else:
                agent_reports.append(f"**DevilsAdvocateAgent Challenge:**\n{report}")

        all_reports = "\n\n" + "="*80 + "\n\n".join(agent_reports)

        synthesis_prompt = f"""Please synthesize the following comprehensive analysis for {ticker} from our agent squad.

{past_context}

AGENT SQUAD REPORTS:
{all_reports}

Your task is to produce a final, integrated investment research report that:

1. **Executive Summary** (3-4 sentences capturing all perspectives)
2. **Company Overview** (from DataAgent findings)
3. **Technical Analysis Summary** (trend, momentum, ML signal score)
4. **Fundamental Analysis Summary** (valuation, profitability, growth, health)
5. **News Sentiment Analysis** (institutional perspective, news themes from SentimentAgent)
6. **Social Media Sentiment** (retail investor mood):
   - Grok-based analysis (SocialMediaAgent - X/Twitter & Reddit)
   - Multi-platform analysis (SocialSentimentAgent - xpoz.ai: Twitter, Reddit, TikTok, Instagram)
7. **Risk Assessment** (Devil's Advocate challenges and concerns)
8. **Multi-Agent Synthesis** (how do perspectives align or conflict?)
9. **Investment Thesis**:
   - Bull Case (strongest arguments for investing)
   - Bear Case (strongest arguments against)
10. **Final Recommendation** (BUY/HOLD/SELL with confidence level)
11. **Key Risks to Monitor**
12. **Comparison to Past Analyses** (if available)

Integrate all agent perspectives. Pay special attention to comparing the two social sentiment sources (Grok vs xpoz.ai).
Acknowledge where agents disagree. Be balanced and objective.
Provide clear, actionable guidance for investors."""

        # Send message to orchestrator assistant
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=synthesis_prompt
        )

        # Run the assistant
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        # Wait for completion with timeout
        from agents.assistant_utils import wait_for_run_completion, AssistantTimeoutError
        final_report = None
        try:
            run = wait_for_run_completion(
                client=self.client,
                thread_id=thread.id,
                run_id=run.id,
                timeout=300  # 5 minutes max (increased for complex synthesis with xpoz data)
            )

            # Get the final report
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            assistant_messages = [
                msg for msg in messages.data
                if msg.role == "assistant" and msg.created_at > message.created_at
            ]

            if assistant_messages:
                final_report = assistant_messages[0].content[0].text.value
            else:
                final_report = None

        except AssistantTimeoutError as e:
            print(f"[{self.AGENT_NAME}] Synthesis timeout: {e}")
            print(f"[{self.AGENT_NAME}] Generating fallback report with agent outputs...")
            final_report = None

        # Generate fallback report if synthesis failed
        if not final_report:
            print(f"[{self.AGENT_NAME}] Using fallback report format (synthesis unavailable)")
            fallback_sections = [
                f"# Stock Analysis Report: {ticker}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "**Note**: Automated synthesis timed out. Below are the individual agent reports.",
                "",
                "---",
                ""
            ]
            fallback_sections.extend(agent_reports)
            final_report = "\n".join(fallback_sections)

        print(f"[{self.AGENT_NAME}] Report synthesis complete")

        # Add to memory
        memory.add_message(
            agent=self.AGENT_NAME,
            role="assistant",
            content=final_report,
            metadata={"type": "final_report"}
        )

        # Step 4: Store in long-term memory
        print(f"[{self.AGENT_NAME}] Storing analysis in long-term memory...")

        # Create summary for embedding
        summary = f"Stock analysis for {ticker} on {datetime.now().strftime('%Y-%m-%d')}. "
        if stock_data:
            info = stock_data.get("info", {})
            price_history = stock_data.get("price_history", {})
            summary += f"{info.get('name', ticker)} - {info.get('sector', 'Unknown sector')}. "
            summary += f"Price: ${info.get('current_price', 0):.2f}, "
            summary += f"Change: {price_history.get('price_change_percent', 0):.2f}%"

        # Store full analysis
        agents_involved = [self.AGENT_NAME, DataAgent.AGENT_NAME]
        if technical_result:
            agents_involved.append(TechnicalAgent.AGENT_NAME)
        if sentiment_result:
            agents_involved.append(SentimentAgent.AGENT_NAME)
        if social_media_result:
            agents_involved.append(SocialMediaAgent.AGENT_NAME)
        if social_sentiment_result:
            agents_involved.append(SocialSentimentAgent.AGENT_NAME)
        if fundamentals_result:
            agents_involved.append(FundamentalsAgent.AGENT_NAME)
        if devils_advocate_result:
            agents_involved.append(DevilsAdvocateAgent.AGENT_NAME)

        full_analysis = {
            "ticker": ticker,
            "period": period,
            "data_collection": data_result,
            "technical_analysis": technical_result,
            "sentiment_analysis": sentiment_result,
            "social_media_analysis": social_media_result,
            "social_sentiment_analysis": social_sentiment_result,
            "fundamental_analysis": fundamentals_result,
            "devils_advocate": devils_advocate_result,
            "final_report": final_report,
            "session_data": memory.to_dict(),
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "agents_involved": agents_involved,
        }

        try:
            doc_id = self.long_term_memory.store_analysis(
                ticker=ticker,
                analysis_summary=summary,
                full_analysis=full_analysis,
                metadata={
                    "period": period,
                    "agents_involved": agents_involved,
                }
            )
            print(f"[{self.AGENT_NAME}] Stored as document: {doc_id}")
        except Exception as e:
            print(f"[{self.AGENT_NAME}] Warning: Failed to store in long-term memory: {e}")
            doc_id = None

        # Return complete results
        return {
            "success": True,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "final_report": final_report,
            "document_id": doc_id,
            "execution_time_seconds": (datetime.now() - start_time).total_seconds(),
            "past_analyses_found": len(past_analyses),
            "agents_involved": agents_involved,
            "data_collected": stock_data is not None,
            "technical_analysis_completed": technical_result is not None,
            "sentiment_analysis_completed": sentiment_result is not None,
            "social_media_analysis_completed": social_media_result is not None,
            "social_sentiment_analysis_completed": social_sentiment_result is not None,
            "fundamental_analysis_completed": fundamentals_result is not None,
            "devils_advocate_completed": devils_advocate_result is not None,
            "thread_id": thread.id,
            "session_summary": memory.to_dict(),
        }

    def get_past_analyses(self, ticker: str, limit: int = 5) -> list:
        """
        Retrieve past analyses for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of analyses to retrieve

        Returns:
            List of past analyses
        """
        return self.long_term_memory.retrieve_past_analyses(ticker, limit)

    def cleanup(self):
        """Clean up assistant resources."""
        if self.assistant:
            try:
                self.client.beta.assistants.delete(self.assistant.id)
                self.assistant = None
            except Exception as e:
                print(f"Warning: Failed to delete assistant: {e}")
