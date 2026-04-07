import os
import sys
import json
import threading
import queue
import asyncio
from contextlib import asynccontextmanager

from typing import Optional
from fastapi import FastAPI, HTTPException, Body, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from memory.long_term import LongTermMemory
from agents.orchestrator import OrchestratorAgent
from agents.chat_agent import ChatAgent
from ui.auth import require_create_permission, require_delete_permission, get_current_user
from ui.email_service import email_service, generate_verification_code, store_verification_code, verify_code
from ui.jwt_service import jwt_service

# Define a Thread-Safe stdout interceptor
class ThreadSafeStdout:
    def __init__(self, original_stdout):
        self._original = original_stdout
        self._queues = {} # thread_id -> queue.Queue

    def write(self, text):
        tid = threading.get_ident()
        if tid in self._queues:
            # We are intercepting this thread
            if text and text != '\n': # don't send purely empty newlines if not needed, or just send all
               self._queues[tid].put(text)
        else:
            self._original.write(text)

    def flush(self):
        tid = threading.get_ident()
        if tid not in self._queues:
            self._original.flush()

    def add_queue(self, q: queue.Queue):
        self._queues[threading.get_ident()] = q

    def remove_queue(self):
        tid = threading.get_ident()
        if tid in self._queues:
            del self._queues[tid]

# Override sys.stdout globally once
_original_stdout = sys.stdout
sys.stdout = ThreadSafeStdout(_original_stdout)

# Global memory instance
memory = None

# Store chat sessions (in production, use proper session management)
chat_sessions = {}


# Request/Response models for authentication
class SendCodeRequest(BaseModel):
    email: str


class SendCodeResponse(BaseModel):
    success: bool
    message: str


class VerifyCodeRequest(BaseModel):
    email: str
    code: str


class VerifyCodeResponse(BaseModel):
    success: bool
    token: str | None = None
    email: str | None = None
    message: str


# Request/Response models for chat
class ChatRequest(BaseModel):
    message: str
    ticker: str | None = None
    doc_id: str | None = None
    session_id: str | None = None
    web_search: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str
    context_used: bool
    web_search_used: bool
    timestamp: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory
    print("Loading LongTermMemory...")
    memory = LongTermMemory()
    yield
    print("Shutting down UI API...")
    sys.stdout = _original_stdout

app = FastAPI(title="StockSquad UI API", lifespan=lifespan)

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5180", "http://127.0.0.1:5180",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """Health check endpoint - no authentication required."""
    return {"status": "healthy", "service": "StockSquad API"}


@app.post("/api/auth/send-code", response_model=SendCodeResponse)
async def send_verification_code(request: SendCodeRequest):
    """
    Send a verification code to the user's email.

    This is step 1 of the login process.
    """
    try:
        email = request.email.lower().strip()

        # Validate email format
        if "@" not in email or "." not in email:
            raise HTTPException(status_code=400, detail="Invalid email format")

        # Generate and store verification code
        code = generate_verification_code()
        store_verification_code(email, code, expires_in_minutes=10)

        # Send email
        sent = await email_service.send_verification_code(email, code)

        if not sent:
            raise HTTPException(
                status_code=500,
                detail="Failed to send verification email. Please try again."
            )

        return SendCodeResponse(
            success=True,
            message=f"Verification code sent to {email}. Check your inbox!"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error sending verification code: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/auth/verify-code", response_model=VerifyCodeResponse)
async def verify_verification_code(request: VerifyCodeRequest):
    """
    Verify the code and return a JWT token.

    This is step 2 of the login process.
    """
    try:
        email = request.email.lower().strip()
        code = request.code.strip()

        # Check magic code for development
        is_magic_code = code == "123456" and os.getenv("SKIP_AUTH", "false").lower() == "true"

        # Verify the code (skip if magic code is used in dev mode)
        if not is_magic_code and not verify_code(email, code):
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired verification code"
            )

        # Check if user is authorized in Permit.io
        from ui.auth import permit_auth
        
        # Skip permit check if SKIP_AUTH is enabled
        if os.getenv("SKIP_AUTH", "false").lower() == "true":
            is_authorized = True
        else:
            is_authorized = await permit_auth.check_permission(email, "read", "analysis")

        if not is_authorized:
            raise HTTPException(
                status_code=403,
                detail=f"Email {email} is not authorized. Contact your administrator to get access."
            )

        # Create JWT token
        token = jwt_service.create_access_token(email)

        return VerifyCodeResponse(
            success=True,
            token=token,
            email=email,
            message="Login successful!"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error verifying code: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/reports")
async def get_reports(user_id: str = Depends(get_current_user)):
    if not memory:
        raise HTTPException(status_code=500, detail="Memory not initialized")
    try:
        results = memory.collection.get()
        if not results["ids"]:
            return []
        reports = []
        for doc_id, summary, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
            reports.append({
                "id": doc_id,
                "summary": summary,
                "metadata": metadata
            })
        reports.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
        return reports
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import math

def _sanitize_floats(obj):
    """Recursively replace NaN/Infinity with None so JSON serialization works."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj

@app.get("/api/reports/{doc_id}")
async def get_report_detail(doc_id: str, user_id: str = Depends(get_current_user)):
    if not memory:
        raise HTTPException(status_code=500, detail="Memory not initialized")
    try:
        analysis = memory.get_analysis_by_id(doc_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return _sanitize_floats(analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/{doc_id}/date-insights")
async def get_date_insights(doc_id: str, date: str, user_id: str = Depends(get_current_user)):
    """
    Get insights for a specific date in the analysis.

    Args:
        doc_id: The analysis document ID
        date: Date in YYYY-MM-DD format

    Returns:
        Dictionary with news, sentiment, price movements, and context for that date
    """
    if not memory:
        raise HTTPException(status_code=500, detail="Memory not initialized")

    try:
        from datetime import datetime, timedelta

        # Get the full analysis
        analysis = memory.get_analysis_by_id(doc_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Parse the requested date
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        # Extract data from analysis
        full_analysis = analysis.get("full_analysis", {})
        ticker = analysis.get("ticker", "N/A")

        # Get price data for the specific date
        price_history = full_analysis.get("data_collection", {}).get("stock_data", {}).get("price_history", {})
        price_data = price_history.get("data", [])

        # Find the price point for this date
        date_price_info = None
        prev_price_info = None

        for i, point in enumerate(price_data):
            point_date = datetime.fromisoformat(point["Date"].replace("Z", "+00:00"))
            if point_date.date() == target_date.date():
                date_price_info = point
                if i > 0:
                    prev_price_info = price_data[i - 1]
                break

        if not date_price_info:
            raise HTTPException(status_code=404, detail=f"No data found for date {date}")

        # Calculate price movement
        close_price = date_price_info.get(f"{ticker}_Close", date_price_info.get("Close", 0))
        open_price = date_price_info.get("Open", close_price)
        high_price = date_price_info.get("High", close_price)
        low_price = date_price_info.get("Low", close_price)
        volume = date_price_info.get("Volume", 0)

        day_change = close_price - open_price
        day_change_pct = (day_change / open_price * 100) if open_price else 0

        # Calculate change from previous day
        prev_close = prev_price_info.get(f"{ticker}_Close", prev_price_info.get("Close", close_price)) if prev_price_info else close_price
        prev_day_change = close_price - prev_close
        prev_day_change_pct = (prev_day_change / prev_close * 100) if prev_close else 0

        # Get news around this date (±3 days window)
        recent_news = full_analysis.get("data_collection", {}).get("stock_data", {}).get("recent_news", [])
        date_range_start = target_date - timedelta(days=3)
        date_range_end = target_date + timedelta(days=3)

        relevant_news = []
        for article in recent_news:
            if article.get("published"):
                try:
                    pub_date = datetime.fromisoformat(article["published"].replace("Z", "+00:00"))
                    if date_range_start <= pub_date <= date_range_end:
                        relevant_news.append(article)
                except:
                    continue

        # Sort news by date (closest first)
        relevant_news.sort(key=lambda x: abs((datetime.fromisoformat(x["published"].replace("Z", "+00:00")) - target_date).total_seconds()))

        # Get social sentiment if available
        social_analysis = full_analysis.get("social_media_analysis", {})
        overall_sentiment = social_analysis.get("sentiment_analysis", {}).get("overall", {}).get("sentiment", "N/A")

        # Calculate volume context (compare to average)
        avg_volume = price_history.get("average_volume", 0)
        volume_vs_avg = ((volume - avg_volume) / avg_volume * 100) if avg_volume else 0

        # Determine if this was a significant day
        is_significant = abs(day_change_pct) > 3 or abs(volume_vs_avg) > 50

        # Build response
        insights = {
            "date": date,
            "ticker": ticker,
            "price_info": {
                "close": round(close_price, 2),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "volume": int(volume),
                "day_change": round(day_change, 2),
                "day_change_percent": round(day_change_pct, 2),
                "prev_day_change": round(prev_day_change, 2),
                "prev_day_change_percent": round(prev_day_change_pct, 2),
                "volume_vs_average": round(volume_vs_avg, 2),
            },
            "news": relevant_news[:5],  # Top 5 most relevant
            "sentiment": {
                "overall": overall_sentiment,
            },
            "context": {
                "is_significant_day": is_significant,
                "period_high": price_history.get("period_high", 0),
                "period_low": price_history.get("period_low", 0),
                "average_volume": avg_volume,
            },
            "analysis_summary": _generate_date_summary(
                ticker, date, day_change_pct, volume_vs_avg, len(relevant_news), is_significant
            )
        }

        return insights

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _generate_date_summary(ticker: str, date: str, day_change_pct: float, volume_vs_avg: float, news_count: int, is_significant: bool) -> str:
    """Generate a brief summary for the date."""
    direction = "up" if day_change_pct > 0 else "down"
    magnitude = "significantly" if abs(day_change_pct) > 3 else "slightly" if abs(day_change_pct) > 1 else "marginally"

    summary = f"On {date}, {ticker} moved {magnitude} {direction} by {abs(day_change_pct):.2f}%. "

    if abs(volume_vs_avg) > 50:
        volume_dir = "higher" if volume_vs_avg > 0 else "lower"
        summary += f"Trading volume was {abs(volume_vs_avg):.0f}% {volume_dir} than average. "

    if news_count > 0:
        summary += f"There were {news_count} news articles around this time. "
    else:
        summary += "No major news events were detected around this date. "

    if is_significant:
        summary += "This was a significant trading day."

    return summary

@app.get("/api/analyze/stream")
async def analyze_stream(
    ticker: str,
    period: str = "1y",
    token: str = None,  # Accept JWT token from query param for EventSource compatibility
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
):
    """
    Kicks off an analysis and streams the 'print' stdout logs as Server-Sent Events,
    ending with the final document id and result.

    Note: EventSource doesn't support custom headers, so token can be passed as query parameter.
    """
    # Get user_id from token (query param), Authorization header, or X-User-Id header
    user_id = None

    # Try query parameter token first (for EventSource)
    if token:
        user_id = jwt_service.verify_token(token)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
    # Try Authorization header
    elif authorization and authorization.startswith("Bearer "):
        token_from_header = authorization.replace("Bearer ", "")
        user_id = jwt_service.verify_token(token_from_header)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
    # Fall back to X-User-Id header
    elif x_user_id:
        user_id = x_user_id
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide token query parameter or Authorization header."
        )

    # Check permissions
    from ui.auth import permit_auth
    await permit_auth.require_permission(user_id, "create", "analysis")
    def execution_generator():
        log_queue = queue.Queue()
        # Create a thread to run the orchestrator
        result_container = {}
        
        def run_agent():
            # Intercept stdout for this thread
            sys.stdout.add_queue(log_queue)
            try:
                orchestrator = OrchestratorAgent()
                result = orchestrator.analyze_stock(ticker, period)
                result_container['data'] = result
            except Exception as e:
                result_container['error'] = str(e)
            finally:
                sys.stdout.remove_queue()
                log_queue.put(None) # EOF marker
                
        agent_thread = threading.Thread(target=run_agent)
        agent_thread.start()

        # Yield logs from queue
        while True:
            try:
                # wait with timeout to allow client disconnect check
                line = log_queue.get(timeout=0.1)
                if line is None:
                    break
                # Only yield non-empty lines cleanly formatted
                cleaned = line.strip()
                if cleaned:
                    # SSE format: data: <payload>\n\n
                    payload = json.dumps({"type": "log", "message": cleaned})
                    yield f"data: {payload}\n\n"
            except queue.Empty:
                continue

        agent_thread.join()

        # Yield final result
        if 'error' in result_container:
            payload = json.dumps({"type": "error", "message": result_container['error']}, default=str)
            yield f"data: {payload}\n\n"
        else:
            payload = json.dumps({"type": "complete", "result": result_container['data']}, default=str)
            yield f"data: {payload}\n\n"

    return StreamingResponse(execution_generator(), media_type="text/event-stream")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user_id: str = Depends(require_create_permission)
):
    """
    Chat with the ChatAgent about analysis reports.

    Supports:
    - Asking questions about specific analyses (by doc_id or ticker)
    - Multi-turn conversations (via session_id)
    - Optional web search for additional context
    """
    try:
        # Get or create chat session
        session_id = request.session_id or f"session_{len(chat_sessions)}"

        if session_id not in chat_sessions:
            chat_sessions[session_id] = ChatAgent(web_search_enabled=request.web_search)

        chat_agent = chat_sessions[session_id]

        # Process the message
        result = chat_agent.chat(
            user_message=request.message,
            ticker=request.ticker,
            doc_id=request.doc_id,
            include_web_search=request.web_search
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("response"))

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            context_used=result.get("context_used", False),
            web_search_used=result.get("web_search_used", False),
            timestamp=result["timestamp"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/{session_id}")
async def clear_chat_session(
    session_id: str,
    user_id: str = Depends(require_delete_permission)
):
    """Clear a chat session and clean up resources."""
    if session_id in chat_sessions:
        try:
            chat_sessions[session_id].cleanup()
            del chat_sessions[session_id]
            return {"message": "Chat session cleared"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Session not found"}


@app.get("/api/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get the conversation history for a chat session."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        history = chat_sessions[session_id].get_conversation_history()
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


ui_dist_path = os.path.join(os.path.dirname(__file__), "web", "dist")
if os.path.exists(ui_dist_path):
    app.mount("/", StaticFiles(directory=ui_dist_path, html=True), name="web")
