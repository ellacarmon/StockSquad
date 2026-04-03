import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { format, parseISO } from 'date-fns';
import {
  LineChart as LineChartIcon, LayoutDashboard, Loader2, FileText,
  AlertCircle, Moon, Sun, Plus, Terminal, Play, MessageCircle, Send, X, Globe,
  PanelRightClose, PanelRightOpen
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceDot, ReferenceLine
} from 'recharts';
import globalEventsData from './globalEvents.json';

export default function App() {
  const [reports, setReports] = useState([]);
  const [selectedReportId, setSelectedReportId] = useState(null);
  const [reportDetails, setReportDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [error, setError] = useState(null);

  // Theme State
  const [theme, setTheme] = useState(localStorage.getItem('stocksquad-theme') || 'dark');

  // Analysis State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showAnalysisForm, setShowAnalysisForm] = useState(false);
  const [tickerInput, setTickerInput] = useState('');
  const [periodInput, setPeriodInput] = useState('1y');
  const [logs, setLogs] = useState([]);

  // Chat State
  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatSessionId, setChatSessionId] = useState(null);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);

  // Date Insights State
  const [selectedDate, setSelectedDate] = useState(null);
  const [dateInsights, setDateInsights] = useState(null);
  const [loadingInsights, setLoadingInsights] = useState(false);
  const [insightsError, setInsightsError] = useState(null);

  // Chart Legend Toggle State
  const [hiddenSeries, setHiddenSeries] = useState(new Set());

  // Sidebar grouping state
  const [expandedTickers, setExpandedTickers] = useState(new Set());

  // Global events toggle
  const [showGlobalEvents, setShowGlobalEvents] = useState(false);
  const [hiddenEventCategories, setHiddenEventCategories] = useState(new Set());

  const toggleEventCategory = (category) => {
    setHiddenEventCategories(prev => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  const consoleEndRef = useRef(null);
  const chatEndRef = useRef(null);

  const getPillStyle = (signal) => {
    if (!signal) return { color: 'var(--text-muted)', background: 'transparent' };
    const s = signal.toLowerCase();
    
    let color = 'var(--text-main)';
    if (s.includes('bull') || s.includes('buy') || s.includes('po') || s.includes('strong')) color = '#10b981'; // emerald
    else if (s.includes('bear') || s.includes('sell') || s.includes('neg') || s.includes('weak')) color = '#ef4444'; // red
    else color = '#f59e0b'; // amber

    return { 
      color: color,
      fontSize: '1.1rem',
      fontWeight: 700,
      letterSpacing: '0.5px'
    };
  };

  const API_URL = import.meta.env.DEV ? 'http://127.0.0.1:8000' : '';

  useEffect(() => {
    fetchReports();
  }, []);

  useEffect(() => {
    document.documentElement.className = theme === 'light' ? 'light-theme' : '';
    localStorage.setItem('stocksquad-theme', theme);
  }, [theme]);

  useEffect(() => {
    if (selectedReportId) {
      setShowAnalysisForm(false);
      fetchReportDetails(selectedReportId);

      // Auto-expand the ticker group for the selected report
      const selectedReport = reports.find(r => r.id === selectedReportId);
      if (selectedReport && selectedReport.metadata.ticker) {
        setExpandedTickers(prev => new Set([...prev, selectedReport.metadata.ticker]));
      }
    }
  }, [selectedReportId, reports]);

  useEffect(() => {
    if (consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages]);

  useEffect(() => {
    // Reset chat state and chart toggles when report changes
    if (selectedReportId) {
      setChatMessages([]);
      setChatSessionId(null);
      // Don't force-close chat on report switch — keep user's preference
      setHiddenSeries(new Set());
    }
  }, [selectedReportId]);

  const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');

  const fetchReports = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/api/reports`);
      if (!response.ok) throw new Error('Failed to fetch reports');
      const data = await response.json();
      setReports(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchReportDetails = async (id) => {
    try {
      setLoadingDetails(true);
      const response = await fetch(`${API_URL}/api/reports/${id}`);
      if (!response.ok) throw new Error('Failed to fetch report details');
      const data = await response.json();
      setReportDetails(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingDetails(false);
    }
  };

  const startAnalysis = (e) => {
    e.preventDefault();
    if (!tickerInput) return;

    setIsAnalyzing(true);
    setLogs([`> Starting Squad Analysis for ${tickerInput.toUpperCase()}...`]);
    
    // Connect to Server Sent Events
    const sseUrl = `${API_URL}/api/analyze/stream?ticker=${tickerInput.toUpperCase()}&period=${periodInput}`;
    const eventSource = new EventSource(sseUrl);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'log') {
          setLogs(prev => [...prev, data.message]);
        } else if (data.type === 'error') {
          setLogs(prev => [...prev, `[ERROR] ${data.message}`]);
          eventSource.close();
          setIsAnalyzing(false);
        } else if (data.type === 'complete') {
          setLogs(prev => [...prev, `> Analysis finished successfully.`]);
          eventSource.close();
          setIsAnalyzing(false);
          
          // Refresh list and select new report
          fetchReports().then(() => {
            if (data.result && data.result.document_id) {
              setSelectedReportId(data.result.document_id);
            }
          });
        }
      } catch (err) {
        console.error('Failed to parse SSE data', err);
      }
    };

    eventSource.onerror = () => {
      setLogs(prev => [...prev, `[SYSTEM] Stream disconnected.`]);
      eventSource.close();
      setIsAnalyzing(false);
    };
  };

  const formatDate = (dateString) => {
    try {
      return format(parseISO(dateString), 'MMM d, yyyy • h:mm a');
    } catch (e) {
      return dateString;
    }
  };

  const sendChatMessage = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const userMessage = chatInput.trim();
    setChatInput('');

    // Add user message to chat
    setChatMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    }]);

    setChatLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          ticker: reportDetails?.ticker,
          doc_id: selectedReportId,
          session_id: chatSessionId,
          web_search: webSearchEnabled
        })
      });

      if (!response.ok) throw new Error('Chat request failed');

      const data = await response.json();

      // Store session ID for continuation
      if (!chatSessionId) {
        setChatSessionId(data.session_id);
      }

      // Add assistant response to chat
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        timestamp: data.timestamp,
        context_used: data.context_used,
        web_search_used: data.web_search_used
      }]);

    } catch (err) {
      setChatMessages(prev => [...prev, {
        role: 'error',
        content: `Failed to get response: ${err.message}`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  const clearChat = async () => {
    if (chatSessionId) {
      try {
        await fetch(`${API_URL}/api/chat/${chatSessionId}`, { method: 'DELETE' });
      } catch (err) {
        console.error('Failed to clear chat session:', err);
      }
    }
    setChatMessages([]);
    setChatSessionId(null);
  };

  const handleChartClick = async (data) => {
    if (!data || !data.activePayload || !data.activePayload[0]) return;

    const clickedData = data.activePayload[0].payload;
    const rawDate = clickedData.date;

    // Convert "MMM d, yyyy" back to YYYY-MM-DD
    const dateStr = rawDate; // e.g., "Jan 15, 2024"

    // Find the matching date in the raw data
    const rawData = reportDetails.full_analysis.data_collection.stock_data.price_history.data;
    const matchingPoint = rawData.find(item => {
      const itemDate = format(new Date(item.Date), 'MMM d, yyyy');
      return itemDate === dateStr;
    });

    if (!matchingPoint) return;

    // Extract the actual ISO date
    const isoDate = new Date(matchingPoint.Date).toISOString().split('T')[0];

    setSelectedDate(isoDate);
    setLoadingInsights(true);
    setInsightsError(null);

    try {
      const response = await fetch(`${API_URL}/api/reports/${selectedReportId}/date-insights?date=${isoDate}`);
      if (!response.ok) throw new Error('Failed to fetch date insights');
      const insights = await response.json();
      setDateInsights(insights);
    } catch (err) {
      setInsightsError(err.message);
      console.error('Error fetching date insights:', err);
    } finally {
      setLoadingInsights(false);
    }
  };

  const closeInsightsModal = () => {
    setSelectedDate(null);
    setDateInsights(null);
    setInsightsError(null);
  };

  const toggleTicker = (ticker) => {
    setExpandedTickers(prev => {
      const next = new Set(prev);
      if (next.has(ticker)) {
        next.delete(ticker);
      } else {
        next.add(ticker);
      }
      return next;
    });
  };

  const groupReportsByTicker = (reports) => {
    const grouped = {};
    reports.forEach(report => {
      const ticker = report.metadata.ticker || 'Unknown';
      if (!grouped[ticker]) {
        grouped[ticker] = [];
      }
      grouped[ticker].push(report);
    });
    return grouped;
  };

  const identifyImportantEvents = (earningsData, newsData, priceData) => {
    if (!priceData) return [];

    const events = [];

    console.log('🔍 Processing events...');
    console.log('Earnings dates:', earningsData?.length || 0);
    console.log('News items:', newsData?.length || 0);
    console.log('Price data points:', priceData.length);

    // Add earnings dates from yfinance
    if (earningsData && earningsData.length > 0) {
      console.log('Sample earnings date:', earningsData[0]);

      earningsData.forEach(earning => {
        if (!earning.date) return;

        // Parse the date and format for matching
        const earningDate = new Date(earning.date);
        const dateStr = format(earningDate, 'MMM d, yyyy');

        // Check if this date exists in our chart data
        const matchingDataPoint = priceData.find(p => {
          const pDate = format(new Date(p.Date), 'MMM d, yyyy');
          return pDate === dateStr;
        });

        if (matchingDataPoint) {
          console.log(`✅ Earnings date found in chart: ${dateStr}`);
          events.push({
            date: dateStr,
            type: 'earnings',
            icon: '📊',
            title: 'Earnings Report',
            upcoming: earning.upcoming || false
          });
        } else {
          console.log(`⚠️ Earnings date not in chart range: ${dateStr}`);
        }
      });
    }

    // Also look for important events in news (acquisitions, FDA approvals, etc.)
    if (newsData && newsData.length > 0) {
      const importantKeywords = [
        'acquisition', 'merger', 'acquire',
        'fda approval', 'fda',
        'lawsuit', 'investigation',
        'ceo', 'chief executive',
        'dividend increase', 'special dividend',
        'stock split', 'split',
        'partnership', 'strategic deal',
        'breakthrough', 'unveils'
      ];

      newsData.forEach(article => {
        if (!article.published || !article.title) return;

        const title = article.title.toLowerCase();
        const pubDate = new Date(article.published);
        const dateStr = format(pubDate, 'MMM d, yyyy');

        // Check if important event
        const isImportant = importantKeywords.some(kw => title.includes(kw));

        if (isImportant) {
          const matchingDataPoint = priceData.find(p => {
            const pDate = format(new Date(p.Date), 'MMM d, yyyy');
            return pDate === dateStr;
          });

          if (matchingDataPoint) {
            // Don't duplicate if we already have an earnings marker for this date
            const alreadyMarked = events.some(e => e.date === dateStr);
            if (!alreadyMarked) {
              console.log(`✅ Important event: ${dateStr} - ${article.title}`);
              events.push({
                date: dateStr,
                type: 'important',
                icon: '⚡',
                title: article.title,
                publisher: article.publisher,
                link: article.link
              });
            }
          }
        }
      });
    }

    console.log(`📊 Total events to display: ${events.length}`);
    return events;
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-section">
            <LineChartIcon className="w-6 h-6 text-blue-500" />
            <h1>StockSquad</h1>
          </div>
          <button onClick={toggleTheme} className="icon-btn" title="Toggle Theme">
            {theme === 'dark' ? <Sun size={20}/> : <Moon size={20}/>}
          </button>
        </div>
        
        <button 
          className="new-analysis-btn"
          onClick={() => {
            setSelectedReportId(null);
            setShowAnalysisForm(true);
          }}
        >
          <Plus size={20} /> Run Analysis
        </button>

        <div className="reports-list">
          {loading && (
            <div className="loading-spinner">
              <Loader2 className="w-6 h-6 animate-spin" />
            </div>
          )}

          {!loading && reports.map((report, index) => {
            const isSelected = selectedReportId === report.id;
            const currentTicker = report.metadata.ticker;
            const prevTicker = index > 0 ? reports[index - 1].metadata.ticker : null;
            const showTicker = currentTicker !== prevTicker;

            return (
              <div key={report.id}>
                {/* Show ticker header only when it changes */}
                {showTicker && (
                  <div style={{
                    padding: '6px 16px',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    color: 'var(--accent-blue)',
                    marginTop: index === 0 ? '0' : '12px',
                    marginBottom: '4px',
                  }}>
                    {currentTicker}
                  </div>
                )}
                <div
                  className={`report-item ${isSelected ? 'active' : ''}`}
                  onClick={() => setSelectedReportId(report.id)}
                >
                  <div className="report-item-header">
                    <span className="report-date">{formatDate(report.metadata.timestamp)}</span>
                  </div>
                  <div className="report-summary">
                    {report.summary}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {!selectedReportId && !showAnalysisForm && (
          <div className="hero-empty">
            <LayoutDashboard size={48} strokeWidth={1.5} />
            <h2>Welcome to StockSquad</h2>
            <p>Select a report from the sidebar or run a new analysis.</p>
          </div>
        )}

        {showAnalysisForm && (
          <div className="analysis-form-container animate-fade-in">
            <form className="analysis-form" onSubmit={startAnalysis}>
              <h2><Terminal className="inline mr-2 text-blue-500" /> New Analysis Request</h2>
              <p>Deploy the AI agent squad to analyze a stock ticker.</p>

              <div className="form-group">
                <label>Stock Ticker Symbol</label>
                <input 
                  type="text" 
                  className="form-input" 
                  placeholder="e.g. AAPL, META, TSLA" 
                  value={tickerInput}
                  onChange={(e) => setTickerInput(e.target.value)}
                  disabled={isAnalyzing}
                  required
                />
              </div>

              <div className="form-group">
                <label>Historical Period</label>
                <select 
                  className="form-input"
                  value={periodInput}
                  onChange={(e) => setPeriodInput(e.target.value)}
                  disabled={isAnalyzing}
                >
                  <option value="1mo">1 Month</option>
                  <option value="3mo">3 Months</option>
                  <option value="6mo">6 Months</option>
                  <option value="1y">1 Year</option>
                  <option value="5y">5 Years</option>
                </select>
              </div>

              <button 
                type="submit" 
                className="new-analysis-btn" 
                style={{ width: '100%', margin: '16px 0 0 0' }}
                disabled={isAnalyzing || !tickerInput}
              >
                {isAnalyzing ? (
                  <><Loader2 className="animate-spin" size={20} /> Deploying Squad...</>
                ) : (
                  <><Play size={20} /> Run Complete Analysis</>
                )}
              </button>
            </form>

            {(isAnalyzing || logs.length > 0) && (
              <div className="console-view">
                {logs.map((log, index) => (
                  <div key={index} className={`console-line ${log.startsWith('[ERROR]') ? 'error' : ''}`}>
                    {log}
                  </div>
                ))}
                {isAnalyzing && (
                  <div className="console-line flex items-center gap-2">
                    <Loader2 size={12} className="animate-spin" /> ...
                  </div>
                )}
                <div ref={consoleEndRef} />
              </div>
            )}
          </div>
        )}

        {selectedReportId && loadingDetails && (
          <div className="hero-empty">
            <Loader2 size={48} className="animate-spin text-blue-500" />
            <p>Loading deep analysis...</p>
          </div>
        )}

        {selectedReportId && !loadingDetails && reportDetails && (
          <div className="report-view animate-fade-in">
            <div className="report-header">
              <div className="report-title">
                <h2>{reportDetails.ticker} Analysis Report</h2>
                <div className="report-meta">
                  <span className="flex items-center gap-1">
                    <FileText size={16} />
                    Generated {formatDate(reportDetails.timestamp)}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="px-4 py-2 bg-blue-500/10 border border-blue-500/20 rounded-full text-blue-500 font-medium text-sm">
                  AI Squad Confirmed
                </div>
              </div>
            </div>

            {/* Agent Consensus Summary */}
            {(() => {
              const fa = reportDetails?.full_analysis || {};

              // Deep search helper: recursively finds a value by key name
              const deepFind = (obj, targetKey) => {
                if (!obj || typeof obj !== 'object') return undefined;
                if (obj[targetKey] !== undefined && obj[targetKey] !== null) return obj[targetKey];
                for (const key of Object.keys(obj)) {
                  const found = deepFind(obj[key], targetKey);
                  if (found !== undefined) return found;
                }
                return undefined;
              };

              // --- Technical Signal ---
              const techAnalysis = fa.technical_analysis || {};
              let techSignal =
                techAnalysis?.signal_score?.direction ||
                techAnalysis?.signal_score?.recommendation ||
                techAnalysis?.overall_signal ||
                techAnalysis?.signal;

              // Fallback: search deeply in technical_analysis
              if (!techSignal) {
                techSignal = deepFind(techAnalysis, 'direction') ||
                             deepFind(techAnalysis, 'recommendation') ||
                             deepFind(techAnalysis, 'overall_signal');
              }

              // Last resort: parse from report text
              if (!techSignal && techAnalysis?.report) {
                const r = techAnalysis.report.toLowerCase();
                if (r.includes('strong buy') || r.includes('strongly bullish')) techSignal = 'STRONG BUY';
                else if (r.includes('bullish') || r.includes('buy')) techSignal = 'BULLISH';
                else if (r.includes('strong sell') || r.includes('strongly bearish')) techSignal = 'STRONG SELL';
                else if (r.includes('bearish') || r.includes('sell')) techSignal = 'BEARISH';
                else if (r.includes('neutral') || r.includes('hold')) techSignal = 'NEUTRAL';
              }

              // --- Fundamental Health ---
              const fundAnalysis = fa.fundamental_analysis || {};
              let fundRating =
                fundAnalysis?.fundamental_score?.rating ||
                fundAnalysis?.rating;

              if (!fundRating) {
                fundRating = deepFind(fundAnalysis, 'rating') ||
                             deepFind(fundAnalysis, 'recommendation');
              }

              if (!fundRating && fundAnalysis?.report) {
                const r = fundAnalysis.report.toLowerCase();
                if (r.includes('strong buy')) fundRating = 'STRONG BUY';
                else if (r.includes('buy')) fundRating = 'BUY';
                else if (r.includes('sell')) fundRating = 'SELL';
                else if (r.includes('hold')) fundRating = 'HOLD';
              }

              // --- Social Sentiment ---
              const socialAnalysis = fa.social_media_analysis || {};
              let socialSentiment =
                socialAnalysis?.sentiment_analysis?.overall?.sentiment;

              if (!socialSentiment) {
                socialSentiment = deepFind(socialAnalysis, 'sentiment');
              }

              if (!socialSentiment && socialAnalysis?.report) {
                const r = socialAnalysis.report.toLowerCase();
                // Look for the explicit section
                const idx = r.indexOf('overall retail sentiment');
                const section = idx !== -1 ? r.substring(idx, idx + 200) : r;
                if (section.includes('bullish')) socialSentiment = 'Bullish';
                else if (section.includes('bearish')) socialSentiment = 'Bearish';
                else if (section.includes('neutral')) socialSentiment = 'Neutral';
              }

              return (
              <div style={{ 
                width: '100%', 
                marginBottom: 40, 
                padding: '24px 32px', 
                background: 'var(--bg-card)', 
                borderRadius: 16, 
                border: '1px solid var(--border-color)', 
                boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                display: 'flex',
                flexDirection: 'column',
                gap: 20
              }}>
                <h3 style={{ margin: 0, fontSize: '1.2rem', color: 'var(--text-main)', fontWeight: 600 }}>AI Squad Consensus</h3>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>Technical Signal</span>
                    <span style={getPillStyle(techSignal)}>
                      {techSignal || 'N/A'}
                    </span>
                  </div>
                  
                  <div style={{ width: 1, height: 40, background: 'var(--border-color)' }}></div>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center' }}>
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>Fundamental Health</span>
                    <span style={getPillStyle(fundRating)}>
                      {fundRating || 'N/A'}
                    </span>
                  </div>

                  <div style={{ width: 1, height: 40, background: 'var(--border-color)' }}></div>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-end' }}>
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>Social Sentiment</span>
                    <span style={getPillStyle(socialSentiment)}>
                      {socialSentiment || 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
              );
            })()}

            {reportDetails?.full_analysis?.data_collection?.stock_data?.price_history?.data && (() => {
              const rawData = reportDetails.full_analysis.data_collection.stock_data.price_history.data;
              const ticker = reportDetails.ticker;
              const newsData = reportDetails.full_analysis?.data_collection?.stock_data?.recent_news || [];
              const earningsData = reportDetails.full_analysis?.data_collection?.stock_data?.earnings_dates || [];

              // Identify important events (earnings + major news)
              const importantEvents = identifyImportantEvents(earningsData, newsData, rawData);

              // Check if new format (_Percent keys) exists
              const hasPercentKeys = rawData.length > 0 && Object.keys(rawData[0]).some(k => k.endsWith('_Percent'));

              let chartData;
              let tickerKeys;

              if (hasPercentKeys) {
                // New format: use _Percent keys directly
                chartData = rawData.map(item => {
                  const formatted = { date: format(new Date(item.Date), 'MMM d, yyyy') };
                  Object.keys(item).forEach(key => {
                    if (key.endsWith('_Percent')) {
                      formatted[key.replace('_Percent', '')] = parseFloat(item[key].toFixed(2));
                    }
                  });
                  return formatted;
                });
                tickerKeys = Object.keys(chartData[0] || {}).filter(k => k !== 'date');
              } else {
                // Legacy format: compute % change from Close or {TICKER}_Close
                const closeKey = rawData[0][`${ticker}_Close`] !== undefined ? `${ticker}_Close` : 'Close';
                const firstClose = rawData[0][closeKey];

                // Also check for any other {TICKER}_Close columns (peers)
                const allCloseKeys = Object.keys(rawData[0]).filter(k => k.endsWith('_Close'));
                const peerCloseKeys = allCloseKeys.length > 0 ? allCloseKeys : [closeKey];
                
                // Build first-price map for all tickers
                const firstPrices = {};
                if (allCloseKeys.length > 0) {
                  allCloseKeys.forEach(k => { firstPrices[k] = rawData[0][k]; });
                } else {
                  firstPrices[closeKey] = firstClose;
                }

                chartData = rawData.map(item => {
                  const formatted = { date: format(new Date(item.Date), 'MMM d, yyyy') };
                  
                  if (allCloseKeys.length > 0) {
                    // Multiple _Close columns (peers present but no _Percent)
                    allCloseKeys.forEach(k => {
                      const label = k.replace('_Close', '');
                      const fp = firstPrices[k];
                      if (fp && item[k] !== undefined) {
                        formatted[label] = parseFloat((((item[k] - fp) / fp) * 100).toFixed(2));
                      }
                    });
                  } else {
                    // Single Close column — use ticker name as label
                    if (firstClose && item[closeKey] !== undefined) {
                      formatted[ticker] = parseFloat((((item[closeKey] - firstClose) / firstClose) * 100).toFixed(2));
                    }
                  }
                  return formatted;
                });
                tickerKeys = Object.keys(chartData[0] || {}).filter(k => k !== 'date');
              }

              const colors = ['var(--accent-blue)', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

              // Filter global events that fall within the chart date range
              const chartStartDate = new Date(rawData[0].Date);
              const chartEndDate = new Date(rawData[rawData.length - 1].Date);

              // Helper: find the nearest trading day in chartData for a given date
              const findNearestChartDate = (targetDate) => {
                const target = new Date(targetDate).getTime();
                let bestMatch = null;
                let bestDist = Infinity;
                for (const point of chartData) {
                  // Parse the formatted date back to timestamp for comparison
                  const pointTime = new Date(point.date).getTime();
                  const dist = Math.abs(pointTime - target);
                  if (dist < bestDist) {
                    bestDist = dist;
                    bestMatch = point.date;
                  }
                }
                // Only snap if within 5 trading days (~7 calendar days)
                return bestDist <= 7 * 24 * 60 * 60 * 1000 ? bestMatch : null;
              };

              const relevantGlobalEvents = showGlobalEvents ? globalEventsData
                .filter(event => {
                  const eventDate = new Date(event.date);
                  return eventDate >= chartStartDate && eventDate <= chartEndDate;
                })
                .filter(event => !hiddenEventCategories.has(event.category))
                .map(event => {
                  const exactFormatted = format(new Date(event.date), 'MMM d, yyyy');
                  // Check if exact date exists in chart data
                  const exactMatch = chartData.some(d => d.date === exactFormatted);
                  const snappedDate = exactMatch
                    ? exactFormatted
                    : findNearestChartDate(event.date);
                  return { ...event, chartDate: snappedDate };
                })
                .filter(event => event.chartDate !== null)
              : [];

              // Debug logging
              if (showGlobalEvents) {
                console.log('🌍 Global Events Debug:');
                console.log('Chart range:', chartStartDate.toISOString().split('T')[0], 'to', chartEndDate.toISOString().split('T')[0]);
                console.log('Total global events in database:', globalEventsData.length);
                console.log('Events in range:', relevantGlobalEvents.length);
                if (relevantGlobalEvents.length > 0) {
                  console.log('Events to display:', relevantGlobalEvents.map(e => `${e.date}: ${e.title} → chart: ${e.chartDate}`));
                  console.log('Sample chart dates:', chartData.slice(0, 5).map(d => d.date));
                }
              }

              return (
              <div className="chart-container" style={{ width: '100%', height: 380, marginBottom: 40, padding: 24, background: 'var(--bg-card)', borderRadius: 16, border: '1px solid var(--border-color)', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: showGlobalEvents ? 8 : 16 }}>
                  <h3 style={{ margin: 0, fontSize: '1.2rem', color: 'var(--text-main)', fontWeight: 600 }}>Historical Price Action</h3>
                  <button
                    onClick={() => setShowGlobalEvents(!showGlobalEvents)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '6px 12px',
                      fontSize: '0.85rem',
                      backgroundColor: showGlobalEvents ? 'var(--accent-blue)' : 'var(--bg-main)',
                      color: showGlobalEvents ? 'white' : 'var(--text-muted)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                    }}
                    onMouseEnter={(e) => {
                      if (!showGlobalEvents) e.currentTarget.style.backgroundColor = 'var(--bg-card)';
                    }}
                    onMouseLeave={(e) => {
                      if (!showGlobalEvents) e.currentTarget.style.backgroundColor = 'var(--bg-main)';
                    }}
                  >
                    <Globe size={14} />
                    <span>Global Events</span>
                  </button>
                </div>
                {showGlobalEvents && (
                  <div style={{ display: 'flex', gap: '6px', marginBottom: 10, flexWrap: 'wrap' }}>
                    {[
                      { key: 'monetary', label: 'Fed / Monetary', color: '#f59e0b' },
                      { key: 'geopolitical', label: 'Geopolitical', color: '#ef4444' },
                      { key: 'economic', label: 'Economic', color: '#06b6d4' },
                      { key: 'political', label: 'Political', color: '#8b5cf6' },
                    ].map(cat => {
                      const isHidden = hiddenEventCategories.has(cat.key);
                      return (
                        <button
                          key={cat.key}
                          onClick={() => toggleEventCategory(cat.key)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '5px',
                            padding: '3px 10px',
                            fontSize: '0.75rem',
                            fontWeight: 500,
                            fontFamily: 'inherit',
                            backgroundColor: isHidden ? 'transparent' : `${cat.color}18`,
                            color: isHidden ? 'var(--text-muted)' : cat.color,
                            border: `1px solid ${isHidden ? 'var(--border-color)' : cat.color + '40'}`,
                            borderRadius: '20px',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            opacity: isHidden ? 0.5 : 1,
                            textDecoration: isHidden ? 'line-through' : 'none',
                          }}
                        >
                          <span style={{ width: 8, height: 8, borderRadius: '50%', background: isHidden ? 'var(--text-muted)' : cat.color, display: 'inline-block', opacity: isHidden ? 0.4 : 1 }}></span>
                          {cat.label}
                        </button>
                      );
                    })}
                  </div>
                )}
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 12 }}>
                  Click on any point to see what happened that day{tickerKeys.length > 1 ? ' · Click legend names to toggle' : ''}
                  {showGlobalEvents && relevantGlobalEvents.length > 0 && ` · ${relevantGlobalEvents.length} global events shown`}
                </p>
                <ResponsiveContainer width="100%" height="90%">
                  <LineChart data={chartData} onClick={handleChartClick} style={{ cursor: 'pointer' }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                    <XAxis 
                      dataKey="date" 
                      stroke="var(--text-muted)" 
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                      minTickGap={30}
                    />
                    <YAxis 
                      stroke="var(--text-muted)" 
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(value) => `${value}%`}
                      domain={['auto', 'auto']}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'var(--bg-main)', border: '1px solid var(--border-color)', borderRadius: 8, color: 'var(--text-main)', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }}
                      itemStyle={{ fontWeight: 600 }}
                      formatter={(value, name) => hiddenSeries.has(name) ? null : [`${value}%`, name]}
                    />
                    <Legend
                      onClick={(e) => {
                        const key = e.dataKey || e.value;
                        setHiddenSeries(prev => {
                          const next = new Set(prev);
                          if (next.has(key)) {
                            next.delete(key);
                          } else {
                            next.add(key);
                          }
                          return next;
                        });
                      }}
                      formatter={(value) => (
                        <span style={{
                          cursor: 'pointer',
                          color: hiddenSeries.has(value) ? 'var(--text-muted)' : 'var(--text-main)',
                          textDecoration: hiddenSeries.has(value) ? 'line-through' : 'none',
                          opacity: hiddenSeries.has(value) ? 0.5 : 1,
                          transition: 'all 0.2s ease',
                          userSelect: 'none',
                        }}>{value}</span>
                      )}
                    />
                    {tickerKeys.map((key, index) => (
                      <Line
                        key={key}
                        name={key}
                        type="monotone"
                        dataKey={key}
                        stroke={colors[index % colors.length]}
                        strokeWidth={key === ticker ? 4 : 2}
                        dot={false}
                        activeDot={{ r: 6 }}
                        hide={hiddenSeries.has(key)}
                      />
                    ))}
                    {/* Event Markers */}
                    {importantEvents.map((event, idx) => {
                      // Find the data point for this event
                      const dataPoint = chartData.find(d => d.date === event.date);
                      if (!dataPoint) return null;

                      // Use the main ticker's value for positioning
                      const yValue = dataPoint[tickerKeys[0]];
                      if (yValue === undefined) return null;

                      return (
                        <ReferenceDot
                          key={`event-${idx}`}
                          x={event.date}
                          y={yValue}
                          r={8}
                          fill={event.type === 'earnings' ? '#10b981' : '#f59e0b'}
                          stroke="#fff"
                          strokeWidth={2}
                          style={{ cursor: 'pointer' }}
                          label={{
                            value: event.icon,
                            position: 'top',
                            fontSize: 16,
                            offset: 10
                          }}
                        />
                      );
                    })}
                    {/* Global Event Lines */}
                    {relevantGlobalEvents.map((event, idx) => {
                      // Use the snapped chart date that we pre-computed
                      const xValue = event.chartDate;
                      if (!xValue) return null;

                      // Pick color based on category
                      const categoryColors = {
                        geopolitical: '#ef4444',
                        monetary: '#f59e0b',
                        political: '#8b5cf6',
                        economic: '#06b6d4',
                      };
                      const lineColor = categoryColors[event.category] || 'var(--text-muted)';

                      return (
                        <ReferenceLine
                          key={`global-${idx}`}
                          x={xValue}
                          stroke={lineColor}
                          strokeDasharray="6 4"
                          strokeWidth={1.5}
                          opacity={0.6}
                          label={{
                            value: `${event.title}`,
                            position: 'insideTopRight',
                            fill: lineColor,
                            fontSize: 10,
                            angle: -90,
                            offset: 10
                          }}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
                {/* Event Legend */}
                {(importantEvents.length > 0 || (showGlobalEvents && relevantGlobalEvents.length > 0)) && (
                  <div style={{ marginTop: '12px', display: 'flex', gap: '16px', fontSize: '0.85rem', color: 'var(--text-muted)', flexWrap: 'wrap', alignItems: 'center' }}>
                    {importantEvents.length > 0 && (
                      <>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ fontSize: '16px' }}>📊</span>
                          <span>Earnings</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ fontSize: '16px' }}>⚡</span>
                          <span>Important Events</span>
                        </div>
                      </>
                    )}
                    {showGlobalEvents && relevantGlobalEvents.length > 0 && (
                      <>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ width: 14, height: 2, background: '#ef4444', display: 'inline-block', borderRadius: 1 }}></span>
                          <span>Geopolitical</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ width: 14, height: 2, background: '#f59e0b', display: 'inline-block', borderRadius: 1 }}></span>
                          <span>Monetary</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ width: 14, height: 2, background: '#8b5cf6', display: 'inline-block', borderRadius: 1 }}></span>
                          <span>Political</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ width: 14, height: 2, background: '#06b6d4', display: 'inline-block', borderRadius: 1 }}></span>
                          <span>Economic</span>
                        </div>
                      </>
                    )}
                    <div style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                      {importantEvents.length + (showGlobalEvents ? relevantGlobalEvents.length : 0)} event{(importantEvents.length + (showGlobalEvents ? relevantGlobalEvents.length : 0)) !== 1 ? 's' : ''} marked
                    </div>
                  </div>
                )}
              </div>
              );
            })()}

            <div className="markdown-body">
              <ReactMarkdown>
                {reportDetails.full_analysis?.final_report ||
                 reportDetails.full_analysis?.data_collection?.response ||
                 "### No detailed markdown report available.\n\nRaw summary:\n" + reportDetails.summary}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </main>

      {/* Floating Chat Toggle Button */}
      {selectedReportId && reportDetails && !showChat && (
        <button
          className="chat-toggle-btn"
          onClick={() => setShowChat(true)}
          title="Open Report Chat"
        >
          <MessageCircle size={20} />
        </button>
      )}

      {/* Right Sidebar Chat */}
      {selectedReportId && reportDetails && showChat && (
        <aside className="chat-sidebar-right animate-slide-in-right">
          <div className="chat-header">
            <div className="chat-header-left">
              <MessageCircle size={18} />
              <h3>Report Chat</h3>
            </div>
            <div className="chat-header-right">
              <label className="chat-web-search-toggle">
                <input
                  type="checkbox"
                  checked={webSearchEnabled}
                  onChange={(e) => setWebSearchEnabled(e.target.checked)}
                />
                <span>Web</span>
              </label>
              <button onClick={clearChat} className="chat-action-btn" title="Clear chat">
                Clear
              </button>
              <button onClick={() => setShowChat(false)} className="chat-close-btn" title="Close chat">
                <X size={18} />
              </button>
            </div>
          </div>

          <div className="chat-messages">
            {chatMessages.length === 0 && (
              <div className="chat-empty">
                <MessageCircle size={32} style={{ marginBottom: 12, opacity: 0.3 }} />
                <p>Ask anything about this<br/>specific analysis!</p>
              </div>
            )}

            {chatMessages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.role}`}>
                <div className="chat-message-header">
                  <span className="chat-role">
                    {msg.role === 'user' ? 'You' : msg.role === 'error' ? 'Error' : 'AI'}
                  </span>
                  {msg.web_search_used && (
                    <span className="chat-badge">🌐 Web</span>
                  )}
                </div>
                <div className="chat-message-content">
                  {msg.role === 'assistant' ? (
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  ) : (
                    <p>{msg.content}</p>
                  )}
                </div>
              </div>
            ))}

            {chatLoading && (
              <div className="chat-message assistant">
                <div className="chat-message-header">
                  <span className="chat-role">AI</span>
                </div>
                <div className="chat-message-content">
                  <Loader2 className="w-4 h-4 animate-spin" />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <form onSubmit={sendChatMessage} className="chat-input-form">
            <div className="chat-input-wrapper">
              <input
                type="text"
                placeholder="Ask follow-up questions..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={chatLoading}
              />
              <button
                type="submit"
                className="chat-send-btn"
                disabled={chatLoading || !chatInput.trim()}
              >
                <Send size={16} />
              </button>
            </div>
          </form>
        </aside>
      )}

      {/* Date Insights Modal */}
      {selectedDate && (
        <div
          className="modal-overlay"
          onClick={closeInsightsModal}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            backdropFilter: 'blur(4px)',
          }}
        >
          <div
            className="modal-content"
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: 'var(--bg-card)',
              borderRadius: '16px',
              border: '1px solid var(--border-color)',
              boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
              maxWidth: '700px',
              width: '90%',
              maxHeight: '80vh',
              overflow: 'auto',
              position: 'relative',
            }}
          >
            {/* Close Button */}
            <button
              onClick={closeInsightsModal}
              style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--text-muted)',
                padding: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: '8px',
                transition: 'all 0.2s',
              }}
              onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-main)'}
              onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
            >
              <X size={24} />
            </button>

            {loadingInsights && (
              <div style={{ padding: '60px', textAlign: 'center' }}>
                <Loader2 className="w-12 h-12 animate-spin" style={{ margin: '0 auto', color: 'var(--accent-blue)' }} />
                <p style={{ marginTop: '16px', color: 'var(--text-muted)' }}>Loading insights...</p>
              </div>
            )}

            {insightsError && (
              <div style={{ padding: '40px', textAlign: 'center' }}>
                <AlertCircle size={48} style={{ margin: '0 auto', color: '#ef4444' }} />
                <p style={{ marginTop: '16px', color: '#ef4444' }}>Error: {insightsError}</p>
              </div>
            )}

            {dateInsights && !loadingInsights && (
              <div style={{ padding: '32px' }}>
                {/* Header */}
                <div style={{ marginBottom: '24px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
                      {dateInsights.ticker} - {format(new Date(selectedDate), 'MMMM d, yyyy')}
                    </h2>
                    {dateInsights.news && dateInsights.news.length > 0 && dateInsights.news.some(n => {
                      const title = n.title.toLowerCase();
                      return title.includes('earnings') || title.includes('quarterly') || title.includes('q1') || title.includes('q2') || title.includes('q3') || title.includes('q4');
                    }) && (
                      <span style={{
                        padding: '4px 12px',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                        borderRadius: '6px',
                        fontSize: '0.85rem',
                        color: '#10b981',
                        fontWeight: 600
                      }}>
                        📊 Earnings
                      </span>
                    )}
                  </div>
                  <p style={{ fontSize: '0.95rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                    {dateInsights.analysis_summary}
                  </p>
                </div>

                {/* Price Movement */}
                <div style={{
                  padding: '20px',
                  backgroundColor: 'var(--bg-main)',
                  borderRadius: '12px',
                  marginBottom: '24px',
                  border: '1px solid var(--border-color)'
                }}>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '16px', color: 'var(--text-main)' }}>
                    Price Movement
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
                    <div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>Close Price</div>
                      <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--text-main)' }}>
                        ${dateInsights.price_info.close}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>Day Change</div>
                      <div style={{
                        fontSize: '1.3rem',
                        fontWeight: 700,
                        color: dateInsights.price_info.day_change_percent >= 0 ? '#10b981' : '#ef4444'
                      }}>
                        {dateInsights.price_info.day_change_percent >= 0 ? '+' : ''}
                        {dateInsights.price_info.day_change_percent}%
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>High / Low</div>
                      <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-main)' }}>
                        ${dateInsights.price_info.high} / ${dateInsights.price_info.low}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>Volume</div>
                      <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-main)' }}>
                        {(dateInsights.price_info.volume / 1000000).toFixed(2)}M
                        {dateInsights.price_info.volume_vs_average !== 0 && (
                          <span style={{
                            fontSize: '0.85rem',
                            marginLeft: '8px',
                            color: dateInsights.price_info.volume_vs_average > 0 ? '#10b981' : '#ef4444'
                          }}>
                            ({dateInsights.price_info.volume_vs_average > 0 ? '+' : ''}
                            {dateInsights.price_info.volume_vs_average.toFixed(0)}%)
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* News Section */}
                {dateInsights.news && dateInsights.news.length > 0 && (
                  <div style={{ marginBottom: '24px' }}>
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: 'var(--text-main)' }}>
                      News Around This Time ({dateInsights.news.length})
                    </h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      {dateInsights.news.map((article, index) => (
                        <a
                          key={index}
                          href={article.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{
                            padding: '12px 16px',
                            backgroundColor: 'var(--bg-main)',
                            borderRadius: '8px',
                            border: '1px solid var(--border-color)',
                            textDecoration: 'none',
                            color: 'var(--text-main)',
                            transition: 'all 0.2s',
                            display: 'block',
                          }}
                          onMouseOver={(e) => {
                            e.currentTarget.style.borderColor = 'var(--accent-blue)';
                            e.currentTarget.style.transform = 'translateX(4px)';
                          }}
                          onMouseOut={(e) => {
                            e.currentTarget.style.borderColor = 'var(--border-color)';
                            e.currentTarget.style.transform = 'translateX(0)';
                          }}
                        >
                          <div style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '4px', color: 'var(--text-main)' }}>
                            {article.title}
                          </div>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                            {article.publisher} • {article.published ? format(new Date(article.published), 'MMM d, h:mm a') : 'N/A'}
                          </div>
                        </a>
                      ))}
                    </div>
                  </div>
                )}

                {(!dateInsights.news || dateInsights.news.length === 0) && (
                  <div style={{
                    padding: '20px',
                    backgroundColor: 'var(--bg-main)',
                    borderRadius: '12px',
                    textAlign: 'center',
                    marginBottom: '24px',
                    border: '1px solid var(--border-color)'
                  }}>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                      No news articles found around this date
                    </p>
                  </div>
                )}

                {/* Context */}
                {dateInsights.context.is_significant_day && (
                  <div style={{
                    padding: '16px',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderRadius: '8px',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    marginTop: '16px'
                  }}>
                    <div style={{ fontSize: '0.9rem', color: 'var(--accent-blue)', fontWeight: 600 }}>
                      ⚡ Significant Trading Day
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
