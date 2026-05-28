import { useState, useEffect } from 'react';
import { getStocks, collectNews, getNews, getNewsSummary, getDailySentiment } from '../api/client';

const PERIOD_OPTIONS = [
    { value: '1d', label: '1 วัน' },
    { value: '7d', label: '1 สัปดาห์' },
    { value: '30d', label: '1 เดือน' },
    { value: 'all', label: 'ทั้งหมด' },
];

export default function NewsSentiment() {
    const [stocks, setStocks] = useState([]);
    const [selected, setSelected] = useState('');
    const [period, setPeriod] = useState('7d');
    const [news, setNews] = useState([]);
    const [summary, setSummary] = useState(null);
    const [dailySentiment, setDailySentiment] = useState([]);
    const [collecting, setCollecting] = useState(false);
    const [loading, setLoading] = useState(false);
    const [collectResult, setCollectResult] = useState(null);
    const [overallFromCollect, setOverallFromCollect] = useState(null);

    useEffect(() => {
        getStocks().then(r => {
            const list = r.data?.stocks || r.data || [];
            setStocks(list);
            if (list.length > 0) setSelected(list[0].symbol);
        }).catch(() => { });
    }, []);

    useEffect(() => {
        if (selected) loadNews();
    }, [selected, period]);

    const loadNews = async () => {
        setLoading(true);
        try {
            const [newsRes, summaryRes, dailyRes] = await Promise.all([
                getNews(selected, 50, period),
                getNewsSummary(selected, period),
                getDailySentiment(selected, period === '1d' ? 7 : period === '7d' ? 14 : 30),
            ]);
            setNews(newsRes.data?.news || []);
            setSummary(summaryRes.data || null);
            setDailySentiment(dailyRes.data?.daily_sentiment || []);
        } catch (err) {
            console.error('Failed to load news:', err);
        }
        setLoading(false);
    };

    const handleCollect = async () => {
        if (!selected || collecting) return;
        setCollecting(true);
        setCollectResult(null);
        setOverallFromCollect(null);
        try {
            const res = await collectNews(selected);
            const data = res.data;
            setCollectResult(`เก็บข่าว ${data.collected} รายการ, บันทึกใหม่ ${data.saved} รายการ`);
            setOverallFromCollect(data.overall || null);
            await loadNews();
        } catch (err) {
            setCollectResult('❌ เกิดข้อผิดพลาดในการเก็บข่าว');
        }
        setCollecting(false);
    };

    const sentimentBadge = (label, score) => {
        const cfg = {
            bullish: { bg: '#22c55e20', color: '#22c55e', icon: '🟢', text: 'Bullish' },
            bearish: { bg: '#ef444420', color: '#ef4444', icon: '🔴', text: 'Bearish' },
            neutral: { bg: '#eab30820', color: '#eab308', icon: '🟡', text: 'Neutral' },
        };
        const c = cfg[label] || cfg.neutral;
        return (
            <span style={{
                background: c.bg, color: c.color, padding: '3px 10px', borderRadius: 12,
                fontSize: 12, fontWeight: 600, whiteSpace: 'nowrap',
            }}>
                {c.icon} {c.text} ({score > 0 ? '+' : ''}{score?.toFixed(2)})
            </span>
        );
    };

    // Group stocks by market for the dropdown
    const groupedStocks = stocks.reduce((acc, s) => {
        const market = s.market || (s.symbol.endsWith('.BK') ? 'SET' : 'US');
        if (!acc[market]) acc[market] = [];
        acc[market].push(s);
        return acc;
    }, {});

    const overallData = overallFromCollect || summary;

    return (
        <div style={{ padding: '0 24px 24px' }}>
            <h2 className="page-title" style={{ color: 'var(--accent-cyan)' }}>📰 News & Sentiment</h2>
            <p className="page-subtitle" style={{ marginBottom: 16 }}>วิเคราะห์อารมณ์ตลาดจากข่าวสาร ด้วย AI</p>

            {/* Controls */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', gap: 16, alignItems: 'end', flexWrap: 'wrap' }}>
                    <div className="form-group" style={{ marginBottom: 0, minWidth: 220 }}>
                        <label className="form-label">เลือกหุ้น</label>
                        <select className="form-select" value={selected} onChange={e => setSelected(e.target.value)}>
                            {Object.entries(groupedStocks).map(([market, items]) => (
                                <optgroup key={market} label={`📊 ${market}`}>
                                    {items.map(s => (
                                        <option key={s.symbol} value={s.symbol}>
                                            {s.symbol} — {s.name}
                                        </option>
                                    ))}
                                </optgroup>
                            ))}
                        </select>
                    </div>
                    <div className="form-group" style={{ marginBottom: 0, minWidth: 140 }}>
                        <label className="form-label">ช่วงเวลา</label>
                        <div style={{ display: 'flex', gap: 4 }}>
                            {PERIOD_OPTIONS.map(opt => (
                                <button key={opt.value} onClick={() => setPeriod(opt.value)}
                                    style={{
                                        padding: '6px 12px', borderRadius: 8, border: 'none',
                                        fontSize: 12, fontWeight: 600, cursor: 'pointer',
                                        background: period === opt.value
                                            ? 'linear-gradient(135deg, #8b5cf6, #06b6d4)'
                                            : 'rgba(255,255,255,0.08)',
                                        color: period === opt.value ? '#fff' : 'var(--text-secondary)',
                                        transition: 'all 0.2s',
                                    }}>
                                    {opt.label}
                                </button>
                            ))}
                        </div>
                    </div>
                    <button className="btn btn-primary" onClick={handleCollect} disabled={collecting}
                        style={{ background: 'linear-gradient(135deg, #8b5cf6, #06b6d4)', border: 'none' }}>
                        {collecting ? '⏳ กำลังเก็บข่าว...' : '📥 Collect & Analyze News'}
                    </button>
                    <button className="btn btn-secondary" onClick={loadNews} disabled={loading}>
                        🔄 Refresh
                    </button>
                </div>
                {collectResult && (
                    <p style={{ marginTop: 12, fontSize: 13, color: 'var(--accent-cyan)' }}>{collectResult}</p>
                )}
            </div>

            {/* Overall Sentiment Card */}
            {overallData && (
                <div className="card" style={{ marginBottom: 20, borderLeft: '3px solid #8b5cf6' }}>
                    <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap' }}>
                        <div>
                            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Market Mood — {selected}</div>
                            <div style={{ fontSize: 36, fontWeight: 700 }}>
                                {overallData.overall_score > 0.15 ? '🟢' :
                                    overallData.overall_score < -0.15 ? '🔴' : '🟡'}
                                <span style={{
                                    marginLeft: 10, fontSize: 24,
                                    color: overallData.overall_score > 0.15 ? '#22c55e' :
                                        overallData.overall_score < -0.15 ? '#ef4444' : '#eab308',
                                }}>
                                    {(overallData.overall_label || 'neutral').toUpperCase()}
                                </span>
                            </div>
                        </div>
                        <div style={{ flex: 1 }}>
                            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Sentiment Score</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <div style={{ flex: 1, height: 12, background: 'rgba(255,255,255,0.1)', borderRadius: 6, overflow: 'hidden' }}>
                                    <div style={{
                                        width: `${((overallData.overall_score || 0) + 1) / 2 * 100}%`,
                                        height: '100%',
                                        background: 'linear-gradient(90deg, #ef4444, #eab308, #22c55e)',
                                        borderRadius: 6,
                                        transition: 'width 0.5s',
                                    }} />
                                </div>
                                <span style={{ fontSize: 14, fontWeight: 600, minWidth: 50 }}>
                                    {(overallData.overall_score || 0) > 0 ? '+' : ''}
                                    {(overallData.overall_score || 0).toFixed(2)}
                                </span>
                            </div>
                        </div>
                        {overallData.news_count !== undefined && (
                            <div style={{ textAlign: 'center' }}>
                                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>ข่าวทั้งหมด</div>
                                <div style={{ display: 'flex', gap: 12 }}>
                                    <div><span style={{ color: '#22c55e', fontWeight: 700 }}>{overallData.bullish_count || 0}</span> 🟢</div>
                                    <div><span style={{ color: '#eab308', fontWeight: 700 }}>{overallData.neutral_count || 0}</span> 🟡</div>
                                    <div><span style={{ color: '#ef4444', fontWeight: 700 }}>{overallData.bearish_count || 0}</span> 🔴</div>
                                </div>
                            </div>
                        )}
                    </div>
                    {overallData.overall_summary && (
                        <div style={{ marginTop: 16, padding: 12, background: 'rgba(139,92,246,0.08)', borderRadius: 8, fontSize: 14, lineHeight: 1.6 }}>
                            🤖 <b>AI สรุป:</b> {overallData.overall_summary}
                        </div>
                    )}
                </div>
            )}

            {/* Sentiment Trend */}
            {dailySentiment.length > 1 && (
                <div className="card" style={{ marginBottom: 20 }}>
                    <div className="card-header">
                        <span className="card-title">📈 Sentiment Trend (Daily)</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'end', gap: 3, height: 80, padding: '12px 0' }}>
                        {dailySentiment.slice().reverse().map((d, i) => {
                            const score = d.avg_sentiment || 0;
                            const pct = ((score + 1) / 2) * 100;
                            const color = score > 0.1 ? '#22c55e' : score < -0.1 ? '#ef4444' : '#eab308';
                            return (
                                <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}
                                    title={`${d.date}: ${score.toFixed(2)} (${d.news_count} news)`}>
                                    <div style={{
                                        width: '100%', height: `${Math.max(pct, 5)}%`, minHeight: 4,
                                        background: color, borderRadius: 2, opacity: 0.8,
                                    }} />
                                    {i % 5 === 0 && (
                                        <span style={{ fontSize: 8, color: 'var(--text-muted)', transform: 'rotate(-45deg)' }}>
                                            {d.date?.slice(5) || ''}
                                        </span>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
                        <span>Bearish (-1)</span>
                        <span>Neutral (0)</span>
                        <span>Bullish (+1)</span>
                    </div>
                </div>
            )}

            {/* News List */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">📋 ข่าวล่าสุด — {selected}</span>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                        {news.length} รายการ • {PERIOD_OPTIONS.find(p => p.value === period)?.label}
                    </span>
                </div>
                {loading && <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 20 }}>⏳ กำลังโหลด...</p>}
                {!loading && news.length === 0 && (
                    <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
                        <p style={{ fontSize: 32 }}>📰</p>
                        <p>ยังไม่มีข่าวในช่วงเวลานี้</p>
                        <p style={{ fontSize: 13 }}>กด <b>"Collect & Analyze News"</b> เพื่อดึงข่าวและวิเคราะห์ sentiment</p>
                    </div>
                )}
                <div style={{ maxHeight: 600, overflowY: 'auto' }}>
                    {news.map((item, i) => (
                        <div key={i} style={{
                            padding: '14px 0',
                            borderBottom: '1px solid rgba(255,255,255,0.05)',
                            display: 'flex', gap: 12, alignItems: 'flex-start',
                        }}>
                            <div style={{ minWidth: 85, textAlign: 'center', paddingTop: 2 }}>
                                {sentimentBadge(item.sentiment_label, item.sentiment_score)}
                            </div>
                            <div style={{ flex: 1 }}>
                                <a href={item.url} target="_blank" rel="noopener noreferrer"
                                    style={{ color: 'var(--text-primary)', textDecoration: 'none', fontWeight: 600, fontSize: 14, lineHeight: 1.4 }}>
                                    {item.title}
                                </a>
                                {item.summary_th && (
                                    <p style={{ margin: '4px 0 0', fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                                        {item.summary_th}
                                    </p>
                                )}
                                <div style={{ marginTop: 6, display: 'flex', gap: 12, fontSize: 11, color: 'var(--text-muted)', flexWrap: 'wrap' }}>
                                    <span>📡 {item.source}</span>
                                    <span>🕐 {item.published_at?.slice(0, 16)?.replace('T', ' ')}</span>
                                    {item.key_topics?.length > 0 && (
                                        <span>🏷️ {item.key_topics.join(', ')}</span>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
