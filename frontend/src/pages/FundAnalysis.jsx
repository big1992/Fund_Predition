import { useState, useEffect } from 'react';
import PlotChart from '../components/PlotChart';
import { getStocks, getStockPrices, getIndicators } from '../api/client';
import { getCurrencyInfo } from '../utils/currencyUtils';

export default function FundAnalysis() {
    const [stocks, setStocks] = useState([]);
    const [selected, setSelected] = useState('');
    const [prices, setPrices] = useState(null);
    const [indicators, setIndicators] = useState(null);
    const [loading, setLoading] = useState(false);
    const [showRSI, setShowRSI] = useState(true);
    const [showMACD, setShowMACD] = useState(true);
    const [showBB, setShowBB] = useState(false);
    const [showSMA, setShowSMA] = useState(true);

    useEffect(() => {
        getStocks().then(r => {
            setStocks(r.data);
            if (r.data.length > 0) setSelected(r.data[0].symbol);
        }).catch(() => { });
    }, []);

    useEffect(() => {
        if (!selected) return;
        setLoading(true);
        Promise.all([
            getStockPrices(selected).then(r => setPrices(r.data)).catch(() => { }),
            getIndicators(selected).then(r => setIndicators(r.data)).catch(() => { }),
        ]).finally(() => setLoading(false));
    }, [selected]);

    const p = prices?.prices || [];
    const ind = indicators?.indicators || [];
    const dates = p.map(x => x.date);

    // Current stats
    const latest = p.length > 0 ? p[p.length - 1] : null;
    const prev = p.length > 1 ? p[p.length - 2] : null;
    const change = latest && prev ? ((latest.close - prev.close) / prev.close * 100) : 0;
    const high52w = p.length > 0 ? Math.max(...p.slice(-252).map(x => x.high)) : 0;
    const low52w = p.length > 0 ? Math.min(...p.slice(-252).map(x => x.low)) : 0;

    const { symbol: currencySymbol, label: currencyLabel } = getCurrencyInfo(stocks, selected);

    const candlestickTraces = [
        {
            x: dates, open: p.map(x => x.open), high: p.map(x => x.high),
            low: p.map(x => x.low), close: p.map(x => x.close),
            type: 'candlestick', name: selected,
            increasing: { line: { color: '#10b981' } },
            decreasing: { line: { color: '#ef4444' } },
        },
    ];

    if (showSMA && ind.length > 0) {
        const d = ind.map(x => x.date);
        candlestickTraces.push(
            { x: d, y: ind.map(x => x.sma_20), type: 'scatter', mode: 'lines', name: 'SMA 20', line: { color: '#f59e0b', width: 1 } },
            { x: d, y: ind.map(x => x.sma_50), type: 'scatter', mode: 'lines', name: 'SMA 50', line: { color: '#8b5cf6', width: 1 } },
        );
    }

    if (showBB && ind.length > 0) {
        const d = ind.map(x => x.date);
        candlestickTraces.push(
            { x: d, y: ind.map(x => x.bb_upper), type: 'scatter', mode: 'lines', name: 'BB Upper', line: { color: '#06b6d4', width: 1, dash: 'dot' } },
            { x: d, y: ind.map(x => x.bb_lower), type: 'scatter', mode: 'lines', name: 'BB Lower', line: { color: '#06b6d4', width: 1, dash: 'dot' }, fill: 'tonexty', fillcolor: 'rgba(6,182,212,0.05)' },
        );
    }

    return (
        <div>
            <div className="page-header">
                <h2>Fund Analysis</h2>
                <p>วิเคราะห์หุ้นด้วย Technical Indicators แบบ Interactive</p>
            </div>

            {/* Controls */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', gap: 16, alignItems: 'end', flexWrap: 'wrap' }}>
                    <div className="form-group" style={{ marginBottom: 0, minWidth: 200 }}>
                        <label className="form-label">เลือกหุ้น</label>
                        <select className="form-select" value={selected} onChange={e => setSelected(e.target.value)}>
                            {stocks.map(s => <option key={s.symbol} value={s.symbol}>{s.symbol} — {s.name}</option>)}
                        </select>
                    </div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer' }}>
                        <input type="checkbox" checked={showSMA} onChange={() => setShowSMA(!showSMA)} /> SMA
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer' }}>
                        <input type="checkbox" checked={showBB} onChange={() => setShowBB(!showBB)} /> Bollinger
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer' }}>
                        <input type="checkbox" checked={showRSI} onChange={() => setShowRSI(!showRSI)} /> RSI
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-secondary)', cursor: 'pointer' }}>
                        <input type="checkbox" checked={showMACD} onChange={() => setShowMACD(!showMACD)} /> MACD
                    </label>
                </div>
            </div>

            {/* Stats */}
            {latest && (
                <div className="metrics-grid">
                    <div className="metric-card blue">
                        <div className="metric-label">ราคาปัจจุบัน</div>
                        <div className="metric-value">{currencySymbol}{latest.close.toFixed(2)}</div>
                        <div className={`metric-change ${change >= 0 ? 'positive' : 'negative'}`}>
                            {change >= 0 ? '▲' : '▼'} {Math.abs(change).toFixed(2)}%
                        </div>
                    </div>
                    <div className="metric-card green">
                        <div className="metric-label">Volume</div>
                        <div className="metric-value">{(latest.volume / 1e6).toFixed(1)}M</div>
                    </div>
                    <div className="metric-card purple">
                        <div className="metric-label">52W High</div>
                        <div className="metric-value">{currencySymbol}{high52w.toFixed(2)}</div>
                    </div>
                    <div className="metric-card red">
                        <div className="metric-label">52W Low</div>
                        <div className="metric-value">{currencySymbol}{low52w.toFixed(2)}</div>
                    </div>
                </div>
            )}

            {loading ? (
                <div className="loading-container"><div className="spinner" /><p>Loading chart...</p></div>
            ) : p.length === 0 ? (
                <div className="card"><p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 40 }}>ไม่มีข้อมูล — กรุณาดาวน์โหลดข้อมูลที่หน้า Dashboard ก่อน</p></div>
            ) : (
                <>
                    {/* Candlestick Chart */}
                    <div className="card" style={{ marginBottom: 20 }}>
                        <div className="card-header"><span className="card-title">📈 Price Chart — {selected}</span></div>
                        <div className="chart-container">
                            <PlotChart
                                data={candlestickTraces}
                                layout={{
                                    xaxis: { rangeslider: { visible: false }, gridcolor: '#1e293b', type: 'date' },
                                    yaxis: { gridcolor: '#1e293b', title: `Price (${currencyLabel})` },
                                    height: 450,
                                    legend: { orientation: 'h', y: 1.06, font: { size: 11 } },
                                }}
                                config={{ displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }}
                                style={{ width: '100%' }}
                            />
                        </div>
                    </div>

                    {/* Sub-charts Row */}
                    <div className="grid-2">
                        {showRSI && ind.length > 0 && (
                            <div className="card">
                                <div className="card-header"><span className="card-title">RSI (14)</span></div>
                                <PlotChart
                                    data={[
                                        { x: ind.map(x => x.date), y: ind.map(x => x.rsi_14), type: 'scatter', mode: 'lines', name: 'RSI', line: { color: '#8b5cf6', width: 2 } },
                                        { x: ind.map(x => x.date), y: ind.map(() => 70), type: 'scatter', mode: 'lines', name: 'Overbought', line: { color: '#ef4444', dash: 'dash', width: 1 } },
                                        { x: ind.map(x => x.date), y: ind.map(() => 30), type: 'scatter', mode: 'lines', name: 'Oversold', line: { color: '#10b981', dash: 'dash', width: 1 } },
                                    ]}
                                    layout={{
                                        margin: { t: 10, b: 30, l: 40, r: 10 },
                                        height: 200, showlegend: false,
                                        yaxis: { range: [0, 100], gridcolor: '#1e293b' },
                                        xaxis: { gridcolor: '#1e293b' },
                                    }}
                                    style={{ width: '100%' }}
                                />
                            </div>
                        )}
                        {showMACD && ind.length > 0 && (
                            <div className="card">
                                <div className="card-header"><span className="card-title">MACD</span></div>
                                <PlotChart
                                    data={[
                                        { x: ind.map(x => x.date), y: ind.map(x => x.macd), type: 'scatter', mode: 'lines', name: 'MACD', line: { color: '#3b82f6', width: 2 } },
                                        { x: ind.map(x => x.date), y: ind.map(x => x.macd_signal), type: 'scatter', mode: 'lines', name: 'Signal', line: { color: '#f59e0b', width: 1 } },
                                        { x: ind.map(x => x.date), y: ind.map(x => x.macd_histogram), type: 'bar', name: 'Histogram', marker: { color: ind.map(x => (x.macd_histogram || 0) >= 0 ? '#10b981' : '#ef4444') } },
                                    ]}
                                    layout={{
                                        margin: { t: 10, b: 30, l: 40, r: 10 },
                                        height: 200, showlegend: false,
                                        yaxis: { gridcolor: '#1e293b' },
                                        xaxis: { gridcolor: '#1e293b' },
                                    }}
                                    style={{ width: '100%' }}
                                />
                            </div>
                        )}
                    </div>
                </>
            )}
        </div>
    );
}
