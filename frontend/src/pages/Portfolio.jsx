import { useState, useEffect } from 'react';
import PlotChart from '../components/PlotChart';
import { getStocks, optimizePortfolio } from '../api/client';

export default function Portfolio() {
    const [stocks, setStocks] = useState([]);
    const [selectedStocks, setSelectedStocks] = useState([]);
    const [method, setMethod] = useState('max_sharpe');
    const [riskLevel, setRiskLevel] = useState(0.5);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        getStocks().then(r => {
            setStocks(r.data);
            setSelectedStocks(r.data.slice(0, 5).map(s => s.symbol));
        }).catch(() => { });
    }, []);

    const toggleStock = (sym) => {
        setSelectedStocks(prev =>
            prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym]
        );
    };

    const handleOptimize = async () => {
        if (selectedStocks.length < 2) {
            setError('เลือกอย่างน้อย 2 หุ้น');
            return;
        }
        setLoading(true); setError('');
        try {
            const res = await optimizePortfolio(selectedStocks, method, riskLevel);
            setResult(res.data);
        } catch (err) {
            setError(err.response?.data?.detail || err.message);
        } finally {
            setLoading(false);
        }
    };

    const methodLabels = {
        max_sharpe: '⚡ Max Sharpe Ratio',
        min_volatility: '🛡️ Min Volatility',
        risk_parity: '⚖️ Risk Parity',
    };

    const weights = result?.weights || {};
    const weightEntries = Object.entries(weights).filter(([, v]) => v > 0.001);
    const colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#6366f1'];

    return (
        <div>
            <div className="page-header">
                <h2>Portfolio Optimization</h2>
                <p>จัดพอร์ตลงทุนอัตโนมัติด้วย Modern Portfolio Theory</p>
            </div>

            {/* Stock Selection + Controls */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header"><span className="card-title">เลือกหุ้นเข้าพอร์ต ({selectedStocks.length} selected)</span></div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
                    {stocks.map(s => (
                        <button key={s.symbol}
                            className={`btn ${selectedStocks.includes(s.symbol) ? 'btn-primary' : 'btn-secondary'}`}
                            style={{ fontSize: 12, padding: '6px 14px' }}
                            onClick={() => toggleStock(s.symbol)}>
                            {s.symbol.replace('.BK', '')}
                        </button>
                    ))}
                </div>

                <div style={{ display: 'flex', gap: 20, alignItems: 'end', flexWrap: 'wrap' }}>
                    <div className="form-group" style={{ marginBottom: 0, minWidth: 200 }}>
                        <label className="form-label">Optimization Method</label>
                        <select className="form-select" value={method} onChange={e => setMethod(e.target.value)}>
                            {Object.entries(methodLabels).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                        </select>
                    </div>
                    <div className="form-group" style={{ marginBottom: 0, minWidth: 200 }}>
                        <label className="form-label">Risk Level: {(riskLevel * 100).toFixed(0)}%</label>
                        <input type="range" min="0" max="1" step="0.05" value={riskLevel}
                            onChange={e => setRiskLevel(parseFloat(e.target.value))}
                            style={{ width: '100%' }} />
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }}>
                            <span>Conservative</span><span>Aggressive</span>
                        </div>
                    </div>
                    <button className="btn btn-primary" onClick={handleOptimize} disabled={loading}>
                        {loading ? '⏳ Optimizing...' : '🎯 Optimize'}
                    </button>
                </div>
                {error && <p style={{ marginTop: 12, color: 'var(--accent-red)', fontSize: 13 }}>{error}</p>}
            </div>

            {result && (
                <>
                    {/* Metrics */}
                    <div className="metrics-grid">
                        <div className="metric-card green">
                            <div className="metric-label">Expected Return</div>
                            <div className="metric-value">{(result.expected_return * 100).toFixed(2)}%</div>
                            <div className="metric-change" style={{ color: 'var(--text-muted)' }}>ต่อปี</div>
                        </div>
                        <div className="metric-card red">
                            <div className="metric-label">Volatility</div>
                            <div className="metric-value">{(result.volatility * 100).toFixed(2)}%</div>
                            <div className="metric-change" style={{ color: 'var(--text-muted)' }}>ความเสี่ยง</div>
                        </div>
                        <div className="metric-card blue">
                            <div className="metric-label">Sharpe Ratio</div>
                            <div className="metric-value">{result.sharpe_ratio.toFixed(3)}</div>
                            <div className="metric-change" style={{ color: 'var(--text-muted)' }}>Risk-adjusted return</div>
                        </div>
                        <div className="metric-card purple">
                            <div className="metric-label">Method</div>
                            <div className="metric-value" style={{ fontSize: 16 }}>{methodLabels[result.method]}</div>
                        </div>
                    </div>

                    <div className="grid-2">
                        {/* Efficient Frontier */}
                        <div className="card">
                            <div className="card-header"><span className="card-title">📈 Efficient Frontier</span></div>
                            <PlotChart
                                data={[
                                    {
                                        x: result.frontier.map(f => f.volatility * 100),
                                        y: result.frontier.map(f => f.expected_return * 100),
                                        mode: 'markers', type: 'scatter', name: 'Portfolios',
                                        marker: { color: result.frontier.map(f => f.sharpe_ratio), colorscale: 'Viridis', size: 5, showscale: true, colorbar: { title: 'Sharpe', tickfont: { color: '#94a3b8' }, titlefont: { color: '#94a3b8' } } },
                                    },
                                    {
                                        x: [result.volatility * 100], y: [result.expected_return * 100],
                                        mode: 'markers', type: 'scatter', name: 'Optimal',
                                        marker: { color: '#ef4444', size: 16, symbol: 'star', line: { width: 2, color: '#fff' } },
                                    },
                                ]}
                                layout={{
                                    margin: { t: 20, b: 50, l: 60, r: 80 },
                                    height: 380,
                                    xaxis: { title: 'Volatility (%)', gridcolor: '#1e293b' },
                                    yaxis: { title: 'Expected Return (%)', gridcolor: '#1e293b' },
                                    legend: { orientation: 'h', y: 1.1 },
                                }}
                                style={{ width: '100%' }}
                            />
                        </div>

                        {/* Allocation Pie */}
                        <div className="card">
                            <div className="card-header"><span className="card-title">🥧 Allocation</span></div>
                            <PlotChart
                                data={[{
                                    labels: weightEntries.map(([k]) => k.replace('.BK', '')),
                                    values: weightEntries.map(([, v]) => (v * 100).toFixed(1)),
                                    type: 'pie', hole: 0.55,
                                    marker: { colors: colors.slice(0, weightEntries.length) },
                                    textinfo: 'label+percent',
                                    textfont: { color: '#f1f5f9', size: 12 },
                                }]}
                                layout={{
                                    margin: { t: 20, b: 20, l: 20, r: 20 },
                                    height: 380, showlegend: false,
                                    annotations: [{ text: `Sharpe<br>${result.sharpe_ratio.toFixed(2)}`, showarrow: false, font: { size: 16, color: '#f1f5f9' } }],
                                }}
                                style={{ width: '100%' }}
                            />
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
