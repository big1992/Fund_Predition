import { useState, useEffect } from 'react';
import PlotChart from '../components/PlotChart';
import { getStocks, getModelPerformance, getValidationReportCard, getFeatureImportance, getAutogluonLeaderboard, runBacktest, explainPerformance } from '../api/client';
import { getCurrencyInfo } from '../utils/currencyUtils';

export default function ModelPerformance() {
    const [stocks, setStocks] = useState([]);
    const [selected, setSelected] = useState('');
    const [performance, setPerformance] = useState(null);
    const [features, setFeatures] = useState(null);
    const [agLeaderboard, setAgLeaderboard] = useState(null);
    const [reportCard, setReportCard] = useState(null);
    const [backtest, setBacktest] = useState(null);
    const [loading, setLoading] = useState(false);
    const [btLoading, setBtLoading] = useState(false);
    const [aiExplanation, setAiExplanation] = useState('');
    const [aiLoading, setAiLoading] = useState(false);

    useEffect(() => {
        getStocks().then(r => {
            setStocks(r.data);
            if (r.data.length > 0) setSelected(r.data[0].symbol);
        }).catch(() => { });
        getModelPerformance().then(r => setPerformance(r.data)).catch(() => { });
    }, []);

    useEffect(() => {
        if (!selected) return;
        getValidationReportCard(selected).then(r => setReportCard(r.data)).catch(() => setReportCard(null));
        getFeatureImportance(selected).then(r => setFeatures(r.data)).catch(() => setFeatures(null));
        getAutogluonLeaderboard(selected).then(r => setAgLeaderboard(r.data)).catch(() => setAgLeaderboard(null));
    }, [selected]);

    const handleBacktest = async () => {
        setBtLoading(true);
        try {
            const res = await runBacktest(selected);
            setBacktest(res.data);
        } catch (err) {
            setBacktest(null);
        } finally {
            setBtLoading(false);
        }
    };

    const fetchAIExplanation = async (bt = null) => {
        const models = performance?.models || [];
        if (models.length === 0 && !bt) return;
        setAiLoading(true);
        try {
            const res = await explainPerformance({
                models: models,
                backtest: bt,
            });
            setAiExplanation(res.data.explanation);
        } catch {
            setAiExplanation('⚠️ ไม่สามารถสร้างคำอธิบาย AI ได้');
        } finally {
            setAiLoading(false);
        }
    };

    const models = performance?.models || [];

    return (
        <div>
            <div className="page-header">
                <h2>Model Performance</h2>
                <p>ประเมินผลและเปรียบเทียบ ML Models</p>
            </div>

            {/* Metrics Table */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                    <span className="card-title">📊 Performance Metrics</span>
                    {models.length > 0 && (
                        <button className="btn btn-primary" style={{ fontSize: 12 }} onClick={() => fetchAIExplanation(backtest)} disabled={aiLoading}>
                            {aiLoading ? '⏳ Analyzing...' : '🧠 AI วิเคราะห์'}
                        </button>
                    )}
                </div>
                {models.length === 0 ? (
                    <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 20 }}>ยังไม่มีข้อมูล — กรุณา Train models ที่หน้า Prediction ก่อน</p>
                ) : (
                    <table className="data-table">
                        <thead>
                            <tr><th>Model</th><th>Symbol</th><th>RMSE</th><th>MAE</th><th>MAPE %</th><th>Direction Acc %</th><th>R²</th></tr>
                        </thead>
                        <tbody>
                            {models.map((m, i) => (
                                <tr key={i}>
                                    <td><span className="signal-badge hold" style={{ fontSize: 11, padding: '3px 10px', textTransform: 'uppercase' }}>{m.model_name}</span></td>
                                    <td style={{ fontWeight: 600 }}>{m.symbol}</td>
                                    <td>{m.rmse.toFixed(4)}</td>
                                    <td>{m.mae.toFixed(4)}</td>
                                    <td style={{ color: m.mape < 5 ? 'var(--accent-green)' : m.mape < 10 ? 'var(--accent-yellow)' : 'var(--accent-red)' }}>{m.mape.toFixed(2)}</td>
                                    <td style={{ color: m.directional_accuracy > 55 ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>{m.directional_accuracy.toFixed(1)}</td>
                                    <td>{m.r_squared.toFixed(4)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Validation Report Card */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                    <span className="card-title">Validation Report Card ({selected || '—'})</span>
                </div>
                {!reportCard?.cards?.length ? (
                    <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 20 }}>
                        No report card yet. Train models first.
                    </p>
                ) : (
                    <table className="data-table">
                        <thead>
                            <tr><th>Model</th><th>Status</th><th>MAPE %</th><th>Dir Acc %</th><th>Baseline Delta</th><th>Caveats</th></tr>
                        </thead>
                        <tbody>
                            {reportCard.cards.map((c, i) => {
                                const statusColor =
                                    c.status === 'pass' ? 'var(--accent-green)' :
                                        c.status === 'warn' ? 'var(--accent-yellow)' :
                                            c.status === 'reference' ? 'var(--accent-cyan)' :
                                                'var(--accent-red)';
                                return (
                                    <tr key={i}>
                                        <td style={{ fontWeight: 600 }}>{c.model_name}</td>
                                        <td><span className="signal-badge hold" style={{ fontSize: 11, padding: '3px 10px', color: statusColor }}>{c.status.toUpperCase()}</span></td>
                                        <td>{typeof c.mape === 'number' ? c.mape.toFixed(2) : '—'}</td>
                                        <td>{typeof c.directional_accuracy === 'number' ? c.directional_accuracy.toFixed(1) : '—'}</td>
                                        <td>
                                            {typeof c.mape_improvement_pct === 'number'
                                                ? `${c.mape_improvement_pct > 0 ? '+' : ''}${c.mape_improvement_pct.toFixed(2)}%`
                                                : '—'}
                                        </td>
                                        <td style={{ maxWidth: 360 }}>
                                            {Array.isArray(c.caveats) && c.caveats.length > 0 ? c.caveats.join(' | ') : '—'}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                )}
            </div>

            {/* AI Explanation */}
            {(aiExplanation || aiLoading) && (
                <div className="card" style={{ marginBottom: 20, borderLeft: '3px solid var(--accent-cyan)' }}>
                    <div className="card-header">
                        <span className="card-title">🧠 AI Performance Analysis</span>
                        {!aiLoading && (
                            <button className="btn btn-secondary" style={{ fontSize: 11 }} onClick={() => fetchAIExplanation(backtest)}>
                                🔄 วิเคราะห์ใหม่
                            </button>
                        )}
                    </div>
                    {aiLoading ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 0' }}>
                            <div className="spinner" style={{ width: 20, height: 20 }} />
                            <span style={{ color: 'var(--text-secondary)', fontSize: 14 }}>🧠 AI กำลังวิเคราะห์ผล Models...</span>
                        </div>
                    ) : (
                        <div style={{
                            color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.8,
                            whiteSpace: 'pre-wrap', padding: '4px 0',
                        }}>
                            {aiExplanation}
                        </div>
                    )}
                </div>
            )}

            <div className="grid-2">
                {/* Feature Importance */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">🔍 Feature Importance</span>
                        <select className="form-select" style={{ width: 'auto' }} value={selected} onChange={e => setSelected(e.target.value)}>
                            {stocks.map(s => <option key={s.symbol} value={s.symbol}>{s.symbol}</option>)}
                        </select>
                    </div>
                    {features?.features?.length > 0 ? (
                        <PlotChart
                            data={[{
                                y: features.features.map(f => f.feature).reverse(),
                                x: features.features.map(f => f.importance).reverse(),
                                type: 'bar', orientation: 'h',
                                marker: {
                                    color: features.features.map((_, i) => {
                                        const t = i / features.features.length;
                                        return `hsl(${220 + t * 100}, 70%, ${50 + t * 20}%)`;
                                    }).reverse(),
                                },
                            }]}
                            layout={{
                                margin: { t: 10, b: 30, l: 120, r: 20 },
                                height: 400, xaxis: { gridcolor: '#1e293b', title: 'Importance' },
                                yaxis: { gridcolor: '#1e293b' },
                            }}
                            style={{ width: '100%' }}
                        />
                    ) : (
                        <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 40 }}>Train XGBoost model for this stock first</p>
                    )}
                </div>

                {/* AutoGluon Leaderboard */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">🏆 AutoGluon Leaderboard</span>
                        {agLeaderboard?.best_model && (
                            <span style={{ fontSize: 11, color: 'var(--accent-green)', fontWeight: 600 }}>
                                Best: {agLeaderboard.best_model}
                            </span>
                        )}
                    </div>
                    {agLeaderboard?.leaderboard?.length > 0 ? (
                        <div style={{ overflowX: 'auto' }}>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Model</th>
                                        <th>Score (Val)</th>
                                        <th>Fit Time</th>
                                        <th>Pred Time</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {agLeaderboard.leaderboard.map((m, i) => (
                                        <tr key={i} style={m.model === agLeaderboard.best_model ? { background: 'rgba(16,185,129,0.08)' } : {}}>
                                            <td style={{ color: 'var(--text-muted)', fontSize: 11 }}>{i + 1}</td>
                                            <td style={{ fontWeight: m.model === agLeaderboard.best_model ? 700 : 400, fontSize: 12 }}>
                                                {m.model === agLeaderboard.best_model && '⭐ '}
                                                {m.model}
                                            </td>
                                            <td style={{ fontFamily: 'monospace', fontSize: 12 }}>
                                                {typeof m.score_val === 'number' ? m.score_val.toFixed(4) : m.score_val}
                                            </td>
                                            <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                                {typeof m.fit_time === 'number' ? `${m.fit_time.toFixed(1)}s` : '—'}
                                            </td>
                                            <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                                {typeof m.pred_time_val === 'number' ? `${m.pred_time_val.toFixed(3)}s` : '—'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 40 }}>Train AutoGluon model to see leaderboard</p>
                    )}
                </div>
            </div>

            {/* Backtest */}
            <div className="card" style={{ marginTop: 20 }}>
                <div className="card-header">
                    <span className="card-title">📉 Backtest Results</span>
                    <button className="btn btn-primary" style={{ fontSize: 12 }} onClick={handleBacktest} disabled={btLoading}>
                        {btLoading ? '⏳ Running...' : '▶ Run Backtest'}
                    </button>
                </div>
                {backtest ? (
                    <>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 16 }}>
                            {[
                                { label: 'Total Return', value: `${(backtest.total_return * 100).toFixed(2)}%`, positive: backtest.total_return >= 0 },
                                { label: 'Sharpe Ratio', value: backtest.sharpe_ratio.toFixed(3), positive: backtest.sharpe_ratio > 0 },
                                { label: 'Max Drawdown', value: `${(backtest.max_drawdown * 100).toFixed(2)}%`, positive: false },
                                { label: 'Win Rate', value: `${backtest.win_rate.toFixed(1)}%`, positive: backtest.win_rate > 50 },
                                { label: 'Total Trades', value: backtest.total_trades, positive: true },
                                { label: 'Profit Factor', value: backtest.profit_factor.toFixed(2), positive: backtest.profit_factor > 1 },
                            ].map(({ label, value, positive }) => (
                                <div key={label} style={{ textAlign: 'center', padding: 10, background: 'var(--bg-input)', borderRadius: 'var(--radius-sm)' }}>
                                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
                                    <div style={{ fontSize: 18, fontWeight: 700, color: positive ? 'var(--accent-green)' : 'var(--accent-red)' }}>{value}</div>
                                </div>
                            ))}
                        </div>
                        {backtest.equity_curve.length > 0 && (
                            <PlotChart
                                data={[{
                                    x: backtest.equity_curve.map(e => e.date),
                                    y: backtest.equity_curve.map(e => e.portfolio_value),
                                    type: 'scatter', mode: 'lines', name: 'Equity',
                                    line: { color: '#3b82f6', width: 2 },
                                    fill: 'tozeroy', fillcolor: 'rgba(59,130,246,0.1)',
                                }]}
                                layout={{
                                    margin: { t: 10, b: 30, l: 60, r: 10 },
                                    height: 200, showlegend: false,
                                    yaxis: { gridcolor: '#1e293b', title: getCurrencyInfo(stocks, selected).label },
                                    xaxis: { gridcolor: '#1e293b' },
                                }}
                                style={{ width: '100%' }}
                            />
                        )}
                    </>
                ) : (
                    <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 40 }}>กด "Run Backtest" เพื่อจำลองการเทรด</p>
                )}
            </div>
        </div>
    );
}
