import PlotChart from './PlotChart';
import { getCurrencyInfo } from '../utils/currencyUtils';

export default function PredictionResultSection({
    prediction,
    stocks,
    selected,
    validationReport,
    aiLoading,
    aiExplanation,
    onRefreshAi,
}) {
    const pred = prediction?.predictions || [];
    const signal = prediction?.signal || 'HOLD';
    const { symbol: currencySymbol, label: currencyLabel } = getCurrencyInfo(stocks, selected);
    const cards = validationReport?.cards || [];
    const ensembleCard = cards.find(c => c.model_name === 'ensemble');
    const trustStatus = ensembleCard?.status || 'reference';
    const trustColor = trustStatus === 'pass'
        ? 'var(--accent-green)'
        : trustStatus === 'warn'
            ? 'var(--accent-yellow)'
            : trustStatus === 'fail'
                ? 'var(--accent-red)'
                : 'var(--accent-cyan)';
    // Build labeled caveats from each card (prefix with model name), then deduplicate
    const labeledCaveats = cards
        .filter(c => c.model_name !== 'naive' && c.model_name !== 'mean_return' && c.model_name !== 'buy_hold' && c.model_name !== 'benchmark')
        .flatMap(c => (c.caveats || []).filter(Boolean).map(cav => `[${c.model_name?.toUpperCase()}] ${cav}`));
    const topCaveats = [...new Set(labeledCaveats)].slice(0, 5);

    return (
        <>
            <div className="metrics-grid">
                <div className="metric-card blue">
                    <div className="metric-label">Current Price</div>
                    <div className="metric-value">{currencySymbol}{prediction.current_price?.toFixed(2)}</div>
                </div>
                <div className="metric-card green">
                    <div className="metric-label">Signal</div>
                    <div style={{ marginTop: 8 }}>
                        <span className={`signal-badge ${signal.toLowerCase()}`} style={{ fontSize: 18, padding: '8px 24px' }}>
                            {signal === 'BUY' ? '🟢' : signal === 'SELL' ? '🔴' : '🟡'} {signal}
                        </span>
                    </div>
                </div>
                <div className="metric-card purple">
                    <div className="metric-label">Confidence</div>
                    <div className="metric-value">{prediction.confidence?.toFixed(1)}%</div>
                    <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                        {prediction.confidence_band || 'n/a'}
                    </div>
                </div>
                <div className="metric-card orange">
                    <div className="metric-label">Disagreement</div>
                    <div className="metric-value">{typeof prediction.model_disagreement_pct === 'number' ? `${prediction.model_disagreement_pct.toFixed(2)}%` : 'n/a'}</div>
                </div>
                <div className="metric-card red">
                    <div className="metric-label">Model</div>
                    <div className="metric-value" style={{ fontSize: 18, textTransform: 'uppercase' }}>{prediction.model}</div>
                </div>
            </div>

            <div className="card" style={{ marginBottom: 20, borderLeft: '3px solid var(--accent-cyan)' }}>
                <div className="card-header">
                    <span className="card-title">AI Analysis</span>
                    {!aiLoading && aiExplanation && (
                        <button className="btn btn-secondary" style={{ fontSize: 11 }} onClick={onRefreshAi}>
                            Refresh
                        </button>
                    )}
                </div>
                {aiLoading ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 0' }}>
                        <div className="spinner" style={{ width: 20, height: 20 }} />
                        <span style={{ color: 'var(--text-secondary)', fontSize: 14 }}>AI is analyzing...</span>
                    </div>
                ) : aiExplanation ? (
                    <div style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.8, whiteSpace: 'pre-wrap', padding: '4px 0' }}>
                        {aiExplanation}
                    </div>
                ) : (
                    <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Run prediction to generate AI analysis.</p>
                )}
            </div>

            <div className="grid-2" style={{ marginBottom: 20 }}>
                <div className="card" style={{ borderLeft: `3px solid ${trustColor}` }}>
                    <div className="card-header"><span className="card-title">Model Trust Snapshot</span></div>
                    <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 8 }}>
                        Validation status: <span style={{ color: trustColor, fontWeight: 700, textTransform: 'uppercase' }}>{trustStatus}</span>
                    </p>
                    <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                        {ensembleCard?.summary || 'No ensemble validation summary available for this symbol yet.'}
                    </p>
                </div>
                <div className="card" style={{ borderLeft: '3px solid var(--accent-yellow)' }}>
                    <div className="card-header"><span className="card-title">Risk Caveats</span></div>
                    {topCaveats.length > 0 ? (
                        <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.8 }}>
                            {topCaveats.map((item, i) => <li key={i}>{item}</li>)}
                        </ul>
                    ) : cards.length > 0 ? (
                        <p style={{ color: 'var(--accent-green)', fontSize: 13 }}>
                            ✅ ทุกโมเดลผ่านมาตรฐาน — ไม่มีข้อเตือนเพิ่มเติม
                        </p>
                    ) : (
                        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                            ยังไม่มีข้อมูล Validation — กรุณา Train Models ก่อนเพื่อดู Risk Caveats
                        </p>
                    )}
                </div>
            </div>

            <div className="grid-2">
                <div className="card">
                    <div className="card-header"><span className="card-title">Forecast — {selected}</span></div>
                    {pred.length > 0 && (
                        <PlotChart
                            data={[
                                {
                                    x: pred.map(p => p.date), y: pred.map(p => p.predicted_price),
                                    type: 'scatter', mode: 'lines+markers', name: 'Predicted',
                                    line: { color: '#3b82f6', width: 3, dash: 'dash' }, marker: { size: 8 },
                                },
                                {
                                    x: pred.map(p => p.date),
                                    y: pred.map(p => (typeof p.predicted_high === 'number' ? p.predicted_high : p.predicted_price * 1.02)),
                                    type: 'scatter', mode: 'lines', name: 'Upper Band', line: { width: 0 }, showlegend: false,
                                },
                                {
                                    x: pred.map(p => p.date),
                                    y: pred.map(p => (typeof p.predicted_low === 'number' ? p.predicted_low : p.predicted_price * 0.98)),
                                    type: 'scatter', mode: 'lines', name: 'Lower Band', line: { width: 0 }, showlegend: false,
                                    fill: 'tonexty', fillcolor: 'rgba(59,130,246,0.1)',
                                },
                            ]}
                            layout={{
                                height: 320, showlegend: true,
                                yaxis: { gridcolor: '#1e293b', title: `Price (${currencyLabel})` },
                                xaxis: { gridcolor: '#1e293b' },
                                legend: { orientation: 'h', y: 1.1 },
                            }}
                            style={{ width: '100%' }}
                        />
                    )}
                </div>

                <div className="card">
                    <div className="card-header"><span className="card-title">Forecast Table</span></div>
                    <table className="data-table">
                        <thead><tr><th>Date</th><th>Downside</th><th>Base</th><th>Upside</th><th>Uncertainty</th><th>Confidence</th></tr></thead>
                        <tbody>
                            {pred.map((p, i) => (
                                <tr key={i}>
                                    <td>{p.date}</td>
                                    <td>{currencySymbol}{(typeof p.predicted_low === 'number' ? p.predicted_low : p.predicted_price * 0.98).toFixed(2)}</td>
                                    <td style={{ fontWeight: 600 }}>{currencySymbol}{p.predicted_price.toFixed(2)}</td>
                                    <td>{currencySymbol}{(typeof p.predicted_high === 'number' ? p.predicted_high : p.predicted_price * 1.02).toFixed(2)}</td>
                                    <td>{typeof p.uncertainty_pct === 'number' ? `${p.uncertainty_pct.toFixed(2)}%` : 'n/a'}</td>
                                    <td>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                            <div style={{ flex: 1, background: 'var(--bg-input)', borderRadius: 4, height: 6, overflow: 'hidden' }}>
                                                <div style={{ width: `${p.confidence}%`, height: '100%', background: 'var(--gradient-blue)', borderRadius: 4 }} />
                                            </div>
                                            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{p.confidence.toFixed(1)}%</span>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="card" style={{ marginTop: 20, border: '1px dashed rgba(245, 158, 11, 0.5)', background: 'rgba(245, 158, 11, 0.05)' }}>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.7 }}>
                    This output is research analytics, not investment advice. Use downside/base/upside range, disagreement, and confidence
                    together with your own risk limits before making any trade decision.
                </p>
            </div>
        </>
    );
}
