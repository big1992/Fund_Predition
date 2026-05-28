import PlotChart from './PlotChart';
import { getCurrencyInfo } from '../utils/currencyUtils';

export default function PredictionResultSection({
    prediction,
    stocks,
    selected,
    aiLoading,
    aiExplanation,
    onRefreshAi,
}) {
    const pred = prediction?.predictions || [];
    const signal = prediction?.signal || 'HOLD';
    const { symbol: currencySymbol, label: currencyLabel } = getCurrencyInfo(stocks, selected);

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
                                    x: pred.map(p => p.date), y: pred.map(p => p.predicted_price * 1.02),
                                    type: 'scatter', mode: 'lines', name: 'Upper Band', line: { width: 0 }, showlegend: false,
                                },
                                {
                                    x: pred.map(p => p.date), y: pred.map(p => p.predicted_price * 0.98),
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
                        <thead><tr><th>Date</th><th>Predicted Price</th><th>Confidence</th></tr></thead>
                        <tbody>
                            {pred.map((p, i) => (
                                <tr key={i}>
                                    <td>{p.date}</td>
                                    <td style={{ fontWeight: 600 }}>{currencySymbol}{p.predicted_price.toFixed(2)}</td>
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
        </>
    );
}

