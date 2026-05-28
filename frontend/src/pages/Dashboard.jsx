import { useState, useEffect } from 'react';
import PlotChart from '../components/PlotChart';
import { getHealth, getStocks, getFunds, collectData, getModelPerformance, getSchedulerStatus, startScheduler, stopScheduler } from '../api/client';
import { MdShowChart, MdFunctions, MdTrendingUp, MdStorage, MdAutorenew, MdTimeline } from 'react-icons/md';

export default function Dashboard() {
    const [health, setHealth] = useState(null);
    const [stocks, setStocks] = useState([]);
    const [funds, setFunds] = useState([]);
    const [loading, setLoading] = useState(true);
    const [collecting, setCollecting] = useState(false);
    const [collectMsg, setCollectMsg] = useState('');
    const [period, setPeriod] = useState('5y');
    const [modelPerf, setModelPerf] = useState(null);
    const [scheduler, setScheduler] = useState(null);

    useEffect(() => {
        Promise.all([
            getHealth().then(r => setHealth(r.data)).catch(() => { }),
            getStocks().then(r => setStocks(r.data)).catch(() => { }),
            getFunds().then(r => setFunds(r.data)).catch(() => { }),
            getModelPerformance().then(r => setModelPerf(r.data)).catch(() => { }),
            getSchedulerStatus().then(r => setScheduler(r.data)).catch(() => { }),
        ]).finally(() => setLoading(false));
    }, []);

    const handleCollect = async () => {
        setCollecting(true);
        setCollectMsg('กำลังดาวน์โหลดข้อมูล... (อาจใช้เวลา 1-3 นาที)');
        try {
            const res = await collectData([], period);
            setCollectMsg(`✅ ${res.data.message}`);
            const h = await getHealth();
            setHealth(h.data);
        } catch (err) {
            setCollectMsg(`❌ Error: ${err.response?.data?.detail || err.message}`);
        } finally {
            setCollecting(false);
        }
    };

    const sectorCounts = stocks.reduce((acc, s) => {
        acc[s.sector] = (acc[s.sector] || 0) + 1;
        return acc;
    }, {});

    return (
        <div>
            <div className="page-header">
                <h2>Dashboard</h2>
                <p>ภาพรวมระบบวิเคราะห์และพยากรณ์กองทุนรวมหุ้นไทย</p>
            </div>

            {/* KPI Cards */}
            <div className="metrics-grid">
                <div className="metric-card blue">
                    <div className="metric-label">หุ้นที่ Track</div>
                    <div className="metric-value">{stocks.length}</div>
                    <div className="metric-change" style={{ color: 'var(--text-muted)' }}>
                        <MdShowChart /> SET & mai
                    </div>
                </div>
                <div className="metric-card green">
                    <div className="metric-label">กองทุนจำลอง</div>
                    <div className="metric-value">{funds.length}</div>
                    <div className="metric-change" style={{ color: 'var(--text-muted)' }}>
                        <MdFunctions /> Growth, Value, Dividend, BlueChip
                    </div>
                </div>
                <div className="metric-card purple">
                    <div className="metric-label">ML Models</div>
                    <div className="metric-value">{health?.models_loaded?.length || 0}</div>
                    <div className="metric-change" style={{ color: 'var(--text-muted)' }}>
                        <MdTrendingUp /> LSTM + XGBoost + AutoGluon
                    </div>
                </div>
                <div className="metric-card red">
                    <div className="metric-label">สถานะระบบ</div>
                    <div className="metric-value" style={{ fontSize: 20, color: health?.status === 'healthy' ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        {health?.status === 'healthy' ? '🟢 Online' : '🔴 Offline'}
                    </div>
                    <div className="metric-change" style={{ color: 'var(--text-muted)' }}>
                        <MdStorage /> v{health?.version || '—'}
                    </div>
                </div>
            </div>

            {/* Data Collection */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                    <span className="card-title">📥 Data Collection</span>
                    <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                        <select className="form-select" style={{ width: 'auto', minWidth: 100 }} value={period} onChange={e => setPeriod(e.target.value)}>
                            <option value="1y">1 ปี</option>
                            <option value="2y">2 ปี</option>
                            <option value="3y">3 ปี</option>
                            <option value="5y">5 ปี</option>
                            <option value="10y">10 ปี</option>
                            <option value="max">ทั้งหมด</option>
                        </select>
                        <button className="btn btn-primary" onClick={handleCollect} disabled={collecting}>
                            {collecting ? '⏳ กำลังดาวน์โหลด...' : '🔄 ดาวน์โหลดข้อมูลหุ้น'}
                        </button>
                    </div>
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
                    {health?.data_available
                        ? '✅ มีข้อมูลในระบบแล้ว — สามารถใช้งาน Analysis, Prediction, Portfolio ได้'
                        : '⚠️ ยังไม่มีข้อมูล — กรุณากดปุ่ม "ดาวน์โหลดข้อมูลหุ้น" เพื่อเริ่มใช้งาน'}
                </p>
                {collectMsg && <p style={{ marginTop: 8, color: 'var(--accent-cyan)', fontSize: 14 }}>{collectMsg}</p>}
            </div>

            <div className="grid-2">
                {/* Stock Universe */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">📊 Stock Universe</span>
                    </div>
                    {Object.keys(sectorCounts).length > 0 ? (
                        <PlotChart
                            data={[{
                                labels: Object.keys(sectorCounts),
                                values: Object.values(sectorCounts),
                                type: 'pie',
                                hole: 0.5,
                                marker: {
                                    colors: ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899'],
                                },
                                textinfo: 'label+value',
                                textfont: { color: '#f1f5f9', size: 12 },
                            }]}
                            layout={{
                                margin: { t: 20, b: 20, l: 20, r: 20 },
                                height: 300,
                                showlegend: false,
                            }}
                            style={{ width: '100%' }}
                        />
                    ) : (
                        <div className="loading-container"><div className="spinner" /></div>
                    )}
                </div>

                {/* Fund Profiles */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">💼 Fund Profiles</span>
                    </div>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>กองทุน</th>
                                <th>สไตล์</th>
                                <th>หุ้น</th>
                            </tr>
                        </thead>
                        <tbody>
                            {funds.map(f => (
                                <tr key={f.fund_name}>
                                    <td style={{ fontWeight: 600 }}>{f.display_name}</td>
                                    <td>
                                        <span className="signal-badge hold" style={{ fontSize: 11, padding: '3px 10px' }}>
                                            {f.style}
                                        </span>
                                    </td>
                                    <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                                        {f.holdings.map(h => h.symbol.replace('.BK', '')).join(', ')}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Model Accuracy */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title"><MdTimeline style={{ verticalAlign: 'middle' }} /> Model Accuracy</span>
                    </div>
                    {modelPerf?.models && modelPerf.models.length > 0 ? (
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>MAPE %</th>
                                    <th>R²</th>
                                    <th>Dir. Acc</th>
                                </tr>
                            </thead>
                            <tbody>
                                {modelPerf.models.slice(0, 8).map(m => (
                                    <tr key={m.model_name}>
                                        <td style={{ fontWeight: 600, fontSize: 12 }}>{m.model_name}</td>
                                        <td style={{ color: m.mape < 5 ? 'var(--accent-green)' : m.mape < 10 ? 'var(--accent-yellow)' : 'var(--accent-red)' }}>
                                            {m.mape?.toFixed(1) || '—'}%
                                        </td>
                                        <td>{m.r_squared?.toFixed(3) || '—'}</td>
                                        <td>{m.directional_accuracy?.toFixed(0) || '—'}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <p style={{ color: 'var(--text-muted)', fontSize: 13, padding: 10 }}>ยังไม่มีข้อมูล — Train models ก่อน</p>
                    )}
                </div>

                {/* Auto-Retrain Scheduler */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title"><MdAutorenew style={{ verticalAlign: 'middle' }} /> Auto-Retrain</span>
                        <button className={`btn ${scheduler?.running ? 'btn-danger' : 'btn-success'}`}
                            onClick={async () => {
                                if (scheduler?.running) {
                                    await stopScheduler();
                                } else {
                                    await startScheduler(168);
                                }
                                const res = await getSchedulerStatus();
                                setScheduler(res.data);
                            }}
                            style={{ fontSize: 11, padding: '4px 12px' }}>
                            {scheduler?.running ? '⏹ Stop' : '▶ Start Weekly'}
                        </button>
                    </div>
                    <div style={{ padding: '8px 16px', fontSize: 13, color: 'var(--text-secondary)' }}>
                        <p>สถานะ: {scheduler?.running ? '🟢 Running' : '⚪ Stopped'}</p>
                        {scheduler?.last_retrain && <p>Retrain ล่าสุด: {new Date(scheduler.last_retrain).toLocaleString('th-TH')}</p>}
                        {scheduler?.next_retrain && scheduler?.running && <p>ครั้งถัดไป: {new Date(scheduler.next_retrain).toLocaleString('th-TH')}</p>}
                    </div>
                </div>
            </div>
        </div>
    );
}
