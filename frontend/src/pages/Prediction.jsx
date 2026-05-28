import { useState, useEffect, useRef } from 'react';
import PlotChart from '../components/PlotChart';
import PredictionResultSection from '../components/PredictionResultSection';
import { getStocks, getPrediction, trainModels, trainModelsAsync, getIndicators, explainPrediction, explainTraining, getSettings, updateSettings, getBestTrainingRun, getTrainingHistory } from '../api/client';
import { useTraining } from '../context/TrainingContext';

export default function Prediction() {
    const { startPolling } = useTraining();
    const [stocks, setStocks] = useState([]);
    const [selected, setSelected] = useState('');
    const [model, setModel] = useState('ensemble');
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [training, setTraining] = useState(false);
    const [trainMsg, setTrainMsg] = useState('');
    const [trainResult, setTrainResult] = useState(null);
    const [walkForward, setWalkForward] = useState(false);
    const [showTrainResult, setShowTrainResult] = useState(false);
    const [aiExplanation, setAiExplanation] = useState('');
    const [aiLoading, setAiLoading] = useState(false);
    const [trainAiAnalysis, setTrainAiAnalysis] = useState('');
    const [trainAiLoading, setTrainAiLoading] = useState(false);
    const [aiRecommendedParams, setAiRecommendedParams] = useState(null);
    const [applySettingsLoading, setApplySettingsLoading] = useState(false);
    // Auto-Tune state
    const [autoTuning, setAutoTuning] = useState(false);
    const [autoTuneLogs, setAutoTuneLogs] = useState([]);
    const [autoTuneRound, setAutoTuneRound] = useState(0);
    const [autoTuneHistory, setAutoTuneHistory] = useState([]);  // for chart
    const [autoTuneHistoryApi, setAutoTuneHistoryApi] = useState([]);
    const autoTuneCancelRef = useRef(false);
    const MAX_AUTO_TUNE_ROUNDS = 3;

    useEffect(() => {
        getStocks().then(r => {
            setStocks(r.data);
            if (r.data.length > 0) setSelected(r.data[0].symbol);
        }).catch(() => { });
    }, []);

    const handlePredict = async () => {
        if (!selected) return;
        setLoading(true);
        setAiExplanation('');
        try {
            const res = await getPrediction(selected, model);
            setPrediction(res.data);
            if (res.data?.predictions?.length > 0) fetchAIExplanation(res.data);
        } catch (err) {
            setPrediction(null);
            setTrainMsg(`❌ ${err.response?.data?.detail || 'Error getting prediction. Train models first!'}`);
        } finally { setLoading(false); }
    };

    const fetchAIExplanation = async (predData) => {
        setAiLoading(true);
        try {
            let indicators = null;
            try {
                const indRes = await getIndicators(selected);
                const indList = indRes.data?.indicators || [];
                if (indList.length > 0) indicators = indList[indList.length - 1];
            } catch { }
            const res = await explainPrediction({
                symbol: selected, current_price: predData.current_price,
                predictions: predData.predictions, signal: predData.signal,
                signal_reason: predData.signal_reason, model_type: predData.model, indicators,
            });
            setAiExplanation(res.data.explanation);
        } catch { setAiExplanation('⚠️ ไม่สามารถสร้างคำอธิบาย AI ได้'); }
        finally { setAiLoading(false); }
    };

    const handleTrain = async () => {
        setTraining(true);
        setTrainMsg('🔄 กำลัง Train models... (อาจใช้เวลา 2-5 นาที)');
        setTrainResult(null);
        try {
            // Start async training (background) — non-blocking
            const asyncRes = await trainModelsAsync({ symbols: [selected], model_type: 'all', walk_forward: walkForward });
            startPolling();
            setTrainMsg(`✅ Training เริ่มทำงานใน Background (Task: ${asyncRes.data.task_id}) — สามารถเปลี่ยนหน้าได้เลย! ดู status ที่มุมขวาล่าง 👇`);
        } catch (err) {
            setTrainMsg(`❌ ${err.response?.data?.detail || err.message}`);
        } finally { setTraining(false); }
    };

    // Extract training metrics for display
    const getTrainMetrics = () => {
        if (!trainResult?.metrics) return null;
        const symbolKey = Object.keys(trainResult.metrics)[0];
        if (!symbolKey) return null;
        return trainResult.metrics[symbolKey];
    };

    const trainMetrics = getTrainMetrics();

    const handleAiTrainAnalysis = async () => {
        if (!trainMetrics) return;
        setTrainAiLoading(true);
        setTrainAiAnalysis('');
        try {
            // Fetch current settings to pass as context
            let currentParams = {};
            try {
                const settingsRes = await getSettings();
                currentParams = settingsRes.data;
            } catch { }
            const res = await explainTraining({
                symbol: selected,
                metrics: trainMetrics,
                current_params: currentParams,
            });
            setTrainAiAnalysis(res.data.explanation);
            if (res.data.recommended_params && Object.keys(res.data.recommended_params).length > 0) {
                setAiRecommendedParams(res.data.recommended_params);
            } else {
                setAiRecommendedParams(null);
            }
        } catch (err) {
            setTrainAiAnalysis('⚠️ ไม่สามารถวิเคราะห์ผล Train ได้: ' + (err.response?.data?.detail || err.message));
            setAiRecommendedParams(null);
        } finally { setTrainAiLoading(false); }
    };

    const handleApplyAiSettings = async () => {
        if (!aiRecommendedParams) return;
        setApplySettingsLoading(true);
        try {
            const currentRes = await getSettings();
            const currentSettings = currentRes.data;
            const newSettings = { ...currentSettings, ...aiRecommendedParams };
            await updateSettings(newSettings);
            setAiRecommendedParams(null);
            setTrainAiAnalysis('');
            setApplySettingsLoading(false);

            // Auto-train with new settings
            setTrainMsg('✅ นำค่าแนะนำไปปรับใช้สำเร็จ! 🔄 กำลัง Train ใหม่ด้วยค่าที่ปรับแล้ว...');
            setTraining(true);
            setTrainResult(null);
            try {
                const res = await trainModels([selected], 'all');
                setTrainMsg(`✅ Train ด้วยค่าใหม่สำเร็จ! ${res.data.message}`);
                setTrainResult(res.data);
                setShowTrainResult(true);
            } catch (trainErr) {
                setTrainMsg(`❌ Train ล้มเหลว: ${trainErr.response?.data?.detail || trainErr.message}`);
            } finally {
                setTraining(false);
            }
        } catch (err) {
            alert('❌ ไม่สามารถบันทึกค่า Settings ได้: ' + (err.response?.data?.detail || err.message));
            setApplySettingsLoading(false);
        }
    };

    // ============ AUTO-TUNE LOOP ============
    const handleAutoTune = async () => {
        if (!selected || autoTuning) return;
        setAutoTuning(true);
        setAutoTuneLogs([]);
        setAutoTuneRound(0);
        setTrainAiAnalysis('');
        setAiRecommendedParams(null);
        setShowTrainResult(false);
        setAutoTuneHistory([]);
        setAutoTuneHistoryApi([]);
        autoTuneCancelRef.current = false;

        const addLog = (msg) => setAutoTuneLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), msg }]);

        let bestScore = 0;
        let prevRoundMetrics = null;  // Track previous round for before/after
        let prevRoundParams = null;

        for (let round = 1; round <= MAX_AUTO_TUNE_ROUNDS; round++) {
            if (autoTuneCancelRef.current) {
                addLog(`⛔ Auto-Tune ถูกหยุดโดยผู้ใช้`);
                break;
            }
            setAutoTuneRound(round);

            // Step 1: Train
            addLog(`🔄 รอบ ${round}/${MAX_AUTO_TUNE_ROUNDS} — กำลัง Train Models...`);
            setTraining(true);
            let metrics;
            try {
                const trainRes = await trainModels([selected], 'all');
                metrics = trainRes.data.metrics;
                setTrainResult(trainRes.data);
                setShowTrainResult(true);
                const lstmRmse = metrics?.[selected]?.lstm?.rmse ?? '?';
                const xgbRmse = metrics?.[selected]?.xgboost?.rmse ?? '?';
                addLog(`✅ Train สำเร็จ — LSTM RMSE: ${lstmRmse}, XGB RMSE: ${xgbRmse}`);
            } catch (err) {
                addLog(`❌ Train ล้มเหลว: ${err.response?.data?.detail || err.message}`);
                break;
            } finally {
                setTraining(false);
            }

            // Step 2: AI Analysis (with before/after data)
            addLog(`🧠 รอบ ${round} — AI กำลังวิเคราะห์ผลและให้คะแนน...`);
            let aiResult;
            try {
                const currentSettingsRes = await getSettings();
                const currentParams = currentSettingsRes.data;

                // Build request with previous round data for comparison
                const aiPayload = {
                    symbol: selected,
                    metrics: metrics?.[selected] || metrics,
                    current_params: currentParams,
                };

                // Send previous round metrics for before/after comparison
                if (prevRoundMetrics) {
                    aiPayload.previous_metrics = {
                        metrics: prevRoundMetrics,
                        params: prevRoundParams || {},
                    };
                    addLog(`📋 รอบ ${round} — ส่งข้อมูล Before/After ให้ AI เปรียบเทียบ`);
                }

                const aiRes = await explainTraining(aiPayload);
                aiResult = aiRes.data;
                const score = aiResult.satisfaction_score || 5;
                bestScore = score;
                setTrainAiAnalysis(aiResult.explanation);
                addLog(`📊 รอบ ${round} — AI Score: ${score}/10${score >= 8 ? ' ✨ ดีเยี่ยม!' : score >= 6 ? ' 👍 พอใช้ได้' : ' ⚠️ ควรปรับปรุง'}`);

                // Track history for chart
                const curMetrics = metrics?.[selected] || metrics;
                setAutoTuneHistory(prev => [...prev, {
                    round,
                    lstm_mape: curMetrics?.lstm?.mape ?? null,
                    xgb_mape: curMetrics?.xgboost?.mape ?? null,
                    lstm_dir_acc: curMetrics?.lstm?.directional_accuracy ?? null,
                    xgb_dir_acc: curMetrics?.xgboost?.directional_accuracy ?? null,
                    score,
                }]);

                // Save current round as "previous" for next iteration
                prevRoundMetrics = metrics?.[selected] || metrics;
                prevRoundParams = { ...currentParams };

                // Check if satisfactory
                if (score >= 8) {
                    addLog(`🎉 คะแนน ${score}/10 — ผลลัพธ์ดีมาก! หยุด Auto-Tune`);
                    setAiRecommendedParams(null);
                    break;
                }

                // Check if it's the last round
                if (round === MAX_AUTO_TUNE_ROUNDS) {
                    addLog(`⏱️ ครบ ${MAX_AUTO_TUNE_ROUNDS} รอบแล้ว — Score: ${score}/10 — หยุด Auto-Tune`);
                    if (aiResult.recommended_params && Object.keys(aiResult.recommended_params).length > 0) {
                        setAiRecommendedParams(aiResult.recommended_params);
                        addLog(`💡 ยังมีค่าแนะนำเพิ่มเติม กดปุ่ม Apply ด้านล่างเพื่อทดลองต่อด้วยตนเอง`);
                    }
                    break;
                }

                // Step 3: Apply recommended params
                const recParams = aiResult.recommended_params;
                if (!recParams || Object.keys(recParams).length === 0) {
                    addLog(`✅ AI ไม่แนะนำให้เปลี่ยนค่าเพิ่ม — หยุด Auto-Tune`);
                    break;
                }

                const changedKeys = Object.keys(recParams).join(', ');
                addLog(`⚙️ รอบ ${round} — ปรับค่า: ${changedKeys}`);
                try {
                    const newSettings = { ...currentParams, ...recParams };
                    await updateSettings(newSettings);
                    addLog(`✅ อัปเดต Settings สำเร็จ — เริ่มรอบถัดไป...`);
                } catch (err) {
                    addLog(`❌ อัปเดต Settings ล้มเหลว: ${err.message}`);
                    break;
                }

            } catch (err) {
                addLog(`❌ AI วิเคราะห์ล้มเหลว: ${err.response?.data?.detail || err.message}`);
                break;
            }
        }

        // After loop: check best-run from DB and revert to best settings if needed
        addLog(`🔍 กำลังค้นหาค่าที่ดีที่สุดจากประวัติทั้งหมด...`);
        try {
            const bestRes = await getBestTrainingRun(selected);
            const bestRun = bestRes.data?.best_run;
            if (bestRun && bestRun.params && Object.keys(bestRun.params).length > 0) {
                const bestMape = (((bestRun.lstm_mape || 0) + (bestRun.xgb_mape || 0)) / 2).toFixed(2);
                addLog(`🏆 Best Run (id=${bestRun.id}): Avg MAPE = ${bestMape}%`);

                // Apply best-run params to settings
                const currentSettingsRes = await getSettings();
                const newSettings = { ...currentSettingsRes.data, ...bestRun.params };
                await updateSettings(newSettings);
                addLog(`✅ เปลี่ยน Settings กลับไปใช้ค่าจาก Best Run สำเร็จ!`);
            } else {
                addLog(`ℹ️ ยังไม่มีประวัติเพียงพอสำหรับเปรียบเทียบ`);
            }
        } catch (err) {
            addLog(`⚠️ ไม่สามารถดึงข้อมูล Best Run ได้: ${err.message}`);
        }

        addLog(`🏁 Auto-Tune เสร็จสิ้น — Best Score: ${bestScore}/10`);
        try {
            const historyRes = await getTrainingHistory(selected, 20);
            const historyRows = historyRes?.data?.history || [];
            const normalized = historyRows
                .map((row, idx) => {
                    const avgMape = row?.lstm_mape != null || row?.xgb_mape != null
                        ? (((row?.lstm_mape ?? row?.xgb_mape ?? 0) + (row?.xgb_mape ?? row?.lstm_mape ?? 0)) / 2)
                        : null;
                    const avgDir = row?.lstm_directional_accuracy != null || row?.xgb_directional_accuracy != null
                        ? (((row?.lstm_directional_accuracy ?? row?.xgb_directional_accuracy ?? 0) + (row?.xgb_directional_accuracy ?? row?.lstm_directional_accuracy ?? 0)) / 2)
                        : null;
                    return {
                        runLabel: `Run ${historyRows.length - idx}`,
                        createdAt: row?.created_at || '',
                        avgMape,
                        avgDir,
                    };
                })
                .reverse();
            setAutoTuneHistoryApi(normalized);
            if (normalized.length > 0) addLog(`📈 โหลดประวัติการฝึกจากระบบ ${normalized.length} รายการ`);
        } catch (err) {
            addLog(`⚠️ โหลดประวัติการฝึกไม่สำเร็จ: ${err?.message || 'Unknown error'}`);
        }
        setAutoTuning(false);
    };

    return (
        <div>
            <div className="page-header">
                <h2>Prediction</h2>
                <p>พยากรณ์ราคาหุ้นด้วย AI (LSTM + XGBoost + AutoGluon + Ensemble)</p>
            </div>

            {/* Controls */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', gap: 16, alignItems: 'end', flexWrap: 'wrap' }}>
                    <div className="form-group" style={{ marginBottom: 0, minWidth: 200 }}>
                        <label className="form-label">เลือกหุ้น</label>
                        <select className="form-select" value={selected} onChange={e => setSelected(e.target.value)}>
                            {stocks.map(s => <option key={s.symbol} value={s.symbol}>{s.symbol}</option>)}
                        </select>
                    </div>
                    <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">Model</label>
                        <div className="btn-group">
                            {['lstm', 'xgboost', 'autogluon', 'ensemble'].map(m => (
                                <button key={m} className={`btn btn-secondary ${model === m ? 'active' : ''}`}
                                    onClick={() => setModel(m)} style={{ textTransform: 'uppercase', fontSize: 12 }}>
                                    {m === 'autogluon' ? 'AG' : m}
                                </button>
                            ))}
                        </div>
                    </div>
                    <button className="btn btn-primary" onClick={handlePredict} disabled={loading}>
                        {loading ? '⏳ Predicting...' : '🔮 Predict'}
                    </button>
                    <button className="btn btn-success" onClick={handleTrain} disabled={training || autoTuning}>
                        {training ? '⏳ Training...' : walkForward ? '🧠 Walk-Forward Train' : '🧠 Train Models'}
                    </button>
                    <button
                        className={`btn ${walkForward ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => setWalkForward(!walkForward)}
                        style={{ fontSize: 11, padding: '6px 12px', ...(walkForward ? { background: '#8b5cf6', border: 'none' } : {}) }}
                        title="Walk-Forward Validation: train ด้วย expanding window หลาย folds เพื่อ metrics ที่น่าเชื่อถือกว่า"
                    >
                        {walkForward ? '✅ Walk-Forward ON' : '🔄 Walk-Forward'}
                    </button>
                    <button className="btn btn-primary" onClick={handleAutoTune} disabled={training || autoTuning}
                        style={{ background: 'linear-gradient(135deg, #8b5cf6, #06b6d4)', border: 'none', fontSize: 12 }}>
                        {autoTuning ? `⏳ Auto-Tune รอบ ${autoTuneRound}/${MAX_AUTO_TUNE_ROUNDS}...` : '🔄 Auto-Tune (AI ปรับค่าอัตโนมัติ)'}
                    </button>
                    {autoTuning && (
                        <button className="btn" onClick={() => { autoTuneCancelRef.current = true; }}
                            style={{ background: '#ef4444', color: 'white', border: 'none', fontSize: 12 }}>
                            ⛔ Stop
                        </button>
                    )}
                    {trainResult && (
                        <button className="btn btn-secondary" onClick={() => setShowTrainResult(!showTrainResult)}
                            style={{ fontSize: 12 }}>
                            {showTrainResult ? '🔽 ซ่อนผล Train' : '📊 ดูผล Train'}
                        </button>
                    )}
                </div>
                {trainMsg && <p style={{ marginTop: 12, fontSize: 13, color: 'var(--accent-cyan)' }}>{trainMsg}</p>}
            </div>

            {/* Auto-Tune Log Panel */}
            {(autoTuning || autoTuneLogs.length > 0) && (
                <div className="card" style={{ marginBottom: 20, borderLeft: '3px solid #8b5cf6' }}>
                    <div className="card-header">
                        <span className="card-title">🔄 Auto-Tune Log {autoTuning && <span className="spinner" style={{ width: 14, height: 14, marginLeft: 8, display: 'inline-block' }} />}</span>
                        {!autoTuning && (
                            <button className="btn btn-secondary" style={{ fontSize: 11 }}
                                onClick={() => setAutoTuneLogs([])}>✕ ปิด</button>
                        )}
                    </div>
                    <div style={{ maxHeight: 300, overflowY: 'auto', padding: '8px 0' }}>
                        {autoTuneLogs.map((log, i) => (
                            <div key={i} style={{
                                display: 'flex', gap: 10, padding: '4px 0', fontSize: 13,
                                borderBottom: '1px solid rgba(255,255,255,0.05)',
                                color: log.msg.startsWith('❌') ? 'var(--accent-red)' :
                                    log.msg.startsWith('🎉') ? 'var(--accent-green)' :
                                        log.msg.startsWith('🏁') ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                            }}>
                                <span style={{ color: 'var(--text-muted)', fontSize: 11, whiteSpace: 'nowrap', minWidth: 65 }}>{log.time}</span>
                                <span>{log.msg}</span>
                            </div>
                        ))}
                    </div>
                    {/* History Chart (local auto-tune rounds) */}
                    {autoTuneHistory.length > 1 && (
                        <div style={{ padding: '12px 0', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: 'var(--text-primary)' }}>📈 Tuning Trend</div>
                            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                                <div style={{ flex: 1, minWidth: 200 }}>
                                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>MAPE (lower = better)</div>
                                    <div style={{ display: 'flex', alignItems: 'end', gap: 4, height: 60 }}>
                                        {autoTuneHistory.map((h, i) => {
                                            const mape = ((h.lstm_mape || 0) + (h.xgb_mape || 0)) / 2;
                                            const maxMape = Math.max(...autoTuneHistory.map(x => ((x.lstm_mape || 0) + (x.xgb_mape || 0)) / 2), 1);
                                            const pct = Math.min((mape / maxMape) * 100, 100);
                                            return (
                                                <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                                                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{mape.toFixed(1)}%</span>
                                                    <div style={{
                                                        width: '100%', height: `${pct}%`, minHeight: 4,
                                                        background: `linear-gradient(180deg, ${mape <= autoTuneHistory[0].lstm_mape ? '#22c55e' : '#ef4444'}, #8b5cf6)`,
                                                        borderRadius: 3
                                                    }} />
                                                    <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>R{h.round}</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                                <div style={{ flex: 1, minWidth: 200 }}>
                                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Direction Accuracy % (higher = better)</div>
                                    <div style={{ display: 'flex', alignItems: 'end', gap: 4, height: 60 }}>
                                        {autoTuneHistory.map((h, i) => {
                                            const da = ((h.lstm_dir_acc || 50) + (h.xgb_dir_acc || 50)) / 2;
                                            return (
                                                <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                                                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{da.toFixed(0)}%</span>
                                                    <div style={{
                                                        width: '100%', height: `${da}%`, minHeight: 4,
                                                        background: `linear-gradient(180deg, #06b6d4, #8b5cf6)`,
                                                        borderRadius: 3
                                                    }} />
                                                    <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>R{h.round}</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                                <div style={{ flex: 1, minWidth: 200 }}>
                                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>AI Score (higher = better)</div>
                                    <div style={{ display: 'flex', alignItems: 'end', gap: 4, height: 60 }}>
                                        {autoTuneHistory.map((h, i) => (
                                            <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                                                <span style={{ fontSize: 10, color: h.score >= 8 ? '#22c55e' : h.score >= 6 ? '#eab308' : '#ef4444' }}>{h.score}/10</span>
                                                <div style={{
                                                    width: '100%', height: `${h.score * 10}%`, minHeight: 4,
                                                    background: h.score >= 8 ? 'linear-gradient(180deg, #22c55e, #16a34a)' :
                                                        h.score >= 6 ? 'linear-gradient(180deg, #eab308, #ca8a04)' :
                                                            'linear-gradient(180deg, #ef4444, #dc2626)',
                                                    borderRadius: 3
                                                }} />
                                                <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>R{h.round}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* History Chart (API-backed) */}
                    {autoTuneHistoryApi.length > 0 && (
                        <div style={{ padding: '12px 0', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: 'var(--text-primary)' }}>
                                📊 Training History (from API)
                            </div>
                            <PlotChart
                                data={[
                                    {
                                        x: autoTuneHistoryApi.map(h => h.runLabel),
                                        y: autoTuneHistoryApi.map(h => h.avgMape),
                                        type: 'scatter',
                                        mode: 'lines+markers',
                                        name: 'Avg MAPE',
                                        line: { color: '#ef4444', width: 2 },
                                        marker: { size: 6 },
                                    },
                                    {
                                        x: autoTuneHistoryApi.map(h => h.runLabel),
                                        y: autoTuneHistoryApi.map(h => h.avgDir),
                                        type: 'scatter',
                                        mode: 'lines+markers',
                                        name: 'Avg Direction Acc',
                                        yaxis: 'y2',
                                        line: { color: '#06b6d4', width: 2 },
                                        marker: { size: 6 },
                                    },
                                ]}
                                layout={{
                                    height: 280,
                                    margin: { t: 10, b: 40, l: 55, r: 55 },
                                    xaxis: { title: 'Run', gridcolor: '#1e293b' },
                                    yaxis: { title: 'MAPE (%)', gridcolor: '#1e293b' },
                                    yaxis2: {
                                        title: 'Direction Accuracy (%)',
                                        overlaying: 'y',
                                        side: 'right',
                                        range: [0, 100],
                                    },
                                    legend: { orientation: 'h', y: 1.15 },
                                    showlegend: true,
                                }}
                                style={{ width: '100%' }}
                            />
                        </div>
                    )}
                </div>
            )}

            {/* Training Results Panel */}
            {showTrainResult && trainMetrics && (
                <div className="card" style={{ marginBottom: 20, borderLeft: '3px solid var(--accent-purple)' }}>
                    <div className="card-header">
                        <span className="card-title">📊 ผลการ Train — {selected}</span>
                        <div style={{ display: 'flex', gap: 8 }}>
                            <button className="btn btn-primary" style={{ fontSize: 11 }} onClick={handleAiTrainAnalysis} disabled={trainAiLoading}>
                                {trainAiLoading ? '⏳ AI กำลังวิเคราะห์...' : '🤖 AI วิเคราะห์ & แนะนำปรับค่า'}
                            </button>
                            <button className="btn btn-secondary" style={{ fontSize: 11 }} onClick={() => setShowTrainResult(false)}>✕ ปิด</button>
                        </div>
                    </div>

                    <div className="grid-2" style={{ gap: 16 }}>
                        {/* LSTM Results */}
                        {trainMetrics.lstm && !trainMetrics.lstm.error && (
                            <div style={{ padding: 12, background: 'var(--bg-input)', borderRadius: 'var(--radius-sm)' }}>
                                <h4 style={{ fontSize: 14, color: 'var(--accent-blue)', marginBottom: 12 }}>🔮 LSTM Model</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 12 }}>
                                    {[
                                        { label: 'Epochs', value: trainMetrics.lstm.training?.epochs_trained || '—' },
                                        { label: 'Loss', value: trainMetrics.lstm.training?.final_loss?.toFixed(6) || '—' },
                                        { label: 'Val Loss', value: trainMetrics.lstm.training?.val_loss?.toFixed(6) || '—' },
                                        { label: 'RMSE', value: trainMetrics.lstm.rmse?.toFixed(4) || '—' },
                                        { label: 'MAE', value: trainMetrics.lstm.mae?.toFixed(4) || '—' },
                                        { label: 'MAPE %', value: trainMetrics.lstm.mape?.toFixed(2) || '—' },
                                        { label: 'Direction Acc', value: trainMetrics.lstm.directional_accuracy ? `${trainMetrics.lstm.directional_accuracy.toFixed(1)}%` : '—' },
                                        { label: 'R²', value: trainMetrics.lstm.r_squared?.toFixed(4) || '—' },
                                    ].map(({ label, value }) => (
                                        <div key={label} style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
                                            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>{value}</div>
                                        </div>
                                    ))}
                                </div>

                                {/* Loss Curve */}
                                {trainMetrics.lstm.training?.loss_history && (
                                    <PlotChart
                                        data={[
                                            {
                                                y: trainMetrics.lstm.training.loss_history,
                                                x: trainMetrics.lstm.training.loss_history.map((_, i) => i + 1),
                                                type: 'scatter', mode: 'lines', name: 'Train Loss',
                                                line: { color: '#3b82f6', width: 2 },
                                            },
                                            ...(trainMetrics.lstm.training.val_loss_history ? [{
                                                y: trainMetrics.lstm.training.val_loss_history,
                                                x: trainMetrics.lstm.training.val_loss_history.map((_, i) => i + 1),
                                                type: 'scatter', mode: 'lines', name: 'Val Loss',
                                                line: { color: '#ef4444', width: 2, dash: 'dash' },
                                            }] : []),
                                        ]}
                                        layout={{
                                            height: 200, margin: { t: 10, b: 35, l: 50, r: 10 },
                                            xaxis: { title: 'Epoch', gridcolor: '#1e293b' },
                                            yaxis: { title: 'Loss', gridcolor: '#1e293b' },
                                            legend: { orientation: 'h', y: 1.15, font: { size: 10 } },
                                            showlegend: true,
                                        }}
                                        style={{ width: '100%' }}
                                    />
                                )}
                            </div>
                        )}

                        {/* XGBoost Results */}
                        {trainMetrics.xgboost && !trainMetrics.xgboost.error && (
                            <div style={{ padding: 12, background: 'var(--bg-input)', borderRadius: 'var(--radius-sm)' }}>
                                <h4 style={{ fontSize: 14, color: 'var(--accent-green)', marginBottom: 12 }}>🌲 XGBoost Model</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 12 }}>
                                    {[
                                        { label: 'RMSE', value: trainMetrics.xgboost.rmse?.toFixed(4) || '—' },
                                        { label: 'MAE', value: trainMetrics.xgboost.mae?.toFixed(4) || '—' },
                                        { label: 'MAPE %', value: trainMetrics.xgboost.mape?.toFixed(2) || '—' },
                                        { label: 'Direction Acc', value: trainMetrics.xgboost.directional_accuracy ? `${trainMetrics.xgboost.directional_accuracy.toFixed(1)}%` : '—' },
                                        { label: 'R²', value: trainMetrics.xgboost.r_squared?.toFixed(4) || '—' },
                                    ].map(({ label, value }) => (
                                        <div key={label} style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
                                            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>{value}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* AutoGluon Results */}
                        {trainMetrics.autogluon && !trainMetrics.autogluon.error && (
                            <div style={{ padding: 12, background: 'var(--bg-input)', borderRadius: 'var(--radius-sm)' }}>
                                <h4 style={{ fontSize: 14, color: 'var(--accent-cyan)', marginBottom: 12 }}>🏆 AutoGluon Model</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 12 }}>
                                    {[
                                        { label: 'RMSE', value: trainMetrics.autogluon.rmse?.toFixed(4) || '—' },
                                        { label: 'MAE', value: trainMetrics.autogluon.mae?.toFixed(4) || '—' },
                                        { label: 'MAPE %', value: trainMetrics.autogluon.mape?.toFixed(2) || '—' },
                                        { label: 'Direction Acc', value: trainMetrics.autogluon.directional_accuracy ? `${trainMetrics.autogluon.directional_accuracy.toFixed(1)}%` : '—' },
                                        { label: 'R²', value: trainMetrics.autogluon.r_squared?.toFixed(4) || '—' },
                                        { label: 'Best Model', value: trainMetrics.autogluon.training?.best_model || '—' },
                                        { label: 'Models Tried', value: trainMetrics.autogluon.training?.models_trained || '—' },
                                    ].map(({ label, value }) => (
                                        <div key={label} style={{ textAlign: 'center' }}>
                                            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
                                            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>{value}</div>
                                        </div>
                                    ))}
                                </div>
                                <div style={{ padding: 8, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                                    🏆 AutoGluon ทดลอง model หลายตัวอัตโนมัติ<br />
                                    ดู Leaderboard ได้ที่หน้า Model Performance
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Error Messages */}
                    {trainMetrics.lstm?.error && (
                        <p style={{ color: 'var(--accent-red)', fontSize: 13, marginTop: 8 }}>❌ LSTM Error: {trainMetrics.lstm.error}</p>
                    )}
                    {trainMetrics.xgboost?.error && (
                        <p style={{ color: 'var(--accent-red)', fontSize: 13, marginTop: 8 }}>❌ XGBoost Error: {trainMetrics.xgboost.error}</p>
                    )}
                    {trainMetrics.autogluon?.error && (
                        <p style={{ color: 'var(--accent-red)', fontSize: 13, marginTop: 8 }}>❌ AutoGluon Error: {trainMetrics.autogluon.error}</p>
                    )}

                    {/* AI Training Analysis */}
                    {(trainAiLoading || trainAiAnalysis) && (
                        <div style={{
                            marginTop: 16, padding: 16, borderRadius: 'var(--radius-sm)',
                            background: 'rgba(139,92,246,0.06)', border: '1px solid rgba(139,92,246,0.2)',
                        }}>
                            <h4 style={{ fontSize: 14, color: 'var(--accent-purple)', marginBottom: 10 }}>🤖 AI Analysis — แนะนำการปรับค่า</h4>
                            {trainAiLoading ? (
                                <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 0' }}>
                                    <div className="spinner" style={{ width: 18, height: 18 }} />
                                    <span style={{ color: 'var(--text-secondary)', fontSize: 13 }}>🧠 AI กำลังวิเคราะห์ผล Training และ Hyperparameters...</span>
                                </div>
                            ) : (
                                <>
                                    <div style={{ color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.9, whiteSpace: 'pre-wrap' }}>
                                        {trainAiAnalysis}
                                    </div>
                                    {aiRecommendedParams && Object.keys(aiRecommendedParams).length > 0 && (
                                        <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid rgba(139,92,246,0.2)' }}>
                                            <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 12 }}>
                                                💡 ตรวจพบค่าแนะนำที่สามารถเปลี่ยนได้โดยอัตโนมัติ:
                                            </p>
                                            <button
                                                className="btn btn-primary"
                                                onClick={handleApplyAiSettings}
                                                disabled={applySettingsLoading}
                                                style={{ fontSize: 12, display: 'inline-flex', alignItems: 'center', gap: 6 }}
                                            >
                                                {applySettingsLoading || training ? '⏳ กำลังปรับค่าและ Train...' : '⚙️ นำค่าแนะนำไปใช้ & Train ใหม่อัตโนมัติ'}
                                            </button>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                    )}
                </div>
            )}

                        {prediction && (
                <PredictionResultSection
                    prediction={prediction}
                    stocks={stocks}
                    selected={selected}
                    aiLoading={aiLoading}
                    aiExplanation={aiExplanation}
                    onRefreshAi={() => fetchAIExplanation(prediction)}
                />
            )}
        </div>
    );
}

