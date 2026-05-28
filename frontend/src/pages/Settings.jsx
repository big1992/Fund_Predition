import { useState, useEffect } from 'react';
import { getSettings, updateSettings } from '../api/client';

export default function Settings() {
    const [settings, setSettings] = useState(null);
    const [form, setForm] = useState({});
    const [saving, setSaving] = useState(false);
    const [msg, setMsg] = useState('');
    const [apiKeyInput, setApiKeyInput] = useState('');

    useEffect(() => { loadSettings(); }, []);

    const loadSettings = async () => {
        try {
            const res = await getSettings();
            setSettings(res.data);
            setForm(res.data);
        } catch { setMsg('❌ ไม่สามารถโหลด Settings ได้'); }
    };

    const handleChange = (key, value, type = 'text') => {
        let parsed = value;
        if (type === 'number') parsed = parseFloat(value) || 0;
        if (type === 'int') parsed = parseInt(value) || 0;
        setForm(prev => ({ ...prev, [key]: parsed }));
    };

    const handleSave = async () => {
        setSaving(true); setMsg('');
        try {
            const payload = { ...form };
            delete payload.app_name; delete payload.app_version;
            delete payload.debug; delete payload.openai_api_key_set;
            if (apiKeyInput) payload.openai_api_key = apiKeyInput;
            const res = await updateSettings(payload);
            setSettings(res.data); setForm(res.data);
            setMsg('✅ บันทึกสำเร็จ! การเปลี่ยนแปลงมีผลทันที (ต้อง Train ใหม่จึงจะเห็นผลกับ Model)');
            setApiKeyInput('');
        } catch (err) {
            setMsg(`❌ ${err.response?.data?.detail || err.message}`);
        } finally { setSaving(false); }
    };

    const handleReset = () => {
        if (settings) setForm({ ...settings });
        setApiKeyInput(''); setMsg('');
    };

    if (!settings) return <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>Loading...</div>;

    const Section = ({ title, icon, description, children }) => (
        <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-header"><span className="card-title">{icon} {title}</span></div>
            {description && (
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16, lineHeight: 1.7, padding: '0 4px' }}>
                    {description}
                </p>
            )}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 14 }}>
                {children}
            </div>
        </div>
    );

    const Field = ({ label, keyName, type = 'number', step, min, max, desc }) => (
        <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">{label}</label>
            <input
                className="form-input"
                type={type === 'int' ? 'number' : type}
                step={step || (type === 'number' ? '0.01' : type === 'int' ? '1' : undefined)}
                min={min} max={max}
                value={form[keyName] ?? ''}
                onChange={e => handleChange(keyName, e.target.value, type)}
            />
            {desc && <span style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3, display: 'block', lineHeight: 1.5 }}>{desc}</span>}
        </div>
    );

    return (
        <div>
            <div className="page-header">
                <h2>Settings</h2>
                <p>ตั้งค่า Hyperparameters, Backtest, Portfolio และ API Keys — เปลี่ยนแล้วต้อง Train Model ใหม่ถึงจะมีผล</p>
            </div>

            {/* Action Bar */}
            <div className="card" style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
                    {saving ? '⏳ Saving...' : '💾 บันทึก Settings'}
                </button>
                <button className="btn btn-secondary" onClick={handleReset}>🔄 Reset</button>
                <div style={{ flex: 1 }} />
                <span style={{
                    padding: '6px 14px', borderRadius: 'var(--radius-sm)', fontSize: 12,
                    background: 'var(--bg-input)', color: 'var(--text-muted)',
                }}>
                    v{settings.app_version} • {settings.debug ? '🟡 Debug' : '🟢 Production'}
                </span>
            </div>

            {/* Info Banner */}
            <div style={{
                padding: '12px 18px', marginBottom: 16, borderRadius: 'var(--radius-sm)', fontSize: 13,
                background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.2)',
                color: 'var(--text-secondary)', lineHeight: 1.7,
            }}>
                💡 <strong>หมายเหตุ:</strong> การเปลี่ยน Hyperparameters (LSTM, XGBoost, Ensemble) จะมีผลเมื่อ <strong>Train Model ใหม่</strong> เท่านั้น
                — ส่วน Backtest/Portfolio มีผลทันทีเมื่อกด Run ครั้งถัดไป
            </div>

            {msg && (
                <div style={{
                    padding: '10px 16px', marginBottom: 16, borderRadius: 'var(--radius-sm)', fontSize: 13,
                    background: msg.startsWith('✅') ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
                    color: msg.startsWith('✅') ? 'var(--accent-green)' : 'var(--accent-red)',
                    border: `1px solid ${msg.startsWith('✅') ? 'var(--accent-green)' : 'var(--accent-red)'}33`,
                }}>
                    {msg}
                </div>
            )}

            {/* OpenAI */}
            <div className="card" style={{ marginBottom: 16, borderLeft: '3px solid var(--accent-cyan)' }}>
                <div className="card-header"><span className="card-title">🧠 OpenAI API</span></div>
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.7 }}>
                    ใช้สำหรับ AI Analysis ในหน้า Prediction และ Model Performance — ถ้าไม่ตั้งค่า ระบบจะทำงานปกติแต่ไม่มีคำอธิบายจาก AI
                </p>
                <div style={{ display: 'flex', gap: 12, alignItems: 'end', flexWrap: 'wrap' }}>
                    <div className="form-group" style={{ marginBottom: 0, flex: 1, minWidth: 300 }}>
                        <label className="form-label">API Key</label>
                        <input className="form-input" type="password"
                            placeholder={settings.openai_api_key_set ? '••••••••••• (key is set)' : 'sk-proj-...'}
                            value={apiKeyInput} onChange={e => setApiKeyInput(e.target.value)}
                        />
                    </div>
                    <span style={{
                        padding: '6px 14px', borderRadius: 'var(--radius-sm)', fontSize: 12,
                        background: settings.openai_api_key_set ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
                        color: settings.openai_api_key_set ? 'var(--accent-green)' : 'var(--accent-red)',
                    }}>
                        {settings.openai_api_key_set ? '🟢 Connected' : '🔴 Not Set'}
                    </span>
                </div>
            </div>

            {/* LSTM */}
            <Section title="LSTM Model" icon="🔮"
                description="LSTM (Long Short-Term Memory) เป็น Neural Network ที่เก่งเรื่องข้อมูล Time Series — เหมาะสำหรับจับรูปแบบราคาที่มีแนวโน้ม (trend) ค่าที่ปรับจะมีผลเมื่อ Train ใหม่">
                <Field label="Sequence Length" keyName="lstm_sequence_length" type="int" min={10} max={200}
                    desc="จำนวนวันย้อนหลังที่ model ดูเพื่อทำนาย — ค่ามาก = จำรูปแบบระยะยาวได้ดี แต่ train ช้าลง" />
                <Field label="LSTM Units Layer 1" keyName="lstm_units_1" type="int" min={16} max={512}
                    desc="จำนวน neurons ชั้น 1 — ค่ามาก = เรียนรู้ pattern ซับซ้อนได้ดี แต่เสี่ยง overfitting" />
                <Field label="LSTM Units Layer 2" keyName="lstm_units_2" type="int" min={16} max={256}
                    desc="จำนวน neurons ชั้น 2 — ควรน้อยกว่าชั้น 1 เพื่อ compress ข้อมูล" />
                <Field label="Dropout" keyName="lstm_dropout" step="0.05" min={0} max={0.8}
                    desc="สัดส่วนที่สุ่มปิด neurons ขณะ train — ป้องกัน overfitting (0.2-0.3 คือค่าทั่วไป)" />
                <Field label="Dense Units" keyName="lstm_dense_units" type="int" min={8} max={128}
                    desc="จำนวน neurons ชั้น Dense ก่อนผลลัพธ์ — ยิ่งมาก ยิ่งซับซ้อน" />
                <Field label="Learning Rate" keyName="lstm_learning_rate" step="0.0001" min={0.00001} max={0.1}
                    desc="ความเร็วการเรียนรู้ — ค่าน้อย = เรียนรู้ละเอียดแต่ช้า, ค่ามาก = เร็วแต่อาจ overshoot" />
                <Field label="Epochs" keyName="lstm_epochs" type="int" min={10} max={500}
                    desc="จำนวนรอบ train ทั้งหมด — มากเกินอาจ overfitting (Early Stopping จะช่วยหยุดอัตโนมัติ)" />
                <Field label="Batch Size" keyName="lstm_batch_size" type="int" min={8} max={128}
                    desc="จำนวนตัวอย่างต่อรอบ — ค่าเล็ก = train ละเอียดแต่ช้า, ค่าใหญ่ = เร็วแต่หยาบ" />
                <Field label="Early Stopping Patience" keyName="lstm_early_stopping" type="int" min={3} max={50}
                    desc="หยุด train อัตโนมัติถ้า validation loss ไม่ดีขึ้นกี่ epoch ติดต่อกัน — ป้องกัน overfitting" />
                <Field label="Prediction Days" keyName="lstm_prediction_days" type="int" min={1} max={30}
                    desc="จำนวนวันที่ทำนายล่วงหน้า — ยิ่งมาก ยิ่งไม่แม่นยำ (แนะนำ 3-7 วัน)" />
            </Section>

            {/* XGBoost */}
            <Section title="XGBoost Model" icon="🌲"
                description="XGBoost เป็น Gradient Boosted Trees ที่เร็วและแม่นยำ — เก่งเรื่องจับ Feature Importance และข้อมูลที่ไม่เป็น linear ค่าที่ปรับจะมีผลเมื่อ Train ใหม่">
                <Field label="N Estimators" keyName="xgb_n_estimators" type="int" min={50} max={2000}
                    desc="จำนวนต้นไม้ทั้งหมด — มาก = แม่นขึ้นแต่ช้าลง, น้อยเกิน = underfit" />
                <Field label="Max Depth" keyName="xgb_max_depth" type="int" min={2} max={15}
                    desc="ความลึกสูงสุดของแต่ละต้นไม้ — ลึก = จับ pattern ซับซ้อนได้ แต่เสี่ยง overfit (3-8 ทั่วไป)" />
                <Field label="Learning Rate" keyName="xgb_learning_rate" step="0.001" min={0.001} max={0.3}
                    desc="น้ำหนักที่แต่ละต้นไม้ใหม่เพิ่ม — ค่าน้อย + ต้นไม้เยอะ = ผลลัพธ์ดีแต่ช้า" />
                <Field label="Subsample" keyName="xgb_subsample" step="0.05" min={0.5} max={1.0}
                    desc="สัดส่วนข้อมูลที่สุ่มใช้ train แต่ละต้นไม้ — ลด overfitting (0.7-0.9 แนะนำ)" />
                <Field label="Col Sample by Tree" keyName="xgb_colsample_bytree" step="0.05" min={0.5} max={1.0}
                    desc="สัดส่วน features ที่สุ่มใช้แต่ละต้นไม้ — ลด overfitting เหมือน Subsample" />
                <Field label="Reg Alpha (L1)" keyName="xgb_reg_alpha" step="0.1" min={0} max={10}
                    desc="L1 Regularization — ทำให้ feature ที่ไม่สำคัญมีค่า weight เป็น 0 (feature selection)" />
                <Field label="Reg Lambda (L2)" keyName="xgb_reg_lambda" step="0.1" min={0} max={10}
                    desc="L2 Regularization — ลดขนาด weight ทุกตัว ป้องกัน overfitting (ค่ามาก = conservative)" />
            </Section>

            {/* Ensemble */}
            <Section title="Ensemble Weights" icon="⚖️"
                description="Ensemble รวมผลทำนายจาก LSTM + XGBoost โดยถ่วงน้ำหนัก — ผลรวมต้อง = 1.0 ค่าที่ปรับมีผลทันทีกับ Prediction">
                <Field label="LSTM Weight" keyName="ensemble_lstm_weight" step="0.05" min={0} max={1}
                    desc="น้ำหนักของ LSTM — ค่ามาก = เน้น trend/pattern ระยะยาว" />
                <Field label="XGBoost Weight" keyName="ensemble_xgb_weight" step="0.05" min={0} max={1}
                    desc="น้ำหนักของ XGBoost — ค่ามาก = เน้น feature-based decision" />
            </Section>

            {/* Data */}
            <Section title="Data Split" icon="📊"
                description="สัดส่วนการแบ่งข้อมูลสำหรับ Train/Validate/Test — ผลรวมต้อง = 1.0 มีผลเมื่อ Train ใหม่">
                <Field label="Train Ratio" keyName="data_train_ratio" step="0.05" min={0.5} max={0.9}
                    desc="สัดส่วนข้อมูลสำหรับ train — มาก = เรียนรู้จากข้อมูลเยอะ แต่ test น้อยลง" />
                <Field label="Validation Ratio" keyName="data_val_ratio" step="0.05" min={0.05} max={0.3}
                    desc="สัดส่วนสำหรับ validation — ใช้ตรวจ overfitting ระหว่าง train (Early Stopping)" />
                <Field label="Test Ratio" keyName="data_test_ratio" step="0.05" min={0.05} max={0.3}
                    desc="สัดส่วนข้อมูลทดสอบ — ใช้วัดผลจริงหลัง train เสร็จ (ไม่เคยเห็นข้อมูลนี้)" />
            </Section>

            {/* Backtest */}
            <Section title="Backtest" icon="📉"
                description="จำลองการเทรดย้อนหลังจากสัญญาณ model — ใช้ประเมินว่ากลยุทธ์ทำกำไรได้จริงหรือไม่ มีผลทันทีเมื่อกด Run Backtest">
                <Field label="Initial Capital (THB)" keyName="bt_initial_capital" type="int" min={100000} max={100000000}
                    desc="เงินทุนเริ่มต้น — ไม่กระทบ win rate แต่กระทบขนาด trades" />
                <Field label="Buy Threshold" keyName="bt_buy_threshold" step="0.005" min={0.001} max={0.1}
                    desc="ถ้า model ทำนายราคาขึ้น ≥ X% → สัญญาณ BUY (ค่าน้อย = เทรดบ่อย, ค่ามาก = เทรดน้อยแต่คัดกรองมากขึ้น)" />
                <Field label="Sell Threshold" keyName="bt_sell_threshold" step="0.005" min={-0.1} max={-0.001}
                    desc="ถ้า model ทำนายราคาลง ≤ X% → สัญญาณ SELL (เช่น -0.005 = ลง 0.5% ก็ขาย)" />
                <Field label="Commission" keyName="bt_commission" step="0.0005" min={0} max={0.01}
                    desc="ค่าคอมมิชชั่นต่อ trade (0.001 = 0.1%) — กระทบ net profit โดยตรง" />
            </Section>

            {/* Portfolio */}
            <Section title="Portfolio Optimization" icon="💼"
                description="ปรับสัดส่วนการลงทุนให้ optimal โดยใช้ Modern Portfolio Theory (MPT) — มีผลทันทีเมื่อกด Optimize">
                <Field label="Min Weight" keyName="pf_min_weight" step="0.01" min={0} max={0.3}
                    desc="น้ำหนักขั้นต่ำต่อหุ้น — ป้องกันการกระจุกตัว (0 = อนุญาตไม่ถือบางตัว)" />
                <Field label="Max Weight" keyName="pf_max_weight" step="0.05" min={0.1} max={1}
                    desc="น้ำหนักสูงสุดต่อหุ้น — จำกัดความเสี่ยงจากตัวเดียว (0.4 = ถือได้สูงสุด 40%)" />
                <Field label="Risk-Free Rate" keyName="pf_risk_free_rate" step="0.005" min={0} max={0.1}
                    desc="อัตราผลตอบแทนไร้ความเสี่ยง (พันธบัตรรัฐบาล) — ใช้คำนวณ Sharpe Ratio" />
                <Field label="Random Portfolios" keyName="pf_num_portfolios" type="int" min={500} max={50000}
                    desc="จำนวน portfolios ที่สุ่มใน Monte Carlo — มาก = ผลแม่นขึ้นแต่ช้าลง" />
            </Section>
        </div>
    );
}
