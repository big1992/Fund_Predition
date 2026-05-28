import { useTraining } from '../context/TrainingContext';

export default function TrainingStatusBar() {
    const { activeTasks, completedTasks, handleDismiss } = useTraining();

    const allVisible = [...activeTasks, ...completedTasks];
    if (allVisible.length === 0) return null;

    return (
        <div style={{
            position: 'fixed', bottom: 20, right: 20,
            zIndex: 9999, display: 'flex', flexDirection: 'column', gap: 8,
            maxWidth: 380, width: '100%',
        }}>
            {allVisible.map(task => {
                const isActive = task.status === 'running' || task.status === 'pending';
                const isCompleted = task.status === 'completed';
                const isFailed = task.status === 'failed';

                return (
                    <div key={task.task_id} style={{
                        background: isActive ? 'linear-gradient(135deg, #1e1b4b, #312e81)'
                            : isCompleted ? 'linear-gradient(135deg, #052e16, #166534)'
                                : 'linear-gradient(135deg, #450a0a, #991b1b)',
                        border: `1px solid ${isActive ? '#6366f1' : isCompleted ? '#22c55e' : '#ef4444'}`,
                        borderRadius: 12, padding: '12px 16px',
                        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
                        backdropFilter: 'blur(12px)',
                        animation: 'slideIn 0.3s ease-out',
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div style={{ fontSize: 13, fontWeight: 600, color: 'white' }}>
                                {isActive ? '🧠' : isCompleted ? '✅' : '❌'} Training — {task.symbol}
                            </div>
                            {!isActive && (
                                <button onClick={() => handleDismiss(task.task_id)} style={{
                                    background: 'none', border: 'none', color: 'rgba(255,255,255,0.6)',
                                    cursor: 'pointer', fontSize: 16, padding: '0 4px',
                                }}>✕</button>
                            )}
                        </div>
                        <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.8)', marginTop: 4 }}>
                            {task.message}
                        </div>
                        {isActive && (
                            <div style={{ marginTop: 8 }}>
                                <div style={{
                                    width: '100%', height: 4, background: 'rgba(255,255,255,0.1)',
                                    borderRadius: 4, overflow: 'hidden',
                                }}>
                                    <div style={{
                                        width: `${task.progress}%`, height: '100%',
                                        background: 'linear-gradient(90deg, #8b5cf6, #06b6d4)',
                                        borderRadius: 4, transition: 'width 0.5s',
                                    }} />
                                </div>
                                <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.5)', marginTop: 4, textAlign: 'right' }}>
                                    {task.progress}%
                                </div>
                            </div>
                        )}
                        {isCompleted && task.result?.metrics && (
                            <div style={{ marginTop: 6, fontSize: 11, color: 'rgba(255,255,255,0.7)' }}>
                                {Object.entries(task.result.metrics).map(([sym, m]) => (
                                    <span key={sym}>
                                        {sym}: MAPE {m?.lstm?.mape || m?.xgboost?.mape || '?'}%
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>
                );
            })}
            <style>{`
                @keyframes slideIn {
                    from { transform: translateX(100px); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `}</style>
        </div>
    );
}
