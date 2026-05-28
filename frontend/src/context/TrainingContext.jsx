import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { getActiveTasks, dismissTask, cancelTrainTask } from '../api/client';

const TrainingContext = createContext();

export function TrainingProvider({ children }) {
    const [tasks, setTasks] = useState([]);         // All active + recent completed
    const [completedTasks, setCompletedTasks] = useState([]);  // Tasks awaiting dismissal
    const pollingRef = useRef(null);

    // Poll active tasks every 3 seconds
    const poll = useCallback(async () => {
        try {
            const res = await getActiveTasks();
            const active = res.data?.tasks || [];
            const all = res.data?.all || [];

            setTasks(all);

            // Detect newly completed tasks
            const justCompleted = all.filter(
                t => (t.status === 'completed' || t.status === 'failed' || t.status === 'cancelled')
                    && !completedTasks.find(c => c.task_id === t.task_id)
            );
            if (justCompleted.length > 0) {
                setCompletedTasks(prev => [...prev, ...justCompleted]);
            }

            // If there are active tasks, keep polling
            if (active.length > 0) {
                pollingRef.current = setTimeout(poll, 3000);
            }
            // else: stop polling — will resume via startPolling() on next train
        } catch {
            // On error, retry once after 10s then stop
            pollingRef.current = setTimeout(poll, 10000);
        }
    }, [completedTasks]);

    useEffect(() => {
        poll();
        return () => {
            if (pollingRef.current) clearTimeout(pollingRef.current);
        };
    }, []);

    // Re-trigger fast polling when new tasks detected
    const startPolling = useCallback(() => {
        if (pollingRef.current) clearTimeout(pollingRef.current);
        pollingRef.current = setTimeout(poll, 1000);
    }, [poll]);

    const handleDismiss = useCallback(async (taskId) => {
        try {
            await dismissTask(taskId);
        } catch { }
        setCompletedTasks(prev => prev.filter(t => t.task_id !== taskId));
        setTasks(prev => prev.filter(t => t.task_id !== taskId));
    }, []);

    const handleCancel = useCallback(async (taskId) => {
        try {
            await cancelTrainTask(taskId);
            setTasks(prev => prev.map(t => t.task_id === taskId ? { ...t, status: 'cancelling', message: 'Cancellation requested...' } : t));
        } catch { }
    }, []);

    const activeTasks = tasks.filter(t => t.status === 'running' || t.status === 'pending' || t.status === 'cancelling');

    return (
        <TrainingContext.Provider value={{ tasks, activeTasks, completedTasks, startPolling, handleDismiss, handleCancel }}>
            {children}
        </TrainingContext.Provider>
    );
}

export function useTraining() {
    return useContext(TrainingContext);
}
