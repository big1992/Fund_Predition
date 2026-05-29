import axios from 'axios';

const api = axios.create({
    baseURL: '/api',
    timeout: 300000,  // 5 minutes default
    headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(
    (res) => res,
    (err) => {
        console.error('API Error:', err.response?.data || err.message);
        return Promise.reject(err);
    }
);

// ===== Stock APIs =====
export const getStocks = () => api.get('/stocks');
export const getStockPrices = (symbol, start, end) =>
    api.get(`/stocks/${symbol}/prices`, { params: { start, end } });
export const getIndicators = (symbol) => api.get(`/stocks/${symbol}/indicators`);
export const collectData = (symbols = [], period = '5y') =>
    api.post('/stocks/collect', { symbols, period });

// ===== Fund APIs =====
export const getFunds = () => api.get('/funds');
export const getFundNAV = (fundName) => api.get(`/funds/${fundName}/nav`);

// ===== Prediction APIs =====
export const getPrediction = (symbol, model = 'ensemble') =>
    api.get(`/predictions/${symbol}`, { params: { model } });
export const getWatchlistAlerts = (symbol, limit = 20) => api.get(`/predictions/alerts/${symbol}`, { params: { limit } });
export const exportResearchReportCsv = (symbol) => api.get(`/predictions/reports/${symbol}/csv`, { responseType: 'blob' });
export const exportResearchReportPdf = (symbol) => api.get(`/predictions/reports/${symbol}/pdf`, { responseType: 'blob' });
export const trainModels = (symbols = [], model_type = 'all') =>
    api.post('/predictions/train', { symbols, model_type }, { timeout: 600000 });  // 10 min for training
export const getModelPerformance = () => api.get('/predictions/models/performance');
export const getValidationReportCard = (symbol) => api.get(`/predictions/models/report-card/${symbol}`);
export const getModelRegistry = (symbol, limit = 100) => api.get(`/predictions/models/registry/${symbol}`, { params: { limit } });
export const getModelDrift = (symbol) => api.get(`/predictions/models/drift/${symbol}`);
export const getFeatureImportance = (symbol) => api.get(`/predictions/models/features/${symbol}`);
export const getBestTrainingRun = (symbol) => api.get(`/predictions/models/best-run/${symbol}`);
export const getTrainingHistory = (symbol, limit = 10) => api.get(`/predictions/models/history/${symbol}`, { params: { limit } });
export const getAutogluonLeaderboard = (symbol) => api.get(`/predictions/models/autogluon-leaderboard/${symbol}`);

// ===== Portfolio APIs =====
export const optimizePortfolio = (symbols, method = 'max_sharpe', risk_level = 0.5) =>
    api.post('/portfolio/optimize', { symbols, method, risk_level });

// ===== Backtest APIs =====
export const runBacktest = (symbol, model_type = 'ensemble', initial_capital = 1000000) =>
    api.post('/backtest/run', { symbol, model_type, initial_capital });

// ===== AI Explanation APIs =====
export const explainPrediction = (data) => api.post('/ai/explain-prediction', data);
export const explainPerformance = (data) => api.post('/ai/explain-performance', data);
export const explainTraining = (data) => api.post('/ai/explain-training', data);

// ===== Settings APIs =====
export const getSettings = () => api.get('/settings');
export const updateSettings = (data) => api.put('/settings', data);

// ===== News & Sentiment APIs =====
export const collectNews = (symbol) => api.post(`/news/collect/${symbol}`, {}, { timeout: 60000 });
export const getNews = (symbol, limit = 50, period = '30d') => api.get(`/news/${symbol}`, { params: { limit, period } });
export const getNewsSummary = (symbol, period = '7d') => api.get(`/news/${symbol}/summary`, { params: { period } });
export const getDailySentiment = (symbol, days = 30) => api.get(`/news/${symbol}/daily`, { params: { days } });

// ===== Background Training APIs =====
export const trainModelsAsync = (data) => api.post('/predictions/train/async', data);
export const getTrainStatus = (taskId) => api.get(`/predictions/train/status/${taskId}`);
export const getActiveTasks = () => api.get('/predictions/train/active');
export const cancelTrainTask = (taskId) => api.post(`/predictions/train/cancel/${taskId}`);
export const dismissTask = (taskId) => api.delete(`/predictions/train/dismiss/${taskId}`);

// ===== System APIs =====
export const getHealth = () => api.get('/health');
export const getSchedulerStatus = () => api.get('/scheduler/status');
export const startScheduler = (interval_hours = 168) => api.post('/scheduler/start', null, { params: { interval_hours } });
export const stopScheduler = () => api.post('/scheduler/stop');
export const retrainNow = () => api.post('/scheduler/retrain-now', {}, { timeout: 600000 });

export default api;
