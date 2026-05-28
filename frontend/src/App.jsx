import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import TrainingStatusBar from './components/TrainingStatusBar';
import { TrainingProvider } from './context/TrainingContext';
import './index.css';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const FundAnalysis = lazy(() => import('./pages/FundAnalysis'));
const Prediction = lazy(() => import('./pages/Prediction'));
const Portfolio = lazy(() => import('./pages/Portfolio'));
const ModelPerformance = lazy(() => import('./pages/ModelPerformance'));
const Settings = lazy(() => import('./pages/Settings'));
const NewsSentiment = lazy(() => import('./pages/NewsSentiment'));

export default function App() {
  return (
    <BrowserRouter>
      <TrainingProvider>
        <div className="app-layout">
          <Sidebar />
          <main className="main-content">
            <Suspense fallback={<div style={{ padding: 24 }}>Loading...</div>}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/analysis" element={<FundAnalysis />} />
                <Route path="/prediction" element={<Prediction />} />
                <Route path="/portfolio" element={<Portfolio />} />
                <Route path="/performance" element={<ModelPerformance />} />
                <Route path="/news" element={<NewsSentiment />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </Suspense>
          </main>
          <TrainingStatusBar />
        </div>
      </TrainingProvider>
    </BrowserRouter>
  );
}
