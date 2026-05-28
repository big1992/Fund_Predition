import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import TrainingStatusBar from './components/TrainingStatusBar';
import { TrainingProvider } from './context/TrainingContext';
import Dashboard from './pages/Dashboard';
import FundAnalysis from './pages/FundAnalysis';
import Prediction from './pages/Prediction';
import Portfolio from './pages/Portfolio';
import ModelPerformance from './pages/ModelPerformance';
import Settings from './pages/Settings';
import NewsSentiment from './pages/NewsSentiment';
import './index.css';

export default function App() {
  return (
    <BrowserRouter>
      <TrainingProvider>
        <div className="app-layout">
          <Sidebar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analysis" element={<FundAnalysis />} />
              <Route path="/prediction" element={<Prediction />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/performance" element={<ModelPerformance />} />
              <Route path="/news" element={<NewsSentiment />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>
          <TrainingStatusBar />
        </div>
      </TrainingProvider>
    </BrowserRouter>
  );
}

