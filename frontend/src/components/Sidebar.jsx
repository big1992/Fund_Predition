import { NavLink } from 'react-router-dom';
import { MdDashboard, MdAnalytics, MdSmartToy, MdAccountBalance, MdBarChart, MdSettings, MdArticle } from 'react-icons/md';

const navItems = [
    { to: '/', icon: <MdDashboard />, label: 'Dashboard' },
    { to: '/analysis', icon: <MdAnalytics />, label: 'Fund Analysis' },
    { to: '/prediction', icon: <MdSmartToy />, label: 'Prediction' },
    { to: '/portfolio', icon: <MdAccountBalance />, label: 'Portfolio' },
    { to: '/performance', icon: <MdBarChart />, label: 'Model Performance' },
    { to: '/news', icon: <MdArticle />, label: 'News & Sentiment' },
    { to: '/settings', icon: <MdSettings />, label: 'Settings' },
];

export default function Sidebar() {
    return (
        <aside className="sidebar">
            <div className="sidebar-brand">
                <div className="sidebar-brand-icon">📊</div>
                <div>
                    <h1>FundPredict</h1>
                    <span>Thai Equity Fund AI</span>
                </div>
            </div>
            <nav className="sidebar-nav">
                {navItems.map(({ to, icon, label }) => (
                    <NavLink
                        key={to}
                        to={to}
                        end={to === '/'}
                        className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                    >
                        <span className="icon">{icon}</span>
                        {label}
                    </NavLink>
                ))}
            </nav>
            <div style={{ padding: '16px', borderTop: '1px solid var(--border-color)', marginTop: 'auto' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    Fund Prediction System v1.0<br />
                    LSTM • XGBoost • Ensemble
                </div>
            </div>
        </aside>
    );
}
