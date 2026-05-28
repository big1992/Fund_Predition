import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

/**
 * Lightweight Plotly wrapper that works reliably with React 19 + Vite.
 * Uses plotly.js-dist-min directly instead of react-plotly.js which has
 * compatibility issues with React 19.
 */
export default function PlotChart({ data, layout, config, style, className }) {
    const ref = useRef(null);

    useEffect(() => {
        if (!ref.current || !data) return;

        const defaultLayout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: '#0f1729',
            font: { color: '#94a3b8', family: 'Inter' },
            margin: { t: 20, b: 40, l: 60, r: 20 },
            ...layout,
        };

        const defaultConfig = {
            responsive: true,
            displayModeBar: false,
            ...config,
        };

        Plotly.react(ref.current, data, defaultLayout, defaultConfig);

        return () => {
            if (ref.current) {
                Plotly.purge(ref.current);
            }
        };
    }, [data, layout, config]);

    return <div ref={ref} style={style} className={className} />;
}
