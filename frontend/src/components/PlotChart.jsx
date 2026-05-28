import { useEffect, useRef } from 'react';

let plotlyModulePromise;
const PLOTLY_CDN_SRC = 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.4.0/plotly.min.js';

async function loadPlotly() {
    if (!plotlyModulePromise) {
        plotlyModulePromise = new Promise((resolve, reject) => {
            if (window.Plotly) {
                resolve(window.Plotly);
                return;
            }

            const existingScript = document.querySelector('script[data-plotly-cdn="true"]');
            if (existingScript) {
                existingScript.addEventListener('load', () => resolve(window.Plotly), { once: true });
                existingScript.addEventListener('error', () => reject(new Error('Plotly CDN load failed')), { once: true });
                return;
            }

            const script = document.createElement('script');
            script.src = PLOTLY_CDN_SRC;
            script.async = true;
            script.dataset.plotlyCdn = 'true';
            script.onload = () => resolve(window.Plotly);
            script.onerror = () => reject(new Error('Plotly CDN load failed'));
            document.head.appendChild(script);
        });
    }
    return plotlyModulePromise;
}

/**
 * Lightweight Plotly wrapper that works reliably with React 19 + Vite.
 * Uses plotly.js-dist-min directly instead of react-plotly.js which has
 * compatibility issues with React 19.
 */
export default function PlotChart({ data, layout, config, style, className }) {
    const ref = useRef(null);

    useEffect(() => {
        if (!ref.current || !data) return;
        let disposed = false;

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

        loadPlotly()
            .then((Plotly) => {
                if (disposed || !ref.current) return;
                return Plotly.react(ref.current, data, defaultLayout, defaultConfig);
            })
            .catch(() => {
                // Keep UI alive even if plot bundle fails to load.
            });

        return () => {
            disposed = true;
            loadPlotly().then((Plotly) => {
                if (ref.current) {
                    Plotly.purge(ref.current);
                }
            });
        };
    }, [data, layout, config]);

    return <div ref={ref} style={style} className={className} />;
}
