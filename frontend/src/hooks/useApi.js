import { useState, useEffect, useCallback } from 'react';

export function useApi(apiFn, deps = [], autoFetch = true) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(autoFetch);
    const [error, setError] = useState(null);

    const fetch = useCallback(async (...args) => {
        setLoading(true);
        setError(null);
        try {
            const res = await apiFn(...args);
            setData(res.data);
            return res.data;
        } catch (err) {
            setError(err.response?.data?.detail || err.message);
            return null;
        } finally {
            setLoading(false);
        }
    }, [apiFn]);

    useEffect(() => {
        if (autoFetch) fetch();
    }, deps);

    return { data, loading, error, fetch, setData };
}

export default useApi;
