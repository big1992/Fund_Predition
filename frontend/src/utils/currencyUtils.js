/**
 * Currency utility for multi-market stock display.
 * Returns appropriate currency symbol and label based on stock market.
 */

/**
 * Get currency info for a stock based on its market.
 * @param {Array} stocks - Array of stock objects from API (with .market field)
 * @param {string} symbol - The stock symbol to look up
 * @returns {{ symbol: string, label: string }}
 */
export function getCurrencyInfo(stocks, symbol) {
    const stock = stocks?.find(s => s.symbol === symbol);
    const market = stock?.market || '';

    // Thai stocks (SET market) use THB
    if (market === 'SET') {
        return { symbol: '฿', label: 'THB' };
    }

    // US/International stocks use USD
    return { symbol: '$', label: 'USD' };
}
