import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// --- Server Initialization ---
const server = new McpServer({
    name: "math-mcp-server",
    version: "1.0.0",
    capabilities: {
        tools: {}, // Declare that we provide tools
    },
});

console.error("Calculator MCP Server starting..."); // Log to stderr

// --- Helper Functions ---
function calculateFactorial(n: number): number {
    if (n < 0 || !Number.isInteger(n)) {
        throw new Error("Factorial is only defined for non-negative integers.");
    }
    if (n > 170) {
        // Avoid exceeding MAX_SAFE_INTEGER or performance issues
        throw new Error("Input too large for factorial calculation.");
    }
    if (n === 0 || n === 1) {
        return 1;
    }
    let result = 1;
    for (let i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

/**
 * Performs a simple exponential smoothing forecast
 * @param historicalData Array of historical expense values
 * @param alpha Smoothing factor (0-1)
 * @param periodsToForecast Number of periods to forecast
 * @returns Array of forecasted values
 */
function exponentialSmoothingForecast(
    historicalData: number[], 
    alpha: number = 0.3, 
    periodsToForecast: number = 1
): number[] {
    if (historicalData.length === 0) {
        throw new Error("Historical data cannot be empty");
    }
    
    // Initialize with first value
    let lastSmoothedValue = historicalData[0];
    
    // Calculate smoothed values based on historical data
    for (let i = 1; i < historicalData.length; i++) {
        lastSmoothedValue = alpha * historicalData[i] + (1 - alpha) * lastSmoothedValue;
    }
    
    // Generate forecast for future periods
    const forecast: number[] = [];
    for (let i = 0; i < periodsToForecast; i++) {
        forecast.push(lastSmoothedValue);
    }
    
    return forecast;
}

/**
 * ARIMA (AutoRegressive Integrated Moving Average) forecasting implementation
 * 
 * @param historicalData Array of historical values
 * @param p AR (Auto-Regressive) order
 * @param d I (Integrated/Differencing) order
 * @param q MA (Moving Average) order
 * @param periodsToForecast Number of periods to forecast
 * @returns Array of forecasted values
 */
function arimaForecast(
    historicalData: number[],
    p: number = 1,
    d: number = 1,
    q: number = 1,
    periodsToForecast: number = 1
): number[] {
    if (historicalData.length < p + d + q + 1) {
        throw new Error(`Insufficient data for ARIMA(${p},${d},${q}). Need at least ${p + d + q + 1} observations.`);
    }

    // Step 1: Apply differencing 'd' times
    let diffData = [...historicalData];
    for (let i = 0; i < d; i++) {
        diffData = difference(diffData);
    }

    // Step 2: Estimate AR coefficients using Yule-Walker equations
    const arCoefficients = estimateArCoefficients(diffData, p);
    
    // Step 3: Estimate MA coefficients using method of moments
    const maCoefficients = estimateMaCoefficients(diffData, q);
    
    // Step 4: Generate forecasts
    const forecast: number[] = [];
    let workingData = [...diffData];
    
    for (let i = 0; i < periodsToForecast; i++) {
        // Calculate next value using ARIMA equation
        let nextValue = 0;
        
        // AR component
        for (let j = 0; j < p; j++) {
            if (workingData.length > j) {
                nextValue += arCoefficients[j] * workingData[workingData.length - 1 - j];
            }
        }
        
        // MA component - use residuals (simplified)
        // In a full implementation, we would calculate residuals from fitted values
        const recentResiduals = calculateResiduals(workingData, workingData.length - q);
        for (let j = 0; j < q; j++) {
            if (j < recentResiduals.length) {
                nextValue += maCoefficients[j] * recentResiduals[j];
            }
        }
        
        // Add to working data for next iteration
        workingData.push(nextValue);
        
        // Prepare the forecasted value (reverse differencing)
        let forecastedValue = nextValue;
        let tempData = [...workingData];
        
        // Reverse differencing (integrate)
        for (let j = 0; j < d; j++) {
            forecastedValue = integrateValue(forecastedValue, tempData, historicalData, j);
            // Update tempData for next level of integration if needed
            if (j < d - 1) {
                tempData = integrate(tempData, historicalData, j);
            }
        }
        
        forecast.push(forecastedValue);
    }
    
    return forecast;
}

/**
 * Calculate the first difference of a time series
 */
function difference(data: number[]): number[] {
    const result: number[] = [];
    for (let i = 1; i < data.length; i++) {
        result.push(data[i] - data[i - 1]);
    }
    return result;
}

/**
 * Integrate a value (reverse differencing)
 */
function integrateValue(value: number, diffData: number[], originalData: number[], level: number): number {
    if (level === 0) {
        // First level of integration
        return value + originalData[originalData.length - 1];
    } else {
        // Higher levels of integration
        const previousLevelData = integrate(diffData.slice(0, -1), originalData, level - 1);
        return value + previousLevelData[previousLevelData.length - 1];
    }
}

/**
 * Integrate a differenced series (reverse differencing)
 */
function integrate(diffData: number[], originalData: number[], level: number): number[] {
    const result: number[] = [];
    
    if (level === 0) {
        // Base case: first level of integration
        result.push(originalData[0]); // Starting point
        for (let i = 0; i < diffData.length; i++) {
            result.push(result[i] + diffData[i]);
        }
    } else {
        // Recursive case: higher levels of integration
        const previousLevelData = integrate(difference(diffData), originalData, level - 1);
        result.push(previousLevelData[0]); // Starting point
        for (let i = 0; i < diffData.length; i++) {
            result.push(result[i] + diffData[i]);
        }
    }
    
    return result;
}

/**
 * Estimate AR coefficients using Yule-Walker equations (simplified)
 */
function estimateArCoefficients(data: number[], p: number): number[] {
    if (p === 0) return [];
    
    // Calculate autocorrelations
    const acf = calculateAutocorrelation(data, p);
    
    // Solve Yule-Walker equations using a simplified approach
    // For a full implementation, we would use matrix operations
    const coefficients: number[] = Array(p).fill(0);
    
    // Simple AR(1) approximation if p=1
    if (p === 1) {
        coefficients[0] = acf[1] / acf[0];
        return coefficients;
    }
    
    // Simple AR(2) approximation if p=2
    if (p === 2) {
        const denominator = 1 - acf[1] * acf[1];
        coefficients[0] = (acf[1] * (1 - acf[2]) + acf[1]) / denominator;
        coefficients[1] = (acf[2] - acf[1] * acf[1]) / denominator;
        return coefficients;
    }
    
    // For higher orders, use a simplification
    // This is a very simplified approach; a full implementation would use the Levinson-Durbin algorithm
    for (let i = 0; i < p; i++) {
        coefficients[i] = acf[i + 1] / acf[0];
    }
    
    return coefficients;
}

/**
 * Estimate MA coefficients (simplified approach)
 */
function estimateMaCoefficients(data: number[], q: number): number[] {
    if (q === 0) return [];
    
    // For simplicity, we'll use a heuristic approach
    // In a full implementation, this would involve iterative methods
    const coefficients: number[] = Array(q).fill(0);
    
    // Simple approach: decay coefficients
    for (let i = 0; i < q; i++) {
        coefficients[i] = 0.5 / (i + 1);
    }
    
    return coefficients;
}

/**
 * Calculate autocorrelation
 */
function calculateAutocorrelation(data: number[], maxLag: number): number[] {
    const acf: number[] = [];
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    
    // Denominator (variance)
    let denominator = 0;
    for (let i = 0; i < data.length; i++) {
        denominator += Math.pow(data[i] - mean, 2);
    }
    
    // Calculate autocorrelation for each lag
    for (let lag = 0; lag <= maxLag; lag++) {
        let numerator = 0;
        for (let i = 0; i < data.length - lag; i++) {
            numerator += (data[i] - mean) * (data[i + lag] - mean);
        }
        acf.push(numerator / denominator);
    }
    
    return acf;
}

/**
 * Calculate residuals
 */
function calculateResiduals(data: number[], startIndex: number): number[] {
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    return data.slice(Math.max(0, startIndex)).map(value => value - mean);
}

/**
 * Calculate variance of a dataset
 */
function calculateVariance(data: number[]): number {
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
}

// --- Expense Schema ---
// Enhanced schema that supports all features from both files
const ExpenseSchema = z.object({
    id: z.string(),
    amount: z.number(),
    amount_with_tax: z.number().optional(),
    createdAt: z.string().refine(
        (val) => {
            // Accept ISO 8601 datetime or YYYY-MM-DD date
            return (
                /^\d{4}-\d{2}-\d{2}$/.test(val) ||
                /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$/.test(val)
            );
        },
        { message: 'Invalid date or datetime format' }
    ).optional(),
    category: z.string().optional(),
    account_name: z.string().optional(),
    description: z.string().optional(),
    created_time: z.string().refine(
        (val) => {
            // Accept ISO 8601 datetime or YYYY-MM-DD date
            return (
                /^\d{4}-\d{2}-\d{2}$/.test(val) ||
                /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$/.test(val)
            );
        },
        { message: 'Invalid date or datetime format' }
    ).optional(),
});
const ExpensesArraySchema = z.array(ExpenseSchema).describe("Array of expense objects");

// --- Expense Analytics Tools ---
// Version 1 tool
// Utility: Normalize incoming expenses to match schema
function normalizeExpenses(rawExpenses: any[]): any[] {
    return rawExpenses.map((exp) => ({
        id: exp.id || exp.expense_id || '',
        amount: exp.amount !== undefined ? exp.amount : (exp.total !== undefined ? exp.total : 0),
        amount_with_tax: exp.amount_with_tax !== undefined ? exp.amount_with_tax : (exp.total_with_tax !== undefined ? exp.total_with_tax : undefined),
        createdAt: exp.createdAt || exp.date || exp.created_time || '',
        category: exp.category,
        account_name: exp.account_name,
        description: exp.description,
        created_time: exp.created_time || exp.date || exp.createdAt || '',
    }));
}

server.tool(
    "calculateTotalExpenses", 
    "Sum up the total amount from all expenses, grouped by month (YYYY-MM).",
    {
        expenses: ExpensesArraySchema,
    },
    async ({ expenses }) => {
        expenses = normalizeExpenses(expenses);
        const monthlyTotals: Record<string, number> = {};
        for (const exp of expenses) {
            // Use created_time if available, fallback to createdAt for backward compatibility
            const dateStr = exp.created_time || exp.createdAt;
            if (!dateStr) continue;
            // If dateStr is just YYYY-MM-DD, convert to YYYY-MM-DDT00:00:00Z for Date parsing
            let normalizedDateStr = dateStr;
            if (/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
                normalizedDateStr = dateStr + 'T00:00:00Z';
            }
            const date = new Date(normalizedDateStr);
            if (isNaN(date.getTime())) continue;
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            monthlyTotals[monthKey] = (monthlyTotals[monthKey] || 0) + (exp.amount || 0);
        }
        return { content: [{ type: "text", text: JSON.stringify(monthlyTotals) }] };
    },
);

// Version 1 tool
// server.tool(
//     "summarizeExpensesByCategory",
//     "Group and sum expenses by category (e.g., Travel, Food).",
//     {
//         expenses: ExpensesArraySchema,
//     },
//     async ({ expenses }) => {
//         expenses = normalizeExpenses(expenses);
//         const summary: Record<string, number> = {};
//         expenses.forEach(exp => {
//             const cat = exp.category || "Uncategorized";
//             summary[cat] = (summary[cat] || 0) + (exp.amount || 0);
//         });
//         return { content: [{ type: "text", text: JSON.stringify(summary) }] };
//     },
// );

// Version 2 tool
server.tool(
    "calculateExpenseGrowthMoM",
    "Calculate the month-over-month percentage growth in total expenses. Returns an object mapping YYYY-MM to growth percentage (first month is null).",
    {
        expenses: ExpensesArraySchema,
    },
    async ({ expenses }) => {
        expenses = normalizeExpenses(expenses);
        // Group by month
        const monthlyTotals: Record<string, number> = {};
        for (const exp of expenses) {
            // Use created_time if available, fallback to createdAt for backward compatibility
            const dateStr = exp.created_time || exp.createdAt;
            if (!dateStr) continue;
            // If dateStr is just YYYY-MM-DD, convert to YYYY-MM-DDT00:00:00Z for Date parsing
            let normalizedDateStr = dateStr;
            if (/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
                normalizedDateStr = dateStr + 'T00:00:00Z';
            }
            const date = new Date(normalizedDateStr);
            if (isNaN(date.getTime())) continue;
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            monthlyTotals[monthKey] = (monthlyTotals[monthKey] || 0) + (exp.amount || 0);
        }
        // Sort months chronologically
        const months = Object.keys(monthlyTotals).sort();
        const growth: Record<string, number|null> = {};
        let prevTotal: number|null = null;
        for (const month of months) {
            const total = monthlyTotals[month];
            if (prevTotal === null) {
                growth[month] = null;
            } else if (prevTotal === 0) {
                growth[month] = null;
            } else {
                growth[month] = ((total - prevTotal) / prevTotal) * 100;
            }
            prevTotal = total;
        }
        return { content: [{ type: "text", text: JSON.stringify(growth) }] };
    },
);

// Version 2 tool
server.tool(
    "taxImpactAnalysis",
    "Calculates the tax paid per transaction (amount_with_tax - amount) and the total tax paid.",
    {
        expenses: ExpensesArraySchema,
    },
    async ({ expenses }) => {
        expenses = normalizeExpenses(expenses);
        const taxPerTransaction: Record<string, number> = {};
        let totalTax = 0;
        for (const exp of expenses) {
            if (typeof exp.amount === "number" && typeof exp.amount_with_tax === "number") {
                const tax = exp.amount_with_tax - exp.amount;
                taxPerTransaction[exp.id] = tax;
                totalTax += tax;
            }
        }
        return { content: [{ type: "text", text: JSON.stringify({ taxPerTransaction, totalTax }) }] };
    },
);

// Version 2 tool
server.tool(
    "summarizeExpensesByAccountAndDescription",
    "Group and sum expenses by account_name and description (e.g., 'TDS 194C | Payment to vendor').",
    {
        expenses: ExpensesArraySchema,
    },
    async ({ expenses }) => {
        const summary: Record<string, number> = {};
        expenses.forEach(exp => {
            const account = exp.account_name || "Unknown Account";
            const desc = exp.description || "No Description";
            const key = `${account} | ${desc}`;
            summary[key] = (summary[key] || 0) + (exp.amount || 0);
        });
        return { content: [{ type: "text", text: JSON.stringify(summary) }] };
    },
);

// Version 2 tool
server.tool(
    "getTopVendorsByExpense",
    "Identify vendors with the highest total spending, using description and account_name as the vendor identifier. Returns a sorted array of vendors by total expense.",
    {
        expenses: ExpensesArraySchema,
        topN: z.number().optional().describe("Number of top vendors to return. If omitted, returns all sorted."),
    },
    async ({ expenses, topN }) => {
        expenses = normalizeExpenses(expenses);
        const vendorTotals: Record<string, number> = {};
        expenses.forEach(exp => {
            const account = exp.account_name || "Unknown Account";
            const desc = exp.description || "No Description";
            const key = `${account} | ${desc}`;
            vendorTotals[key] = (vendorTotals[key] || 0) + (exp.amount || 0);
        });
        // Convert to array and sort descending by amount
        const sortedVendors = Object.entries(vendorTotals)
            .map(([vendor, total]) => ({ vendor, total }))
            .sort((a, b) => b.total - a.total);
        const result = typeof topN === 'number' ? sortedVendors.slice(0, topN) : sortedVendors;
        return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
);

// --- Time Series Forecasting Tools ---
// Version 1 tool
server.tool(
    "forecastExpensesSimple",
    "Forecast future expenses based on historical expense data using exponential smoothing.",
    {
        historicalExpenses: z.array(z.object({
            date: z.string().describe("The date of the expense entry in ISO format (YYYY-MM-DD)"),
            amount: z.number().describe("The total expense amount for this date/period")
        })).describe("Array of historical expense data points with dates and amounts"),
        periodsToForecast: z.number().int().positive().default(1).describe("Number of future periods to forecast"),
        alpha: z.number().min(0).max(1).default(0.3).describe("Smoothing factor (0-1) for exponential smoothing. Higher values give more weight to recent observations.")
    },
    async ({ historicalExpenses, periodsToForecast, alpha }) => {
        try {
            // Extract and sort historical data by date
            const sortedData = [...historicalExpenses].sort((a, b) => 
                new Date(a.date).getTime() - new Date(b.date).getTime()
            );
            
            // Extract just the amounts for forecasting
            const historicalAmounts = sortedData.map(item => item.amount);
            
            // Perform forecasting
            const forecastResults = exponentialSmoothingForecast(
                historicalAmounts, 
                alpha, 
                periodsToForecast
            );
            
            // Format results
            const lastDate = new Date(sortedData[sortedData.length - 1].date);
            const forecast = forecastResults.map((amount, index) => {
                const forecastDate = new Date(lastDate);
                forecastDate.setMonth(forecastDate.getMonth() + index + 1);
                return {
                    date: forecastDate.toISOString().split('T')[0],
                    amount: Number(amount.toFixed(2))
                };
            });
            
            return { 
                content: [{ 
                    type: "text", 
                    text: JSON.stringify({
                        forecast,
                        method: "Exponential Smoothing",
                        alpha,
                        message: `Forecasted ${periodsToForecast} period(s) using exponential smoothing with alpha=${alpha}`
                    }, null, 2) 
                }]
            };
        } catch (error: unknown) {
            const message = error instanceof Error
                ? error.message
                : "An unknown error occurred during expense forecasting.";
            return {
                content: [{ type: "text", text: `Error: ${message}` }],
                isError: true,
            };
        }
    },
);

// Version 1 tool
server.tool(
    "forecastExpensesArima",
    "Forecast future expenses using ARIMA (AutoRegressive Integrated Moving Average) model.",
    {
        historicalExpenses: z.array(z.object({
            date: z.string().describe("The date of the expense entry in ISO format (YYYY-MM-DD)"),
            amount: z.number().describe("The total expense amount for this date/period")
        })).describe("Array of historical expense data points with dates and amounts"),
        periodsToForecast: z.number().int().positive().default(1).describe("Number of future periods to forecast"),
        p: z.number().int().nonnegative().default(1).describe("AR (Auto-Regressive) order parameter"),
        d: z.number().int().nonnegative().default(1).describe("I (Integrated/Differencing) order parameter"),
        q: z.number().int().nonnegative().default(1).describe("MA (Moving Average) order parameter")
    },
    async ({ historicalExpenses, periodsToForecast, p, d, q }) => {
        try {
            // Validate parameters
            if (p < 0 || d < 0 || q < 0) {
                throw new Error("ARIMA parameters p, d, and q must be non-negative integers");
            }
            
            // Extract and sort historical data by date
            const sortedData = [...historicalExpenses].sort((a, b) => 
                new Date(a.date).getTime() - new Date(b.date).getTime()
            );
            
            // Check if we have enough data points
            if (sortedData.length < p + d + q + 1) {
                throw new Error(
                    `Insufficient data for ARIMA(${p},${d},${q}). Need at least ${p + d + q + 1} observations, but got ${sortedData.length}.`
                );
            }
            
            // Extract just the amounts for forecasting
            const historicalAmounts = sortedData.map(item => item.amount);
            
            // Perform ARIMA forecasting
            const forecastResults = arimaForecast(
                historicalAmounts,
                p,
                d,
                q,
                periodsToForecast
            );
            
            // Format results
            const lastDate = new Date(sortedData[sortedData.length - 1].date);
            const forecast = forecastResults.map((amount, index) => {
                const forecastDate = new Date(lastDate);
                forecastDate.setMonth(forecastDate.getMonth() + index + 1);
                return {
                    date: forecastDate.toISOString().split('T')[0],
                    amount: Number(amount.toFixed(2))
                };
            });
            
            return { 
                content: [{ 
                    type: "text", 
                    text: JSON.stringify({
                        forecast,
                        method: "ARIMA",
                        parameters: { p, d, q },
                        message: `Forecasted ${periodsToForecast} period(s) using ARIMA(${p},${d},${q})`
                    }, null, 2) 
                }]
            };
        } catch (error: unknown) {
            const message = error instanceof Error
                ? error.message
                : "An unknown error occurred during ARIMA forecasting.";
            return {
                content: [{ type: "text", text: `Error: ${message}` }],
                isError: true,
            };
        }
    },
);

// --- Basic Arithmetic Tools ---
server.tool(
    "calculate_add",
    "Adds two numbers.",
    {
        a: z.number().describe("The first number to add."),
        b: z.number().describe("The second number to add."),
    },
    async ({ a, b }) => ({
        content: [{ type: "text", text: String(a + b) }],
    }),
);

// --- Percentage ---
server.tool(
    "calculate_percentage_of",
    "Calculates a specified percentage of a given number (e.g., 20% of 150).",
    {
        percentage: z
            .number()
            .describe("The percentage value (e.g., 20 for 20%)."),
        number: z
            .number()
            .describe("The number to calculate the percentage of."),
    },
    async ({ percentage, number }) => ({
        content: [{ type: "text", text: String((percentage / 100) * number) }],
    }),
);

// --- Main Execution Logic ---
async function main() {
    try {
        // Connect using stdio transport
        const transport = new StdioServerTransport();
        await server.connect(transport);
        console.error("Calculator MCP Server connected via stdio and ready.");
    } catch (error) {
        console.error("Failed to start or connect the server:", error);
        process.exit(1); // Exit with error code
    }
}

// Graceful shutdown handling
process.on("SIGINT", async () => {
    console.error("\nCaught interrupt signal (Ctrl+C). Shutting down...");
    await server.close(); // Close the server gracefully
    process.exit(0);
});

process.on("SIGTERM", async () => {
    console.error("Caught termination signal. Shutting down...");
    await server.close();
    process.exit(0);
});

main();