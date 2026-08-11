-- ==============================================
-- Olist E-Commerce Analytics
-- Advanced SQL Analysis
-- Author: Agu Charles Chibuike
-- Features: CTEs, Window Functions, Cohort 
-- Analysis, Rankings, Rolling Metrics
-- ==============================================


-- 1. MONTHLY REVENUE TREND
-- Simple aggregation showing business growth over time
SELECT
    DATE_TRUNC('month', order_date) AS month,
    ROUND(SUM(total_value)::numeric, 2) AS revenue
FROM marts.fact_orders
WHERE order_status = 'delivered'
AND order_date < '2018-09-01'
GROUP BY 1
ORDER BY 1;


-- 2. TOP 10 PRODUCT CATEGORIES BY REVENUE
-- Joins fact table to dimension for readable category names
SELECT
    p.product_category,
    ROUND(SUM(f.total_value)::numeric, 2) AS revenue,
    COUNT(DISTINCT f.order_id) AS total_orders
FROM marts.fact_orders f
JOIN marts.dim_products p ON f.product_id = p.product_id
WHERE f.order_status = 'delivered'
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10;


-- 3. DELIVERY PERFORMANCE BY STATE
-- Calculates average delivery days and on-time rate per state
SELECT
    ds.full_name AS state,
    COUNT(DISTINCT f.order_id) AS total_orders,
    ROUND(AVG(EXTRACT(EPOCH FROM (f.delivered_date - f.order_date))/86400)::numeric, 1) AS avg_delivery_days,
    ROUND(100.0 * SUM(CASE WHEN f.delivered_date <= f.estimated_delivery_date THEN 1 ELSE 0 END) 
          / COUNT(*), 2) AS on_time_pct
FROM marts.fact_orders f
JOIN marts.dim_customers c ON f.customer_id = c.customer_id
JOIN marts.dim_states ds ON c.customer_state = ds.abbreviation
WHERE f.order_status = 'delivered'
AND f.delivered_date IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC;


-- 4. ROLLING 3-MONTH REVENUE (WINDOW FUNCTION)
-- Shows smoothed revenue trend using rolling average
-- Removes month-to-month noise to reveal true growth trend
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        ROUND(SUM(total_value)::numeric, 2) AS revenue
    FROM marts.fact_orders
    WHERE order_status = 'delivered'
    AND order_date < '2018-09-01'
    GROUP BY 1
)
SELECT
    month,
    revenue,
    ROUND(AVG(revenue) OVER (
        ORDER BY month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )::numeric, 2) AS rolling_3month_avg
FROM monthly_revenue
ORDER BY month;


-- 5. REVENUE RANKING BY STATE (WINDOW FUNCTION)
-- Ranks states by revenue using RANK() window function
WITH state_revenue AS (
    SELECT
        ds.full_name AS state,
        ROUND(SUM(f.total_value)::numeric, 2) AS revenue,
        COUNT(DISTINCT f.order_id) AS total_orders
    FROM marts.fact_orders f
    JOIN marts.dim_customers c ON f.customer_id = c.customer_id
    JOIN marts.dim_states ds ON c.customer_state = ds.abbreviation
    WHERE f.order_status = 'delivered'
    GROUP BY 1
)
SELECT
    state,
    revenue,
    total_orders,
    RANK() OVER (ORDER BY revenue DESC) AS revenue_rank
FROM state_revenue;


-- 6. CUSTOMER SEGMENTATION BY REVENUE DECILE (WINDOW FUNCTION)
-- Splits customers into 10 equal groups by spending
-- Top decile = highest spenders
WITH customer_totals AS (
    SELECT
        c.customer_unique_id,
        ROUND(SUM(f.total_value)::numeric, 2) AS total_revenue,
        COUNT(DISTINCT f.order_id) AS total_orders
    FROM marts.fact_orders f
    JOIN marts.dim_customers c ON f.customer_id = c.customer_id
    WHERE f.order_status = 'delivered'
    GROUP BY 1
)
SELECT
    customer_unique_id,
    total_revenue,
    total_orders,
    NTILE(10) OVER (ORDER BY total_revenue DESC) AS revenue_decile
FROM customer_totals
ORDER BY total_revenue DESC;


-- 7. PARETO ANALYSIS - CUMULATIVE REVENUE CONTRIBUTION
-- Shows what % of customers drive what % of revenue
-- Confirms the 80/20 Pareto principle
WITH customer_totals AS (
    SELECT
        c.customer_unique_id,
        ROUND(SUM(f.total_value)::numeric, 2) AS total_revenue
    FROM marts.fact_orders f
    JOIN marts.dim_customers c ON f.customer_id = c.customer_id
    WHERE f.order_status = 'delivered'
    GROUP BY 1
),
ranked AS (
    SELECT
        customer_unique_id,
        total_revenue,
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC) AS running_total,
        SUM(total_revenue) OVER () AS grand_total
    FROM customer_totals
)
SELECT
    customer_unique_id,
    total_revenue,
    ROUND(100.0 * running_total / grand_total, 2) AS cumulative_pct
FROM ranked
ORDER BY total_revenue DESC;


-- 8. COHORT ANALYSIS - FIRST PURCHASE MONTH
-- Groups customers by the month they first bought
-- Shows how many new customers were acquired each month
WITH first_purchase AS (
    SELECT
        c.customer_unique_id,
        MIN(DATE_TRUNC('month', f.order_date)) AS cohort_month
    FROM marts.fact_orders f
    JOIN marts.dim_customers c ON f.customer_id = c.customer_id
    WHERE f.order_status = 'delivered'
    GROUP BY 1
)
SELECT
    cohort_month,
    COUNT(DISTINCT customer_unique_id) AS new_customers
FROM first_purchase
GROUP BY 1
ORDER BY 1;


-- 9. SELLER PERFORMANCE RANKING
-- Ranks sellers by revenue with order count and avg order value
WITH seller_stats AS (
    SELECT
        f.seller_id,
        s.seller_state,
        ROUND(SUM(f.total_value)::numeric, 2) AS total_revenue,
        COUNT(DISTINCT f.order_id) AS total_orders,
        ROUND(AVG(f.total_value)::numeric, 2) AS avg_order_value
    FROM marts.fact_orders f
    JOIN marts.dim_sellers s ON f.seller_id = s.seller_id
    WHERE f.order_status = 'delivered'
    GROUP BY 1, 2
)
SELECT
    seller_id,
    seller_state,
    total_revenue,
    total_orders,
    avg_order_value,
    RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank
FROM seller_stats
ORDER BY revenue_rank
LIMIT 20;


-- 10. MONTH OVER MONTH REVENUE GROWTH (WINDOW FUNCTION)
-- Calculates % change in revenue from previous month
-- Identifies acceleration and deceleration in growth
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        ROUND(SUM(total_value)::numeric, 2) AS revenue
    FROM marts.fact_orders
    WHERE order_status = 'delivered'
    AND order_date < '2018-09-01'
    GROUP BY 1
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month_revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) 
          / NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 2) AS mom_growth_pct
FROM monthly
ORDER BY month;