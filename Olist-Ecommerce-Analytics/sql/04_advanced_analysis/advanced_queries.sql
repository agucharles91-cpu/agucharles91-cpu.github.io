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


-- 11. ROW_NUMBER - UNIQUE CUSTOMER RANKING
-- Assigns a unique rank to each customer by revenue
-- Unlike RANK(), ROW_NUMBER() never produces ties
WITH customer_totals AS (
    SELECT
        c.customer_unique_id,
        ROUND(SUM(f.total_value)::numeric, 2) AS total_revenue
    FROM marts.fact_orders f
    JOIN marts.dim_customers c ON f.customer_id = c.customer_id
    WHERE f.order_status = 'delivered'
    GROUP BY 1
)
SELECT
    ROW_NUMBER() OVER (ORDER BY total_revenue DESC) AS row_num,
    customer_unique_id,
    total_revenue
FROM customer_totals;


-- 12. DENSE_RANK - PRODUCT CATEGORY REVENUE RANKING
-- DENSE_RANK() doesn't skip numbers after ties
-- e.g. if two categories tie for 1st, next is 2nd not 3rd
WITH category_revenue AS (
    SELECT
        p.product_category,
        ROUND(SUM(f.total_value)::numeric, 2) AS revenue
    FROM marts.fact_orders f
    JOIN marts.dim_products p ON f.product_id = p.product_id
    WHERE f.order_status = 'delivered'
    GROUP BY 1
)
SELECT
    DENSE_RANK() OVER (ORDER BY revenue DESC) AS dense_rank,
    product_category,
    revenue
FROM category_revenue;


-- 13. LEAD() - NEXT MONTH REVENUE COMPARISON
-- LEAD() looks forward to the next row
-- Here it compares each month's revenue to the next month
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
    LEAD(revenue) OVER (ORDER BY month) AS next_month_revenue,
    ROUND(100.0 * (LEAD(revenue) OVER (ORDER BY month) - revenue)
          / NULLIF(revenue, 0), 2) AS growth_pct
FROM monthly
ORDER BY month;


-- 14. RETENTION ANALYSIS
-- Identifies customers who made more than one purchase
-- Calculates repeat purchase rate across the entire dataset
WITH customer_orders AS (
    SELECT
        c.customer_unique_id,
        COUNT(DISTINCT f.order_id) AS total_orders,
        MIN(f.order_date) AS first_order,
        MAX(f.order_date) AS last_order
    FROM marts.fact_orders f
    JOIN marts.dim_customers c ON f.customer_id = c.customer_id
    WHERE f.order_status = 'delivered'
    GROUP BY 1
)
SELECT
    SUM(CASE WHEN total_orders > 1 THEN 1 ELSE 0 END) AS repeat_customers,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN total_orders > 1 THEN 1 ELSE 0 END) 
          / COUNT(*), 2) AS repeat_rate_pct,
    ROUND(AVG(total_orders), 2) AS avg_orders_per_customer
FROM customer_orders;


-- 15. CONDITIONAL AGGREGATION - ORDER STATUS BY CATEGORY
-- Pivots order status counts into columns using CASE
-- Shows delivered, cancelled and other counts per category
SELECT
    p.product_category,
    COUNT(DISTINCT f.order_id) AS total_orders,
    SUM(CASE WHEN f.order_status = 'delivered' THEN 1 ELSE 0 END) AS delivered,
    SUM(CASE WHEN f.order_status = 'canceled' THEN 1 ELSE 0 END) AS cancelled,
    SUM(CASE WHEN f.order_status NOT IN ('delivered','canceled') THEN 1 ELSE 0 END) AS other,
    ROUND(100.0 * SUM(CASE WHEN f.order_status = 'canceled' THEN 1 ELSE 0 END)
          / COUNT(DISTINCT f.order_id), 2) AS cancel_rate_pct
FROM marts.fact_orders f
JOIN marts.dim_products p ON f.product_id = p.product_id
GROUP BY 1
ORDER BY total_orders DESC
LIMIT 15;