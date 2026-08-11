-- ==============================================
-- Olist E-Commerce Analytics
-- Star Schema - Dimension and Fact Tables
-- Author: Agu Charles Chibuike
-- ==============================================

-- Dimension: Customers
CREATE TABLE marts.dim_customers AS
SELECT DISTINCT
    customer_id,
    customer_unique_id,
    customer_city,
    customer_state,
    customer_zip_code_prefix
FROM raw.customers;

-- Dimension: Products (with English category names)
CREATE TABLE marts.dim_products AS
SELECT DISTINCT
    p.product_id,
    COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS product_category,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm
FROM raw.products p
LEFT JOIN raw.product_category_translation t
    ON p.product_category_name = t.product_category_name;

-- Dimension: Sellers
CREATE TABLE marts.dim_sellers AS
SELECT DISTINCT
    seller_id,
    seller_city,
    seller_state,
    seller_zip_code_prefix
FROM raw.sellers;

-- Dimension: Brazilian States (abbreviation to full name)
CREATE TABLE marts.dim_states AS
SELECT abbreviation, full_name FROM (VALUES
    ('AC', 'Acre'), ('AL', 'Alagoas'), ('AM', 'Amazonas'),
    ('AP', 'Amapá'), ('BA', 'Bahia'), ('CE', 'Ceará'),
    ('DF', 'Distrito Federal'), ('ES', 'Espírito Santo'),
    ('GO', 'Goiás'), ('MA', 'Maranhão'), ('MG', 'Minas Gerais'),
    ('MS', 'Mato Grosso do Sul'), ('MT', 'Mato Grosso'),
    ('PA', 'Pará'), ('PB', 'Paraíba'), ('PE', 'Pernambuco'),
    ('PI', 'Piauí'), ('PR', 'Paraná'), ('RJ', 'Rio de Janeiro'),
    ('RN', 'Rio Grande do Norte'), ('RO', 'Rondônia'),
    ('RR', 'Roraima'), ('RS', 'Rio Grande do Sul'),
    ('SC', 'Santa Catarina'), ('SE', 'Sergipe'),
    ('SP', 'São Paulo'), ('TO', 'Tocantins')
) AS t(abbreviation, full_name);

-- Fact Table: Orders
CREATE TABLE marts.fact_orders AS
SELECT
    o.order_id,
    o.customer_id,
    oi.seller_id,
    oi.product_id,
    o.order_status,
    o.order_purchase_timestamp::timestamp AS order_date,
    o.order_delivered_customer_date::timestamp AS delivered_date,
    o.order_estimated_delivery_date::timestamp AS estimated_delivery_date,
    oi.price,
    oi.freight_value,
    oi.price + oi.freight_value AS total_value
FROM raw.orders o
LEFT JOIN raw.order_items oi ON o.order_id = oi.order_id;