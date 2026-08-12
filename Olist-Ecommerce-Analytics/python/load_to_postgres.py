import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL

# Credentials are read from environment variables
# Before running, set these in your terminal:
# Windows: set POSTGRES_PASSWORD=yourpassword
# Windows: set OLIST_DATA_DIR=C:\path\to\your\olist\data

engine = create_engine(URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost",
    port=5432,
    database="olist_ecommerce"
))

data_dir = os.getenv("OLIST_DATA_DIR")
if not data_dir:
    raise ValueError("Please set the OLIST_DATA_DIR environment variable.")

# Create schemas if they don't exist
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS staging"))
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS marts"))
    conn.commit()
    print("✅ Schemas ready")

csv_table_map = {
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_customers_dataset.csv": "customers",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_geolocation_dataset.csv": "geolocation",
    "product_category_name_translation.csv": "product_category_translation"
}

for csv_file, table_name in csv_table_map.items():
    path = os.path.join(data_dir, csv_file)
    print(f"Loading {csv_file}...")
    df = pd.read_csv(path)
    df.to_sql(table_name, engine, schema="raw", if_exists="replace", index=False)
    print(f"  ✅ {len(df):,} rows loaded")

print("\nAll done!")