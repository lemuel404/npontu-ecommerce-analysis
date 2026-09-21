"""
Generates a synthetic e-commerce dataset:
- customers.csv        : demographics
- products.csv         : product catalog
- interactions.csv     : browsing/cart/purchase events (the main "big" table)

Deliberate messiness is injected (missing values, duplicates, bad entries)
so the cleaning step in the notebook has real work to do.
"""
import numpy as np
import pandas as pd
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 2000
N_PRODUCTS = 300
N_INTERACTIONS = 60000

CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Beauty",
              "Sports", "Books", "Toys", "Groceries"]
DEVICES = ["mobile", "desktop", "tablet"]
CHANNELS = ["organic_search", "paid_ad", "email", "social_media", "direct"]
EVENT_TYPES = ["view", "add_to_cart", "purchase"]
MEMBERSHIP = ["basic", "silver", "gold", "platinum"]
REGIONS_GH = ["Greater Accra", "Ashanti", "Western", "Central",
              "Eastern", "Northern", "Volta", "Bono"]

# ---------------------------------------------------------------
# 1. Customers
# ---------------------------------------------------------------
customers = []
for i in range(1, N_CUSTOMERS + 1):
    age = int(np.random.normal(32, 10))
    age = max(16, min(75, age))
    row = {
        "customer_id": f"C{i:05d}",
        "age": age,
        "gender": random.choice(["Male", "Female", "Other"]),
        "region": random.choice(REGIONS_GH),
        "signup_date": fake.date_between(start_date="-3y", end_date="-30d"),
        "membership_tier": random.choices(MEMBERSHIP, weights=[50, 25, 15, 10])[0],
    }
    customers.append(row)

customers_df = pd.DataFrame(customers)

# inject missing values
for col in ["age", "gender", "region"]:
    idx = customers_df.sample(frac=0.03, random_state=1).index
    customers_df.loc[idx, col] = np.nan

# inject a few erroneous ages
bad_idx = customers_df.sample(frac=0.01, random_state=2).index
customers_df.loc[bad_idx, "age"] = -5

# inject duplicate customer rows
dupes = customers_df.sample(frac=0.02, random_state=3)
customers_df = pd.concat([customers_df, dupes], ignore_index=True)

# ---------------------------------------------------------------
# 2. Products
# ---------------------------------------------------------------
products = []
for i in range(1, N_PRODUCTS + 1):
    category = random.choice(CATEGORIES)
    base_price = {
        "Electronics": (50, 2000), "Fashion": (10, 300),
        "Home & Kitchen": (15, 500), "Beauty": (5, 150),
        "Sports": (10, 400), "Books": (5, 80),
        "Toys": (5, 200), "Groceries": (2, 60),
    }[category]
    products.append({
        "product_id": f"P{i:04d}",
        "category": category,
        "price": round(np.random.uniform(*base_price), 2),
        "brand": fake.company(),
    })
products_df = pd.DataFrame(products)

# inject a few erroneous negative prices
bad_idx = products_df.sample(frac=0.02, random_state=4).index
products_df.loc[bad_idx, "price"] = -products_df.loc[bad_idx, "price"]

# ---------------------------------------------------------------
# 3. Interactions (view / add_to_cart / purchase)
# ---------------------------------------------------------------
customer_ids = customers_df["customer_id"].unique().tolist()
product_rows = products_df.to_dict("records")

# give products popularity weights so behavior isn't uniformly random
popularity = np.random.zipf(1.5, size=len(product_rows))
popularity = popularity / popularity.sum()

# give customers an affinity for 1-2 categories (drives believable patterns)
customer_affinity = {cid: random.sample(CATEGORIES, k=random.choice([1, 2]))
                      for cid in customer_ids}

interactions = []
start = pd.Timestamp("2025-09-01")
end = pd.Timestamp("2026-09-01")

for _ in range(N_INTERACTIONS):
    cid = random.choice(customer_ids)
    affinity = customer_affinity[cid]
    # 70% chance the interaction is within the customer's affinity categories
    if random.random() < 0.7:
        pool = [p for p in product_rows if p["category"] in affinity]
        prod = random.choice(pool) if pool else random.choices(product_rows, weights=popularity)[0]
    else:
        prod = random.choices(product_rows, weights=popularity)[0]

    ts = fake.date_time_between(start_date=start, end_date=end)
    # purchase is rarer than view/cart
    event_type = random.choices(EVENT_TYPES, weights=[70, 20, 10])[0]

    interactions.append({
        "event_id": fake.uuid4(),
        "customer_id": cid,
        "product_id": prod["product_id"],
        "category": prod["category"],
        "event_type": event_type,
        "timestamp": ts,
        "device": random.choice(DEVICES),
        "channel": random.choice(CHANNELS),
        "session_id": fake.uuid4()[:8],
        "purchase_amount": round(prod["price"] * random.choice([1, 1, 1, 2, 3]), 2)
                            if event_type == "purchase" else np.nan,
    })

interactions_df = pd.DataFrame(interactions)

# inject missing values in device/channel
for col in ["device", "channel"]:
    idx = interactions_df.sample(frac=0.02, random_state=5).index
    interactions_df.loc[idx, col] = np.nan

# inject an invalid event_type value
bad_idx = interactions_df.sample(frac=0.005, random_state=6).index
interactions_df.loc[bad_idx, "event_type"] = "unknown_event"

# inject exact duplicate interaction rows
dupes = interactions_df.sample(frac=0.015, random_state=7)
interactions_df = pd.concat([interactions_df, dupes], ignore_index=True)

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
customers_df.to_csv("customers.csv", index=False)
products_df.to_csv("products.csv", index=False)
interactions_df.to_csv("interactions.csv", index=False)

print("customers:", customers_df.shape)
print("products:", products_df.shape)
print("interactions:", interactions_df.shape)
