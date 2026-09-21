# Analyzing Customer Behavior for E-commerce Insights

National Service assessment submission for **Npontu Technologies** — Intelligent Systems Services Engineer role.

**Author:** Samuel Benson — BSc. Information Systems and Technology, University of Mines and Technology (UMaT), Tarkwa

## Overview

This project analyzes a synthetic e-commerce customer activity dataset to extract insights that
help a business improve sales, enhance customer engagement, and personalize the shopping
experience. It covers the full brief: exploratory data analysis, data cleaning, big-data-scale
processing with Apache Spark, feature engineering, a cross-validated predictive
product-recommendation model, and business recommendations.

## Repository structure

```
├── notebook/
│   └── Npontu_Ecommerce_Analysis.ipynb   # Full analysis, executed end-to-end (code + outputs)
├── report/
│   ├── Npontu_Ecommerce_Analysis_Report.docx   # Written report (Word)
│   └── Npontu_Ecommerce_Analysis_Report.pdf    # Written report (PDF)
├── data_generation/
│   └── generate_data.py                   # Standalone script that generates the synthetic dataset
└── README.md
```

## Methodology summary

1. **Synthetic data generation** — customers, products, and browsing/cart/purchase interaction
   events, with realistic data-quality issues (missing values, duplicates, invalid entries)
   deliberately injected.
2. **EDA & cleaning** — issues identified and resolved with documented, case-specific decisions.
3. **Big data tool: Apache Spark (PySpark)** — used to aggregate the interaction-level data
   (revenue and purchase volume by category and region) at scale; chosen over Kafka
   (streaming, not needed for this batch dataset) and Elasticsearch (search/indexing, not the
   core task here).
4. **Feature engineering** — behavioral (browsing counts per category), time-based (recency,
   tenure), and demographic features built per customer.
5. **Predictive modeling** — a Random Forest classifier per product category, predicting purchase
   probability from browsing behavior and demographics; validated with 5-fold stratified
   cross-validation (ROC-AUC 0.63–0.98 across categories). Predictions are ranked per customer to
   generate top-N product recommendations.
6. **Insights & business recommendations** — derived from the Spark aggregations and model
   results; documented in the report.

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn faker pyspark jupyter
jupyter notebook notebook/Npontu_Ecommerce_Analysis.ipynb
```

The notebook is self-contained — it generates its own synthetic dataset on run, so no external
data file is required. Requires Java (for PySpark) to be installed locally.

## Deliverables

- [Executed Jupyter notebook](notebook/Npontu_Ecommerce_Analysis.ipynb) — full code, comments, and outputs
- [Written report](report/Npontu_Ecommerce_Analysis_Report.pdf) — methodology, results, and business recommendations
