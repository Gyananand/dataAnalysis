# Data Cleaning: Layoffs Dataset

This project demonstrates the process of cleaning a dataset containing information about layoffs. The cleaning process involves removing duplicates, standardizing data, handling null or blank values, and modifying the dataset structure.

---

## Table of Contents
1. Overview
2. Prerequisites
3. Steps Involved
   - Creating a Staging Table
   - Removing Duplicates
   - Standardizing Data
   - Handling Null or Blank Values
4. Final Output
5. SQL Safe Updates Mode
6. Contributors

---

## Overview

The purpose of this project is to clean the **Layoffs Dataset** to ensure its quality and usability for analysis. The dataset contains columns like `company`, `location`, `industry`, `total_laid_off`, `date`, and more. The data cleaning process is performed in **MySQL** using structured queries.

---

## Prerequisites

- **MySQL Database** with the `layoffs` table imported.
- SQL-safe updates disabled (can be re-enabled after the process).
- Basic understanding of SQL queries.

---

## Steps Involved

### 1. Creating a Staging Table
A staging table `layoffs_staging` is created as a copy of the raw `layoffs` table to ensure the raw data remains unaltered.

```sql
CREATE TABLE layoffs_staging LIKE layoffs;

INSERT INTO layoffs_staging
SELECT * FROM layoffs;
```

# Data Aggregation and Analysis: Layoffs Dataset

This document showcases how various SQL queries are used to analyze and summarize the **Layoffs Dataset**. The focus is on understanding trends, patterns, and key metrics related to layoffs based on industries, companies, countries, and time.

---

## Table of Contents

1. Overview  
2. Queries and Outputs  
   - General Table Overview  
   - Maximum Values  
   - Top Companies by Layoffs  
   - Industry-Wise Layoffs  
   - Country-Wise Layoffs  
   - Yearly Trends  
   - Stage-Wise Layoffs  
   - Monthly Trends and Rolling Totals  
   - Company-Year Aggregation and Rankings  
3. Insights and Interpretations  
4. Contributors  

---

## Overview

The **Layoffs Dataset** contains various columns, such as:
- `company`: Name of the company.
- `industry`: Industry type.
- `country`: Country of the layoffs.
- `total_laid_off`: Number of employees laid off.
- `percentage_laid_off`: Percentage of layoffs.
- `date`: Date of layoffs.
- `stage`: Funding or business stage of the company.

This analysis focuses on summarizing the dataset to uncover actionable insights.

---

## Queries and Outputs

### 1. General Table Overview
Display all rows and columns to check the dataset structure:
```sql
SELECT *
FROM layoffs_staging2;

