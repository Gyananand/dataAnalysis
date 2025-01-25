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
