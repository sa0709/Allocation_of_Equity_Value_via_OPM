## 🧮 Equity Allocation Model for Multi-Class Share Structures

This repository contains a **Python-based valuation engine** designed to allocate a company’s equity value across multiple share classes using a structured, option-pricing–based waterfall model.
It automates the computation of liquidation preference tranches, conversion breakpoints, and per-share values using **Black–Scholes option pricing**, cap table inputs, and conversion mechanics.

---

### 🔍 **Key Features**

* **Automated Cap Table Processing**
  Reads share class details, conversion prices, liquidation preferences, and share counts directly from an Excel input file.

* **Breakpoints & Tranche Construction**
  Computes liquidation preference breakpoints, conversion price tranches, and cumulative allocation layers.

* **Option-Based Waterfall Valuation**
  Uses the **Black–Scholes model** to value each tranche by treating breakpoints as strike prices.

* **Pro-Rata Allocation Matrix**
  Dynamically constructs a distribution matrix to allocate value across investor classes based on liquidation rights and conversion priorities.

* **Per-Share Value Output**
  Calculates total class value and allocates final **fair value per share** for each security class.

---

### 📁 **Workflow Summary**

1. **Import Input Data**
   Loads cap table and valuation assumptions from `Input.xlsx`.

2. **Compute Exit Amounts**
   Converts each class’s conversion price × share count into value (in millions).

3. **Determine Breakpoints**

   * Liquidation preference layers
   * Conversion price tranche boundaries
   * Cumulative breakpoint sums

4. **Apply Black–Scholes Option Pricing**
   Treats each breakpoint as a “strike price” to compute option value decay across tranches.

5. **Allocate Tranche Values**
   Based on share class rights and pro-rata mechanics.

6. **Output Final Values**
   Provides total class value and **value per share** for every share class.

---

### 🛠️ **Technologies Used**

* **Python**, **NumPy**, **Pandas**
* **SciPy** (for cumulative normal distribution)
* Black–Scholes mathematical model
* Excel-based input/output workflow

---

### 📈 **Use Case**

Helpful for:

* Cap table modeling
* 409A valuations
* Investor waterfall analysis
* Fund distribution waterfalls
* Startup and VC valuation scenarios involving **complex share structures**
