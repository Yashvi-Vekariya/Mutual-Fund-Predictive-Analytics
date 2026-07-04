# 📈 Mutual Fund Predictive Analytics

## Overview and Motivation

### Overview
This project focuses on predicting the best mutual funds to invest in based on short- and long-term financial goals, using historical returns data. I selected the top 10 fund families based on the largest Assets Under Management (AUM), resulting in a dataset of approximately 1,277 funds across various Morningstar categories, including Large, MidCap, Small, Growth, Blend, Value, and Index funds. Data sources included Morningstar.com and Yahoo Finance. I used web scraping techniques with Beautiful Soup and `pandas.read_html()` to collect fund-level data such as Alpha, Beta, Sharpe Ratio, Sortino Ratio, Standard Deviation, Returns, Management information, Holdings, and more—most of which were annualized. NAV data from Yahoo Finance (available daily) was used for fund return comparisons, and visualizations were plotted using `matplotlib`.

### Motivation
Having worked in the finance domain, I've gained strong insights into the mutual fund sector by working directly with real-world production data. After completing a Data Science course, I was motivated to apply the machine learning techniques I learned to analyze mutual fund performance. Mutual funds have a wide range of quantitative indicators, making them ideal candidates for building predictive models. My objective was to identify the best-performing funds and evaluate them based on different investment horizons. I believe this analysis has practical value and would like to present it to my managers as a data-driven tool to support investment decisions.

---

## 🧪 MLflow Experiment Tracking

This project includes **production-style MLflow experiment tracking** — every model training run is fully logged with parameters, metrics, artifacts, and the trained model itself.

### What is tracked

| Category | Details |
|---|---|
| **Parameters** | `n_estimators`, `max_depth`, `min_samples_split`, `horizon_years`, features used, train/test split |
| **Metrics** | Accuracy, Precision, Recall, F1 Score, MSE, Cross-validation mean ± std |
| **Feature importances** | Individual importance score per feature logged as metric |
| **Artifacts** | Feature importance chart, confusion matrix, return distribution histogram, predictions CSV |
| **Model** | Trained `RandomForestClassifier` registered in MLflow Model Registry |

### Experiments run

Three experiments were run across different investment horizons and hyperparameter settings:

| Run name | Horizon | n_estimators | max_depth | Accuracy | CV Accuracy |
|---|---|---|---|---|---|
| `3yr_baseline_n100` | 3 years | 100 | None | **97.3%** | 98.0% |
| `5yr_deep_n200` | 5 years | 200 | 10 | **98.0%** | 97.0% |
| `10yr_shallow_n150` | 10 years | 150 | 5 | **96.7%** | 96.0% |

<img width="1826" height="872" alt="Screenshot 2026-07-04 105125" src="https://github.com/user-attachments/assets/f664b8d5-f13a-4566-9a37-78df79a7b0d5" />
<img width="1912" height="362" alt="Screenshot 2026-07-04 105053" src="https://github.com/user-attachments/assets/60f8ac9d-c823-44c4-8d01-e28b35118972" />
<img width="1826" height="876" alt="Screenshot 2026-07-04 104909" src="https://github.com/user-attachments/assets/482c20cb-675e-4537-838b-08919be01d83" />
<img width="1860" height="870" alt="Screenshot 2026-07-04 104942" src="https://github.com/user-attachments/assets/02c977ad-c28e-4b9e-b120-29cb9eb0b86c" />


### How to reproduce

```bash
# Install dependencies
pip install mlflow scikit-learn matplotlib pandas numpy

# Run all 3 experiments
python mlflow_tracking.py

# Launch the tracking dashboard
mlflow ui
# Open: http://127.0.0.1:5000
```

---

## Data Collection and Preparation

I collected data using web scraping methods on Morningstar.com. Since the data was not formatted cleanly, I used various Python techniques to transform and align it into a structured DataFrame. Some funds had missing values due to mergers or gaps in reporting over certain years. I addressed these by using forward-fill (`ffill()`) where applicable, or by excluding those funds when data was insufficient. I also used Yahoo Finance to gather daily NAV data for return-based visualizations. The scraping was performed using BeautifulSoup for navigating through elements like links and table rows, and `pandas.read_html()` to directly extract tabular data from the webpages.

---

## Exploratory Data Analysis

To better understand the data and its relationships, I used several visualization techniques and statistical tools:

- **Box Plot** – To evaluate the accuracy of various estimators used in Random Forest Classifier through cross-validation.
- **Scatter Plot** – To explore the relationship between Expense Ratio vs Returns, and Risk vs Returns.
- **Heatmap** – To display correlation across top-performing funds.
- **Line Chart** – To visualize the growth of fund returns over time.
- **Histogram** – To understand the distribution of fund returns.

### Statistical Methods and Justification

I considered multiple visualization approaches to thoroughly analyze the dataset:

1. **Box Plots**: Used to determine which estimators provide better accuracy for Random Forest classifier using cross-validation.
2. **Scatter Plots**: Applied to examine relationships between Expense Ratio and Returns, as well as Risk and Returns.
3. **Heat Maps**: Implemented to show correlation patterns among multiple top-performing funds.
4. **Line Charts**: Created to plot fund returns for multiple top-performing funds over time.
5. **Histograms**: Generated to visualize the distribution of fund returns.

---

## Final Analysis

### Key Findings

1. **Performance Analysis**: I identified the top-performing funds for 3-year, 5-year, and 10-year investment periods.

2. **Expense Ratio Impact**: In the latest 3-year data, mutual funds with lower expense ratios generally yielded higher returns. However, for 10-year investments, funds with higher expense ratios demonstrated better performance.

3. **Model Performance**:
   - I implemented both Linear Regression and Random Forest Classification approaches
   - Random Forest Classifier achieved **92% accuracy** in predicting top-performing funds
   - Linear Regression effectively identified poor-performing funds with a Mean Squared Error approximately **0**
   - With MLflow experiment tracking across 3 investment horizons, cross-validated accuracy reached up to **98%**

4. **Investment Recommendations**: I identified **155 consistently underperforming funds** from the pool of 1,277 funds, marking them as unsuitable for investment due to poor returns.

5. **Portfolio Visualization**: I calculated the cumulative portfolio returns based on selected top-performing funds to help visualize potential investor gains over time.

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | Random Forest Classifier, Linear Regression |
| MLflow | Experiment tracking, model registry |
| pandas | Data wrangling |
| matplotlib | Visualizations |
| BeautifulSoup | Web scraping (Morningstar) |
| Yahoo Finance API | NAV data collection |

---

Future Scope :- https://predictive-fund-suite.lovable.app/

## 👤 Author

**Yashvi Vekariya**
🌐 [LinkedIn](https://www.linkedin.com/in/yashvi-vekariya)
💻 [GitHub](https://github.com/Yashvi-Vekariya)
📧 [yashviivekariya@gmail.com](mailto:yashviivekariya@gmail.com)
