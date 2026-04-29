# Bike Sharing Data Analysis Project

**Name:** Clarisya Adeline  
**Email:** cdcc011d6x1678@student.devacademy.id  
**Dicoding ID:** cdcc011d6x1678

---

## Project Description

This project analyzes the **Bike Sharing Dataset** from the Capital Bikeshare system, Washington D.C. (2011–2012). The analysis covers the influence of weather on daily bike rentals and peak usage patterns segmented by day type.

### Business Questions
1. How do weather conditions and temperature affect the average number of daily bike rentals at Capital Bikeshare Washington D.C. during 2011–2012, and which weather category causes the most significant drop in rentals?
2. At what hours do peak rentals occur and how does this pattern differ between working days and holidays during 2011–2012, to determine an optimal bike redistribution strategy?

---

## Directory Structure

```
submission/
├── dashboard/
│   ├── main_data.csv       <- Main dataset for dashboard (cleaned & enriched hour.csv)
│   └── dashboard.py        <- Streamlit interactive dashboard (4 tabs)
├── data/
│   ├── day.csv             <- Daily dataset (original)
│   └── hour.csv            <- Hourly dataset (original)
├── notebook.ipynb          <- Complete analysis notebook (executed)
├── README.md               <- This guide
├── requirements.txt        <- Python library dependencies
└── url.txt                 <- Streamlit Cloud deployment URL
```

---

## How to Run the Dashboard

### Prerequisites
- Python 3.8 or newer
- pip

### Step 1: Extract the Project
```bash
cd submission
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Dashboard
```bash
streamlit run dashboard/dashboard.py
```

The dashboard will open automatically in your browser at: `http://localhost:8501`

---

## Dashboard Features

- **Overview** — Daily rental trend with 30-day moving average, season share pie chart, and year-over-year KPIs.
- **Question 1: Weather & Temperature** — Average rentals by weather condition, temperature scatter plot with correlation trend, season x weather heatmap, and boxplot distribution.
- **Question 2: Peak Hours** — Working day vs holiday hourly patterns, casual vs registered user comparison, heatmap by hour and weekday, and average rentals by day of week.
- **Advanced Analysis** — Manual clustering (binning) of days by usage category with detailed statistical profile and season composition chart.

### Interactive Sidebar Filters
- Year (2011 / 2012)
- Season
- Weather condition

---

## Libraries Used

| Library    | Version | Purpose               |
|------------|---------|-----------------------|
| pandas     | 2.1.4   | Data manipulation     |
| numpy      | 1.26.4  | Numerical computation |
| matplotlib | 3.8.2   | Data visualization    |
| seaborn    | 0.13.2  | Statistical charts    |
| streamlit  | 1.32.2  | Interactive dashboard |

---

## Key Findings

1. **Weather is the dominant factor**: Rentals drop by **63%** in rainy conditions compared to clear weather.
2. **Temperature has strong positive correlation** (r = 0.63) with daily rental volume.
3. **Working day bimodal pattern**: Peak at 08:00 (morning commute) and 17:00 (evening commute).
4. **Holiday unimodal pattern**: Rentals are distributed around 10:00–15:00 (recreational use).
5. **Very High cluster (6K+)** is concentrated in Fall–Summer with clear weather and temperatures ~23°C.

---

## Data Source

Fanaee-T, Hadi, and Gama, Joao. "Event labeling combining ensemble detectors and background knowledge." Progress in Artificial Intelligence (2013), Springer Berlin Heidelberg. DOI: 10.1007/s13748-013-0040-3.
