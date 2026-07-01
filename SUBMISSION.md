# 🧠 The Auto-Analyst — Submission Document

---

## 📎 Public Prototype Link

> **Streamlit App:** *(Add your deployed link here after deploying to Streamlit Community Cloud)*
>
> **GitHub Repository:** *(Add your GitHub repo link here after pushing)*

### How to Deploy (Streamlit Community Cloud — Free)

1. Push this project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"Create app"** → select your repo, branch `main`, and main file `app.py`
4. In **Advanced Settings**, add your secret:
   ```
   GROQ_API_KEY = "your_actual_groq_api_key_here"
   ```
5. Click **Deploy** — your public link will be ready in ~2 minutes

---

## 📊 Public Dataset Links

| Dataset | Source | Link |
|---------|--------|------|
| 🌍 **Gapminder** | Plotly Express (built-in) | [plotly.com/python-api-reference](https://plotly.com/python-api-reference/generated/plotly.express.data.html#plotly.express.data.gapminder) |
| 🍽️ **Tips** | Plotly Express (built-in) | [plotly.com/python-api-reference](https://plotly.com/python-api-reference/generated/plotly.express.data.html#plotly.express.data.tips) |
| 😊 **World Happiness Report** | Kaggle | [kaggle.com/datasets/unsdsn/world-happiness](https://www.kaggle.com/datasets/unsdsn/world-happiness) |

---

## 💡 100-Word Summary

**The Auto-Analyst** is an AI-powered data analysis tool built with Streamlit that automatically interprets datasets, detects patterns, and generates natural-language insights. It supports three curated datasets — Gapminder (global development indicators), Tips (restaurant tipping behavior), and the World Happiness Report (country wellbeing from Kaggle). The application performs comprehensive statistical analysis including correlation detection, outlier identification, trend analysis, and K-Means clustering. Its dual-mode insight engine uses GROQ AI (LLaMA 3.3 70B) as the primary generator, with a deterministic rule-based fallback ensuring reliability. Interactive Plotly visualizations with a premium dark theme deliver a professional, intuitive analysis experience.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Statistical Analysis | SciPy, scikit-learn |
| Visualization | Plotly (interactive, dark theme) |
| AI Insights | GROQ API (LLaMA 3.3 70B Versatile) |
| Fallback Engine | Rule-based Natural Language Generation |
| Deployment | Streamlit Community Cloud (free) |

---

## 🔑 Key Features

1. **Three Pre-loaded Datasets** — Gapminder, Tips, World Happiness Report (no file upload)
2. **Smart Filtering** — Dataset-specific sidebar filters (continent, year, region, day, etc.)
3. **Automated Statistical Analysis** — Descriptive stats, data quality assessment, group comparisons
4. **Pattern Detection** — Correlation analysis, IQR-based outlier detection, linear trend detection, K-Means clustering
5. **AI-Powered Insights** — GROQ API generates executive summaries with key findings, patterns, anomalies, and recommendations
6. **Graceful Fallback** — Rule-based insight engine auto-activates if GROQ is unavailable
7. **Interactive Visualizations** — Correlation heatmaps, histograms, box plots, scatter plots, time-series charts, cluster visualization
8. **Premium UI** — Dark gradient theme, glassmorphism cards, animated hover effects, responsive layout
