# 🧠 The Auto-Analyst — AI-Powered Data Insights Engine

An intelligent data analysis application that automatically interprets datasets, surfaces hidden patterns, and writes its own natural-language insights using AI.

## 🚀 Live Demo

> Run locally with `streamlit run app.py`

## 📊 Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| 🌍 **Gapminder** | [Plotly Built-in](https://plotly.com/python-api-reference/generated/plotly.express.data.html) | Country-level life expectancy, GDP per capita, and population (1952–2007) |
| 🍽️ **Tips** | [Plotly Built-in](https://plotly.com/python-api-reference/generated/plotly.express.data.html) | Restaurant tipping behavior — bill, tip, demographics |
| 😊 **World Happiness** | [Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness) | World Happiness Report scores with GDP, social support, freedom |

## 💡 How It Works

1. **Select** one of 3 pre-loaded datasets from the sidebar
2. **Filter** data using dataset-specific controls (continent, year, region, etc.)
3. **Explore** 4 analysis tabs: Overview, Analysis, Correlations, AI Insights
4. **Generate** AI-powered insights with one click using GROQ API (llama-3.3-70b-versatile)

### AI Insight Engine (Dual-Mode)
- **Primary**: GROQ API sends statistical findings to an LLM for narrative synthesis
- **Fallback**: Rule-based NLG automatically activates if GROQ is unavailable

## 🛠️ Setup

```bash
# Clone the repository
git clone <repo-url>
cd auto_analyst

# Install dependencies
pip install -r requirements.txt

# Set up your GROQ API key
# Create a .env file with: GROQ_API_KEY=your_key_here

# Run the app
streamlit run app.py
```

## 📝 100-Word Summary

**The Auto-Analyst** is an AI-powered data analysis tool built with Streamlit that automatically interprets datasets, detects patterns, and generates natural-language insights. It supports three curated datasets — Gapminder (global development indicators), Tips (restaurant tipping behavior), and the World Happiness Report (country wellbeing from Kaggle). The app performs comprehensive statistical analysis including correlation detection, outlier identification, trend analysis, and K-Means clustering. Its dual-mode insight engine uses GROQ AI (llama-3.3-70b-versatile) as the primary generator, with a rule-based fallback for reliability. Interactive Plotly visualizations with a premium dark theme provide an intuitive, professional analysis experience.

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **Data**: Pandas, NumPy
- **Statistics**: SciPy, scikit-learn
- **Visualization**: Plotly
- **AI Insights**: GROQ API (llama-3.3-70b-versatile)
- **Fallback**: Rule-based Natural Language Generation

## 📄 License

MIT License
