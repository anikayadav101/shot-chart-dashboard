# shot-chart-dashboard

NBA shot analytics dashboard: compare player shot-selection profiles with cosine similarity and visualize shot efficiency on a 2D court map (hex bins vs league average). Built with Python, Streamlit, scikit-learn, and the NBA Stats API.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Deploy the live app (Streamlit Community Cloud)

This project is a Streamlit app. GitHub Pages hosts static sites only, so the interactive dashboard is deployed on Streamlit Cloud (free):

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub → **New app**
3. Select this repo, branch `main`, main file `app.py`
4. Click **Deploy**

Your public app URL will look like: `https://shot-chart-dashboard.streamlit.app`

## GitHub Pages

The `docs/` folder is a static landing page. Enable it under **Settings → Pages → Source: GitHub Actions**.
