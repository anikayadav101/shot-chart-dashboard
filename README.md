# Shot Analytics

NBA shot profile similarity and efficiency maps built with Python, Streamlit, and the NBA Stats API.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Deploy the live app (Streamlit Community Cloud)

This project is a Streamlit app. GitHub Pages hosts static sites only, so the interactive dashboard is deployed on Streamlit Cloud (free):

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → **New app**
4. Select this repo, branch `main`, main file `app.py`
5. Click **Deploy**

Your public app URL will look like: `https://your-app-name.streamlit.app`

## GitHub Pages

The `docs/` folder is a static landing page. Enable it under **Settings → Pages → Source: Deploy from branch → main → /docs**.
