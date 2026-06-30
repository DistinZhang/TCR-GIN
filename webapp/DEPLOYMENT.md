# Deployment

## Local Run

From the repository root:

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

The local URL is usually `http://localhost:8501`.

## Streamlit Cloud

1. Push the repository root to GitHub. For this release package, the repository
   root is the inner `TCR-GIN/` directory, not its parent staging directory.
2. Create or edit a Streamlit Community Cloud app.
3. Select repository `DistinZhang/TCR-GIN` and the deployment branch, usually
   `main`.
4. Set the main file path to `webapp/app.py`.
5. Deploy or reboot the app. Streamlit Cloud installs dependencies from
   `webapp/requirements.txt` because the entry file is inside `webapp/`.

Current public deployment: [tcr-gin-early-warning.streamlit.app](https://tcr-gin-early-warning.streamlit.app/)

The public README embeds the walkthrough recording from
[GitHub attachment](https://github.com/user-attachments/assets/48f8d823-1c47-4384-8a0a-8692727953ed).
An archival copy is stored at [`webapp/assets/Supplementary Movie - Step-by-step early warning of network breakdown with collapse distance.mp4`](webapp/assets/Supplementary Movie - Step-by-step early warning of network breakdown with collapse distance.mp4).

Large local model directories are usually not suitable for Streamlit Cloud
uploads. This repository keeps only the small transport-demo checkpoints under
`webapp/examples/transport_demo/`; larger training checkpoints and result
artifacts remain in the Zenodo archive.

## Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r webapp/requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "webapp/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t tcr-gin-webapp .
docker run -p 8501:8501 tcr-gin-webapp
```

## App Configuration

The app theme and toolbar behavior are configured in `.streamlit/config.toml`. The toolbar is minimized, and default multipage navigation remains available as a compatibility fallback beside the custom sidebar navigation.
