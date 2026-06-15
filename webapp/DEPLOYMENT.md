# Deployment

## Local Run

From the repository root:

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

The local URL is usually `http://localhost:8501`.

## Windows Helper

```bash
python run_webapp.py
```

or run `run_webapp.bat`.

## Streamlit Cloud

1. Push the repository to GitHub.
2. Create a new Streamlit Cloud app.
3. Select the repository.
4. Set the main file path to `webapp/app.py`.
5. Deploy.

Large local model directories are usually not suitable for Streamlit Cloud uploads. For public demos, provide small example checkpoints or allow users to upload their own ZIP archive and YAML config.

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

The app theme and toolbar behavior are configured in `webapp/.streamlit/config.toml`. The toolbar is minimized, and default multipage navigation remains available as a compatibility fallback beside the custom sidebar navigation.
