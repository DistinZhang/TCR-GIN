# About The Streamlit App

This web application is a Streamlit multipage app. It does not require a separate backend service or frontend framework.

## Structure

```text
webapp/
├── app.py                    # Home
├── navigation.py             # shared sidebar navigation
└── pages/
    ├── 1_Network_Setup.py
    ├── 2_Real_Time.py
    └── 3_User_Guide.py
```

Run it from the repository root:

```bash
streamlit run webapp/app.py
```

## Pages

- `Home`: status overview and quick start
- `Network Setup`: load the network, collapse target, and model
- `Monitoring`: run attacks and monitor warning metrics
- `User Guide`: English and Chinese user guide

Streamlit serves the app at `http://localhost:8501` by default.
