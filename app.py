"""
Entry point for Streamlit Cloud.
Streamlit Cloud runs the file specified in .streamlit/config.toml
or defaults to the file passed via --server. We keep this thin wrapper
so the import path is always the repo root.
"""
from app.streamlit_app import *  # noqa: F401,F403
