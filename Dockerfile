# Python 3.13: fastembed's BM25 model uses py_rust_stemmers, a Rust extension
# that segfaults on Python 3.14 due to PyO3/C API incompatibility.
# Pin to 3.13 until upstream adds support (qdrant/fastembed#576).
FROM python:3.13-slim

WORKDIR /app

# Stage 1: Install dependencies only (cached layer).
# Build a throwaway package from minimal source so pip resolves all deps,
# then remove the incomplete package itself.
COPY pyproject.toml README.md ./
COPY mnemory/__init__.py mnemory/__init__.py
RUN pip install --no-cache-dir ".[all]" \
    && pip uninstall -y mnemory

# Stage 2: Copy full source and install the package (no deps needed).
COPY mnemory/ mnemory/
RUN pip install --no-cache-dir --no-deps "mcp<2" "httpx-sse" "pydantic-settings" "python-dotenv" .

# Pre-download BM25 sparse embedding model into the image so containers
# start without network dependency (Kubernetes pods lose cache on restart).
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"

# Data directory for vector store, artifacts, and history.
# Override ~/.mnemory default so Docker volumes mount cleanly at /data.
ENV DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8050/health')" || exit 1

CMD ["mnemory"]
