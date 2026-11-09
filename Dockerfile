FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

# Runtime OpenMP library required by scikit-learn wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --create-home --shell /usr/sbin/nologin appuser

COPY requirements.txt .
RUN python -m pip install --retries 10 --timeout 120 --upgrade pip \
    && python -m pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

COPY --chown=appuser:appuser app.py training_data.py ./
COPY --chown=appuser:appuser samples ./samples
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser benchmarks ./benchmarks

USER 10001:10001

CMD ["python", "app.py", "--demo"]
