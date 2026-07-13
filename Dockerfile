FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install .

RUN useradd --create-home gridmind && mkdir -p /app/data /app/artifacts /app/mlruns \
    && chown -R gridmind:gridmind /app
USER gridmind

ENTRYPOINT ["gridmind"]
CMD ["--help"]

