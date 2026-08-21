# Serves the JSON API (/api/v1, /api/docs) and the rendered explorer site
# (index.html + model.md + llms.txt + /history snapshots) from one process.
# Model artifacts are downloaded from the latest GitHub release at boot, so
# a restart is enough to pick up a new weekly build. PORT is injected by
# Railway at runtime.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
COPY site ./site
RUN rm -f site/template.html
ENV RISKPRISM_SITE=/app/site RISKPRISM_ARTIFACTS=/app/artifacts
CMD ["sh", "-c", "uvicorn riskprism.api_server:app --host 0.0.0.0 --port ${PORT:-8080}"]
