# Serves the rendered explorer (site/index.html + model.md + llms.txt).
# Railway builds this automatically; PORT is injected at runtime.
FROM caddy:2-alpine
COPY site/index.html site/model.md site/llms.txt /srv/
CMD ["sh", "-c", "caddy file-server --root /srv --listen :${PORT:-8080}"]
