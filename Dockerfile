# Serves the rendered explorer: index.html + model.md + llms.txt plus the
# per-week history snapshots under /history (as-of models & backtests).
# Railway builds this automatically; PORT is injected at runtime.
FROM caddy:2-alpine
COPY site/ /srv/
RUN rm -f /srv/template.html
CMD ["sh", "-c", "caddy file-server --root /srv --listen :${PORT:-8080}"]
