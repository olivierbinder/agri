# serve the documentation locally, with live reload
[group('docs')]
docs-serve:
    uv run --group=docs zensical serve --dev-addr localhost:8080

# build the documentation into site/
[group('docs')]
docs-build:
    uv run --group=docs zensical build --clean
