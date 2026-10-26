# Running and deployment notes

## Local development

Install the package with `pip install -e ".[dev]"`, generate the CSV, train the model, then start Uvicorn. See the README for exact commands.

## Before production

- Add authentication and authorization to the prediction endpoint.
- Use a production ASGI server setup and a managed secrets/configuration solution.
- Pin dependencies and build a reproducible container image.
- Add structured logging, request IDs, metrics, alerting, and health/readiness checks.
- Monitor data quality, drift, latency, errors, and model performance.
- Establish a retraining and rollback process.
- Define data retention, privacy, and access controls.
- Validate and scan serialized model artifacts. Only load trusted joblib files because pickle-based formats can execute code when loaded.
- Use TLS, rate limiting, and request-size limits at the gateway.
