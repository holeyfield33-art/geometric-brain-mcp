# Deferred Work

Items considered during hardening (Phases 0–9) but intentionally deferred. None are required for MVP launch. Record here to avoid re-debating.

## Infrastructure

- **Database / persistence**: no state is stored between requests. If usage tracking, audit logs, or session state become needed, add a persistence layer. Not needed for stateless diagnostics.
- **Background jobs / task queue**: all operations are synchronous and fast (< 100ms typical). Async processing adds complexity without current need.
- **WebSocket streaming**: spectral analysis results are small JSON payloads. Streaming adds no value at current scale.

## Security / compliance

- **User accounts and organization model**: the API uses simple key-based auth. Multi-tenant identity is out of scope.
- **Enterprise compliance (SOC 2, HIPAA, etc.)**: no PII is processed or stored. Compliance frameworks are premature.
- **Audit logging to external system**: structured logs go to stdout. Shipping to Datadog/Splunk/etc. is a deployment concern, not an application concern.

## Product features

- **Dashboard UI**: the API is headless. A dashboard is a separate product.
- **Alerting platform**: the API provides signals; alerting belongs in the consumer's monitoring stack.
- **Plugin marketplace**: no extension system. Tools are fixed at four.
- **Model serving / fine-tuning**: out of scope. This is a diagnostic tool, not a training platform.
- **Vector search**: not relevant to spectral analysis.
- **Workflow builder**: compose via MCP client or REST calls. No built-in orchestration.

## Packaging

- **Docker image**: Render deployment uses `pip install` directly. A Dockerfile is useful for other platforms but not blocking.
- **Homebrew / system package**: not needed for a Python library.

## Validation / research

- **Committed benchmark results**: `bridge_validation.py` exists as a harness but no results are checked in. Running it and committing results is a research decision, not an engineering task.
- **Advanced eval suite**: current tests verify correctness. Statistical evaluation of spectral metrics against ground truth is research work.
- **Notebook productization**: Colab/Jupyter usage is documented in `bridge_validation.py` comments. A polished notebook is deferred.

## MCP server

- **MCP server authentication**: the MCP server has no built-in auth. In stdio mode this is fine (host process boundary). In HTTP mode, use a reverse proxy. Adding auth to the MCP server is possible but adds complexity for an uncommon deployment.
- **Complex session management**: MCP tools are stateless. Session state belongs in the MCP client.

## Observability

- **Metrics endpoint (Prometheus)**: structured logs are the current observability surface. A `/metrics` endpoint with request counts and latency histograms is a reasonable future addition.
- **Distributed tracing (OpenTelemetry)**: X-Request-ID provides basic tracing. Full OTel integration is deferred.
