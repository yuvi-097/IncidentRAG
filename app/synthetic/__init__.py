"""Deterministic synthetic NovaCart dataset.

Everything here is fictional. The generator simulates a year of NovaCart
operations: code, pull requests and deployments per service; incidents caused by
specific deployments (or by traffic / infrastructure); the rollbacks and hotfixes
that resolved them; runbooks, technical docs and postmortems; and logs.

Entry point: ``app.synthetic.generator.generate_dataset``.
"""
