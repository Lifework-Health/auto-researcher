# Optional upstream dependency

Install explicitly during environment provisioning:

```shell
python -m pip install -c constraints/openevolve-0.3.2.lock 'auto-researcher[openevolve]'
```

No runtime installation occurs. Without the extra, DIRECT, Optuna and internal OpenEvolve continue to work; only the upstream adapter returns `upstream_openevolve_dependency_unavailable`. Verify package version and distribution hashes before promotion.
