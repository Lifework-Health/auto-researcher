# ADR 017: Pin upstream OpenEvolve v0.3.2

Status: accepted.

Use the Apache-2.0 upstream repository `algorithmicsuperintelligence/openevolve`, tag `v0.3.2`, commit `411fb59c886c18704caaffb611e17cf9e7d824d2`, wheel SHA-256 `df998b0731d9c1a80883b4aae452cc43405a3e9c61b46d676d06235b4db49366`, and sdist SHA-256 `cd41800ab54734d02a895892615a7f4b9240a6f307c82fc1df7335e89b546599`. Python support is ≥3.10. It is an optional, unvendored dependency. `constraints/openevolve-0.3.2.lock` records the tested dependency set; runtime installation is forbidden.

The licence permits this narrow unmodified interoperability. `NOTICE` provides attribution. Runtime validation checks exact package version, installed wheel RECORD aggregate and expected dataclass API. Identity or API drift fails closed.
