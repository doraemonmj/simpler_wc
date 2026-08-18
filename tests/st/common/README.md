# Cross-Architecture Scene-Test Assets

This directory contains source bundles jointly owned by the a2a3 and a5 scene
tests. Complete cross-architecture tests remain in their functional directories;
`common` is only for assets whose consumers must evolve and be validated together.

Changes under this directory intentionally trigger both architecture partitions
in CI.

Paths below `common` encode the remaining ownership axes. Assets used by one
runtime live under that runtime's name; assets shared across runtimes live
directly under `common` and are named after their test contract.

Each bundle must document its contract, supported platforms, and consumers. A
bundle must not depend on another bundle under `common`.
