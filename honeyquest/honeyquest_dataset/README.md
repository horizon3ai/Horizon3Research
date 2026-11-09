# honeyquest_dataset

The upstream Honeyquest dataset, redistributed here so the bundle is
self-contained.

- `queries/` — the 67 multi-document YAML query files (the raw source the baked
  `data/*.json` sets were generated from). `src/loader.py` reads these when you
  call `iter_queries()` to regenerate the JSONs after an upstream refresh.
- `human_marks.csv` — the 47-participant human responses (3,669 responses,
  6,659 marks). The human baseline the paper compares the LLM cohort against.

## Source and license

From Honeyquest by Kahlhofer et al. (Dynatrace), RAID 2024.
Copyright 2024 Dynatrace LLC, licensed under Apache License 2.0. The original
`LICENSE.txt`, `LICENSE_HEADER.txt`, and `CITATION.cff` are kept in this folder.
Upstream: https://github.com/dynatrace-oss/honeyquest
