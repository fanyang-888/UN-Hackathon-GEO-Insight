# Declared Data Sources

Per hackathon rules, all external data sources are declared below.

## Primary (provided by UNOCHA)

| Source | URL | Used for |
|--------|-----|----------|
| Humanitarian Needs Overview (HNO) | https://data.humdata.org/dataset/global-hpc-hno | People in need by country & sector |
| Humanitarian Response Plans (HRP) | https://data.humdata.org/dataset/humanitarian-response-plans | Funding targets and HRP status |
| Global Requirements & Funding (FTS) | https://data.humdata.org/dataset/global-requirements-and-funding-data | Financial tracking (requested vs received) |
| CBPF Pooled Funds | https://cbpf.data.unocha.org/ | Country-based pooled fund allocations |
| COD Population | https://data.humdata.org/dataset/cod-ps-global | Population baseline |

## Additional (declared)

| Source | URL | Used for |
|--------|-----|----------|
| INFORM Global Crisis Severity Index | https://data.humdata.org/dataset/inform-global-crisis-severity-index | Independent severity multiplier in gap score |
| HDX HAPI API | https://hdx-hapi.readthedocs.io/en/latest/ | API access layer for HNO and population data |
| FTS API v2 | https://api.fts.unocha.org/ | Fallback funding data when CSV unavailable |

## External Models / APIs

| Service | Used for |
|---------|----------|
| Anthropic Claude API (claude-sonnet-4-6) | Natural-language query decomposition, per-crisis briefing note generation |

## Notes

- All need and funding figures are grounded in the provided datasets; no figures are fabricated or inferred beyond documented data.
- When data is missing or stale (>18 months), rows are flagged as low-confidence in the output.
- The INFORM Severity Index is used only as a multiplier to weight need urgency — it does not replace HNO figures.
