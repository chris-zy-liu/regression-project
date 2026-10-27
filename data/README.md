# Data provenance

The original October 3, 2026 Census workbook was not included in the supplied project. The direct Census download and the side browser returned timeouts during acquisition on October 26, 2026.

`btos_ai_extract_2026-10-15.csv` is an unmodified download from The Uptime Report:

https://theuptimereport.com/wp-content/uploads/2026/10/small-business-ai-adoption-census-btos.csv

The publisher states that it extracted the Census BTOS workbooks on October 15, 2026. The extract contains published AI-use estimates and standard errors, collection dates, and national, size, sector, state, and sector-by-size classifications. Its final collection period is September 21–October 4, 2026 (release 202620, published October 8). It omits suppressed estimates and does not contain other business-condition survey questions. Missing cells are reconstructed as unavailable; their original suppression flags cannot be distinguished.

**Attribution:** U.S. Census Bureau, Business Trends and Outlook Survey; redistribution by The Uptime Report, “Small Business AI Adoption Is About 1 in 4, Not 3 in 4: 23.8% of Firms Use AI, Census Data Shows,” October 2026, https://theuptimereport.com/small-business-ai-adoption/ . The redistributed extract is offered under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is included unchanged. Transformations and estimates in `results/` are this project's calculations.

The accompanying `.provenance.json` records the retrieval date, source URL, and SHA-256. Agreement with the original report's cell counts and forecasting results is tested in `test_analysis.py`. This agreement validates the relevant historical calculations; it does not establish byte identity with the missing workbook or independent verification of every extract cell.

`ai_releases.csv` contains six illustrative release announcements verified against provider publications. It is a selected chronology, not an exhaustive model-release database. These events were selected before estimating event-window changes in this implementation. Their timing overlaps, and the earliest events have insufficient pre-event observations in the comparable survey series.

Use `python download_data.py` to retrieve a Census workbook when the host is available. Supply it with `--workbook` to run the same pipeline directly on the original schema. The workbook path also enables additional first-answer shares from non-AI survey questions. Those predictors are unavailable in the included CSV run and are not represented as observed results.
