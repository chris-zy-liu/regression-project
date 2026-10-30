"""Build a review draft from the repository's existing results and audit."""
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image

styles = getSampleStyleSheet()
styles['Normal'].fontSize = 9
styles['Normal'].leading = 12
doc = SimpleDocTemplate('Milestone2_team_review_draft.pdf', pagesize=(612,792),
                        leftMargin=49, rightMargin=49, topMargin=40, bottomMargin=40)
story = []
def p(text, style='Normal'):
    story.append(Paragraph(text, styles['Normal' if style == 'Caption' else style])); story.append(Spacer(1, 6))
def table(rows):
    t = Table(rows, hAlign='LEFT', colWidths=[270, 70, 90])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf3')),
                           ('GRID',(0,0),(-1,-1),.3,colors.grey),
                           ('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.append(t); story.append(Spacer(1,8))

p('Forecasting small-business AI use', 'Title')
p('ISYE 4031 | Group 1 | TEAM REVIEW DRAFT | October 29, 2026', 'Heading3')
p('Rahul Mehta, Ishaan Mohan, Zhengyu Chris Liu, Timothy Kim, Chau M Duong')
p('1. Question and response to feedback', 'Heading2')
p('Can models forecast the published AI-use percentage for an industry-by-size group three Census releases (about six weeks) ahead more accurately than simple historical forecasts? Such estimates might inform outreach planning, but do not measure an individual firm or the effect of outreach.')
p('The initial TA feedback, relayed by the team, requested a defined prediction target and equation, multiple predictors, a baseline, and attention to the short history. The revised proposal added these. On October 20, notes relayed from the meeting suggested industry comparisons and uncertainty. On October 27–28, a teammate relayed the professor’s interest in model-release timing, benchmark scores and a nonstationary Poisson process; the exact required scope needs team confirmation.')
p('2. Data and preparation', 'Heading2')
p('Census Business Trends and Outlook Survey (BTOS) estimates cover employer businesses in five size bands from 1–4 to 50–99 employees. An observation is a sector, size band and release. The comparable revised-question series runs from release 202524 through 202620. Suppressed values are not set to zero. An early-period screen retains 71 groups in 15 industries; estimates do not represent all small businesses or nonemployers.')
p('The extended analysis used an attributed third-party extraction. An audit against the official workbook downloaded October 20 found exact matches for 1,781 observed use rates, 1,825 expected-use rates and 1,781 standard errors across 2,300 common cells, with matching missingness. One inferred publication date differs by one day. A full workbook run reproduces the baseline and industry-model MAEs. Some richer-model results change because the workbook supplies additional business-condition predictors.')
story.append(PageBreak())
p('3. Exploration and baseline', 'Heading2')
p('The repeated survey estimates are dependent and some cells are suppressed. Measurement uncertainty is larger in some groups, so aggregate accuracy can hide failures. Industry-specific trends and rankings are descriptive; the paper reports overlapping uncertainty among leading industries. Survey noise, a short series, the sample replacement and selection of observable groups limit interpretation.')
story.append(Image('results/figures/industry_growth.png', width=6.4*inch, height=2.7*inch, kind='proportional'))
p('Figure 1. Industry growth estimates from the existing analysis. Intervals are conditional and exploratory; rankings should not be treated as definitive.', 'Caption')
p('The baseline is pooled regression of the future group percentage on current use, industry, size and time. Reference forecasts carry the latest rate forward or average the latest six releases. Training uses only outcomes published by the forecast date. Errors are measured in percentage points using mean absolute error (MAE) and RMSE.')
story.append(PageBreak())
p('4. Forecast results and diagnostics', 'Heading2')
table([['Forecast method','MAE (pp)','RMSE (pp)'],['Latest rate','4.15','5.75'],['Six-release mean','3.39','4.64'],['Pooled regression','3.30','4.55'],['Separate industry regressions','3.04','4.29'],['Elastic net','3.07','4.29'],['Full unregularized regression','5.68','7.27']])
p('Table 1. Extract-based saved results, 706 forecasts across 10 issue dates. All methods use matched observations. The pooled model improves only modestly on the recent average; separate industry models have lower observed MAE. Complex unrestricted regression performs substantially worse. Model selection and comparisons are exploratory and should not be interpreted as proof of reliable future performance.')
story.append(Image('results/figures/errors_by_origin.png', width=6.4*inch, height=2.6*inch, kind='proportional'))
p('Figure 2. Forecast error by issue date in the existing analysis. Dates are shared shocks, not 706 independent experiments.', 'Caption')
p('The tenth issue date was a later check for the original baseline, but its outcome was already available while the expanded methods were designed. Consequently, it is a retrospective extension for those methods, not an untouched prospective test.')
story.append(PageBreak())
p('5. Preliminary answer, limitations and next steps', 'Heading2')
p('Industry models improve observed forecast error over a simple recent average in this dataset, but the evidence remains too limited to establish deployment readiness. Repeated groups, common shocks, data suppression and a short history limit generalization. Industry differences and announcement timing are associations, not causal explanations.')
p('Separate preliminary experiments use Epoch AI data to model monthly language-model publication counts and compare GPQA benchmark-score forecasts. They are feasibility analyses only: catalogues are retrospective, publication dates need not equal product launches, and a benchmark run start does not establish when its score became public. A defensible connection to adoption remains to be developed. Neither experiment is a validated prospective forecast.')
p('Next: verify the professor’s requested scope; evaluate the frozen industry methods on genuinely later releases; check the publication-date discrepancy; investigate comparable benchmark scores and their availability dates; and assess uncertainty and forecast value for a specific outreach decision.')
p('6. Team reflection and reproducibility', 'Heading2')
p('<b>Team input required:</b> confirm each member’s actual contributions, meeting attendance and feedback, the response to each meeting, and named next-stage owners. Confirm the final instructions, deadline and code destination before submission. Do not submit this draft until the group has reviewed these items and the figures and numbers.')
p('Code and saved outputs: github.com/chris-zy-liu/regression-project. The supplied scripts specify dependencies and chronological evaluation. AI assisted code, analysis and drafting; the team must review and take responsibility for all claims.')
p('Sources: U.S. Census Bureau, BTOS data downloads and methodology; Epoch AI, Data on AI models (epoch.ai/data/ai-models); group repository and October 26 changes-and-questions document. References may be expanded outside the page limit.')
doc.build(story)
print('Built Milestone2_team_review_draft.pdf')
