# Sample Q&A

**Live app:** <https://hdfc-nextleap-test.onrender.com/>

Ten questions against the running system. Every answer, source link and
`Last updated from sources` date below was produced by calling the pipeline
(`rag.answer.ask`), not typed by hand. Regenerate it with:

```powershell
.\.venv\Scripts\python.exe -m tools.make_submission_qa
```

**Sources: HDFC Mutual Fund, SEBI and AMFI.** Six of the ten are answered from
HDFC Mutual Fund's own scheme pages on `hdfcfund.com`, and every cited link
points there. The four refusals hand off to SEBI and AMFI's investor-education
material, which is why those two are named in the disclaimer.

Figures are quoted from the source page and never computed. Each answer is at
most three sentences and carries exactly one citation - the one page it was read
from. The `Last updated from sources` date is the day that page was captured, and
it is on every answered question because a scraped page is a snapshot: NAV, AUM
and riskometer levels move without notice, and the AMC edits these pages in
place without changing the URL.


---


## 1. Minimum SIP

**Question:** What is the minimum SIP for HDFC Large Cap Direct Growth?

**Assistant Answer:**

```
The minimum SIP for HDFC Large Cap Fund is ₹ 100.
```

**Official Source Link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct

**Last Updated From Sources:** 2026-09-29


## 2. Expense ratio

**Question:** What is the expense ratio of HDFC ELSS Tax Saver Direct Growth?

**Assistant Answer:**

```
The total expense ratio (TER) for HDFC ELSS Tax Saver Fund is 1.21%.
```

**Official Source Link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct

**Last Updated From Sources:** 2026-09-29


## 3. Exit load

**Question:** What is the exit load on HDFC Large Cap if I redeem within 1 year?

**Assistant Answer:**

```
HDFC Large Cap Fund - Exit Load: In respect of each purchase/switch-in of Units, an Exit Load of 1.00% is payable if Units are redeemed/switched-out within 1 year from the date of allotment. No Exit Load is payable if Units are redeemed/switched-out after 1 year from the date of allotment.
```

**Official Source Link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct

**Last Updated From Sources:** 2026-09-29


## 4. Benchmark

**Question:** What benchmark does HDFC Flexi Cap Direct Growth track?

**Assistant Answer:**

```
The Benchmark for HDFC Flexi Cap Fund is NIFTY 500 Total Returns Index.
```

**Official Source Link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct

**Last Updated From Sources:** 2026-09-29


## 5. Riskometer

**Question:** What is the riskometer level of HDFC Small Cap Fund?

**Assistant Answer:**

```
The Riskometer for HDFC Small Cap Fund is Very High.
```

**Official Source Link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct

**Last Updated From Sources:** 2026-09-29


## 6. ELSS lock-in

**Question:** What is the lock-in period on HDFC ELSS Tax Saver?

**Assistant Answer:**

```
The lock-in period for HDFC ELSS Tax Saver Fund is 3 years.
```

**Official Source Link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct

**Last Updated From Sources:** 2026-09-29


## 7. Capital gains statement

**Question:** How do I download my capital gains statement?

**Assistant Answer:**

```
I can't download or send your statements from here — that needs your account login and your personal data. In your HDFC Mutual Fund account (https://www.hdfcfund.com/) you can download your capital gains statement and Consolidated Account Statement. SEBI's investor-education pages (https://www.sebi.gov.in/) and AMFI (https://www.amfiindia.com/) explain how these statements work.
```

**Refused.** `out_of_corpus` - the question asks for a statement/account document, which the assistant cannot produce

**Official Source Link:** none - nothing was retrieved, so there is nothing to cite.


The document exists, but only behind the holder's own login. The AMC's statement page is in the corpus, so the assistant can name *where* to get it, and it hands off rather than pretending to produce the data.


## 8. Investment advice

**Question:** Should I buy HDFC Small Cap Fund for my retirement?

**Assistant Answer:**

```
I can share facts from the official pages I use, but I can't tell you whether to buy or sell a scheme — that's investment advice, and it's outside what this assistant does. Here's an official guide to understanding scheme risk instead: https://www.sebi.gov.in/ (investor education)
```

**Refused.** `refusal_advice` - advice keyword 'should i'

**Official Source Link:** none - nothing was retrieved, so there is nothing to cite.


The corpus holds all five schemes' expense ratios, so it could rank them by cost - and that ranking would itself be a recommendation. The numbers are available; the judgement is withheld.


## 9. Returns / comparison

**Question:** How much did HDFC Small Cap Fund return last year?

**Assistant Answer:**

```
I don't compute or compare returns. For a scheme's official performance figures, please use the published factsheets from HDFC Mutual Fund (https://www.hdfcfund.com/) or AMFI (https://www.amfiindia.com/).
```

**Refused.** `refusal_performance` - performance keyword 'return'

**Official Source Link:** none - nothing was retrieved, so there is nothing to cite.


A return needs a start date, an end date and a price source, and choosing them is an act of authority this assistant does not have. The refusal hands over to the published factsheet, which does.


## 10. Personal data

**Question:** My PAN is ABCDE1234F. What is the exit load on HDFC Flexi Cap?

**Assistant Answer:**

```
Please don't share personal identifiers like PAN, Aadhaar, account numbers, OTPs, or contact details here — I can't accept, store, or use them. Scheme facts are all public; ask me anything about expense ratio, exit load, SIP, lock-in, or statements.
```

**Refused.** `refusal_pii` - PAN-like ********

**Official Source Link:** none - nothing was retrieved, so there is nothing to cite.


The identifier is not stored, not used and not written to any file. The refusal names the categories it will not accept so the user knows what to do instead.


---


## Also refused, and why

These are not in the ten above. They are listed separately because the honest
answer to "what does it refuse?" needs its own list, and because the interesting
property of this assistant is that it refuses *named* gaps rather than filling
them with a neighbouring fact.

HDFC Mutual Fund does not publish these on a scheme page, so there is nothing in
the corpus to retrieve. All four return the same line, verbatim:

> I couldn't verify that from the available official sources.

| Asked | The assistant says | Why |
|---|---|---|
| Fund manager's name | `out_of_corpus` | the question asks only about 'fund_manager', and no chunk in the corpus carries that fact_key |
| Portfolio P/E | `out_of_corpus` | the question asks only about 'pe_pb_ratio', and no chunk in the corpus carries that fact_key |
| Direct vs Regular plan | `out_of_corpus` | the question asks only about 'plan_variant', and no chunk in the corpus carries that fact_key |
| Minimum lump-sum investment (any scheme except Small Cap) | `out_of_corpus` | the only `minimum_investment` chunk in the corpus belongs to HDFC Small Cap, so asking for any other scheme's is a question the pages do not answer |


The distinction that matters: a refusal is either a **policy** (the four in the
ten above - advice, returns, statements, personal data) or a **gap** (this
table). A policy refusal would happen even with a perfect corpus, because the
answer is withheld on purpose. A gap refusal happens because the source does not
carry the fact, and the point of `known_absent_terms` in `config/app.yaml` is
that the assistant can say so instead of answering the fund-manager question with
the expense ratio.

## Scope of this corpus

68 chunks from 5 HDFC Mutual Fund scheme pages plus the AMC's
Consolidated Account Statement page, captured 29 September 2026.

| Asked | Answered |
|---|---|
| expense ratio / TER | yes, for all five schemes |
| minimum SIP | yes, for all five schemes |
| exit load | yes, for all five schemes |
| ELSS lock-in | yes |
| benchmark | yes, for all five schemes |
| NAV, AUM | yes, for all five schemes |
| riskometer level | yes, for all five schemes |
| minimum lump-sum investment | Small Cap only - the only page that publishes it |
| fund manager, star rating, portfolio P/E | no - not published on a scheme page |
| Direct vs Regular plan | no - the page does not state the plan variant as a fact |
| returns, rankings, recommendations | no - withheld by policy, see above |
| your personal holdings, statements, tax figures | no - requires your login |

**One source per answer, always.** A question that genuinely spans two schemes
is refused as ambiguous rather than answered from two pages.

