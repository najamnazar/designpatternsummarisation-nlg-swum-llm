# Reviewer feedback on fact preservation: discussion notes

Notes from a discussion on 2026-10-18 about the reviewer's comments on the fact-preservation evaluation and on the annotation spreadsheet `design_pattern_preservation_annotation.xlsx`. They record:

- what the reviewer asked;
- what the repo's existing results already answer;
- the status reply prepared for the reviewer;
- what the annotation spreadsheet would need to change so the student's labels can be compared with the LLM judge.

No code or result files were changed during the discussion. All figures come from the existing CSVs in `evaluation-results/`, except one that was computed by hand (see [Section 3.4](#34-evidence-quotes-not-found-verbatim)).

---

## 1. Reviewer comments, verbatim

**Comment A: first comment, fact checklist**

> Since DPS already extracts structured information into JSON, could we turn this into a checklist of facts for each file? For example, for an Observer implementation, the checklist might include which class acts as publisher, which classes are observers, and how notifications work. We could then use an LLM to check whether each fact is present, missing, or incorrectly described in a generated summary.

**Comment B: length budgets**

> We can use the existing 20/40/60/80-word summaries, but instead of looking only at similarity scores, also measure how many design-pattern facts are preserved at each length. This would let us investigate which information gets lost first (pattern names, roles, responsibilities, relationships), whether the results differ between models, and how NLG and SWUM compare at similar lengths.

**Comment C: follow-up on the fact-preservation results**

> I'd report preservation separately by fact type, in particular to distinguish design-pattern roles from generic structural information. Most of the 678 facts come from the parser, which may explain NLG's high preservation rate.
>
> - It would be good to report summary lengths as well as the preservation rates and analyze how preservation changes across the 20/40/60/80-word limits.
>
> - It would be good to investigate the evidence quotes that weren't found verbatim in the summaries, to make sure the judge's classifications are reliable.

**Comment D: annotation spreadsheet**

> The annotation spreadsheet looks good, but if we want to measure agreement with the LLM judge, the student should evaluate the same extracted facts, while also checking their correctness against the source code. We could allow additional facts to be recorded separately.

---

## 2. How the fact-preservation judge works (background)

- **Script:** `python/judge_fact_preservation.py`.
- **Judge model:** Llama-4-Maverick, temperature 0.
- **Labels:** each fact gets PRESERVED, OMITTED or INCORRECT, with an evidence quote from the summary and a short justification.
- **Quote check:** `_quote_in` checks the quote against the summary after normalising case, whitespace, backticks and surrounding quotes. The result is the `evidence_found` column.
- **Fact types:**
  - `role` comes from the detector.
  - `kind`, `inheritance`, `constructor`, `field`, `method` and `call` come from the parser.
- **Caps per class:** role 5, field 5, constructor 5, method 3, call 5. The caps shape the mix of fact types.
- **Corpora judged:** `METHODS` is imported from `rank_summaries_four_way` and fixed to the four default corpora (DPS_NLG, DPS_SWUM, DPS_LLM GPT, DPS_LLM Mistral). The 20/40/60/80-word corpora have **not** been judged. Judging them would need a code change.
- **Size:** 678 facts across 150 classes. 543 facts come from the parser and 135 from the detector; the role facts cover 65 classes.

### Result files

| File | Contents |
|---|---|
| `fact_preservation_facts.csv` | One row per extracted fact: project, design_pattern, file, class_name, fact_id, fact_type, source, fact_text |
| `fact_preservation_verdicts.csv` | One row per fact × method: … fact_id, fact_type, source, fact_text, label, evidence, evidence_found, justification, in_comparison |
| `fact_preservation_results.csv` | Totals per method, including evidence-not-found counts |
| `fact_preservation_summary.csv` | Preservation by all facts, by source and by fact type |
| `fact_preservation_detail.csv` | Per-class detail |
| `summary_length_stats.csv` | Word-count statistics for every corpus, including the budget corpora |

---

## 3. Analysis of comments A–C

### 3.1 Fact checklist (comment A): done

`judge_fact_preservation.py` implements what the reviewer proposed.

The gap is that the reviewer's example was pattern-level: who the publisher is, who the observers are, and how notification works. Most of the checklist is generic structure instead (543 parser facts). Only 135 facts are role facts, and they cover 65 of the 150 classes. That is why the reviewer then asked for a split by fact type.

Pattern-specific role facts exist, for example:
- "EventManager is a publisher for EventListener in the Observer pattern."
- "EmailNotificationListener is a concrete observer for EventListener"
- "EventListener is the observer interface"

**Classes with role facts, by pattern:**

| Pattern | Classes with role facts |
|---|---|
| AbstractFactory | 22 / 28 |
| Adapter | 6 / 14 |
| Decorator | 3 / 14 |
| Facade | **0 / 18** |
| FactoryMethod | 12 / 20 |
| Memento | 3 / 10 |
| Observer | 8 / 14 |
| Singleton | 4 / 12 |
| Visitor | 7 / 19 |

**Not available from DPS:**
- **"How notifications work".** DPS does not extract this as a fact type. Call facts such as "Time calls notifyObservers" cover it only partly.
- **A separate "pattern name" fact type.** Pattern names appear inside the role facts. Pattern naming is scored separately in Table V (`final-results/rq1/table05_pattern_naming_accuracy.csv`).

**Caveat:** `final-results/README.md` notes that some detector role assignments are implausible, for example Button and CheckBox labelled as Memento originators. It also says roles are assigned for only 63 of the 150 classes in the DPS LLM JSON.

### 3.2 Preservation by fact type (comment C, point 1): done in the data

Source: `fact_preservation_summary.csv`.

**Overall preserved rate (678 facts):** NLG 86.4%, GPT 68.3%, Mistral 60.6%, SWUM 20.2%.

**Preserved rate by source and fact type:**

| Preserved rate | NLG | GPT | Mistral | SWUM |
|---|---|---|---|---|
| Role (detector, n = 135, 65 classes) | **83.0%** | 40.7% | 51.1% | 54.1% |
| Parser facts (n = 543) | **87.3%** | 75.1% | 62.9% | 11.8% |
| Kind (36) | 100% | 91.7% | 97.2% | 2.8% |
| Inheritance (111) | 100% | 88.3% | 79.3% | 3.6% |
| Method (260) | 93.1% | 76.5% | 66.9% | 22.7% |
| Call (82) | 100% | 39.0% | 18.3% | 0% |
| Constructor (24) | **0%** | 87.5% | 25.0% | 0% |
| Field (30) | **10.0%** | 83.3% | 80.0% | 0% |

**Incorrect rate on role facts:** NLG 10.4%, GPT 6.7%, Mistral 6.7%, SWUM 2.2%.

**Interpretation:**
- **The reviewer is partly right.** NLG verbalises the same parser output the facts are extracted from, so it scores 100% on kind, inheritance and call facts almost by construction. This should be stated as a threat to validity.
- **NLG still leads on role facts** (83% against 41–54%). The NLG templates also render detector output, so the same caveat applies there, though less strongly.
- **NLG has the highest incorrect rate on roles** (10.4%). This should be reported.
- **There is a new finding.** NLG almost never mentions constructors or fields, while GPT keeps 83–88% of them. The pipelines keep different kinds of information; there is no single winner.

### 3.3 Summary lengths and the 20/40/60/80-word budgets (comment B, and comment C point 2): half done

**Done.** `summary_length_stats.csv` reports lengths for every corpus. Similarity at each budget is in Table XV (`final-results/rq4/table15_length_control.csv`).

**Mean words per summary:**

| Corpus | Mean words | Notes |
|---|---|---|
| NLG | 107.9 | median 47.5, max 1777, SD 215.7 |
| SWUM | 13.85 | |
| GPT (default) | 35.4 | |
| Mistral (default) | 40.6 | |
| Human reference | 29.7 | |
| GPT at budgets 20 / 40 / 60 / 80 | 16.1 / 29.4 / 40.8 / 47.9 | |
| Mistral at budgets 20 / 40 / 60 / 80 | 18.9 / 35.1 / 46.0 / 59.0 | |

**Not done.** Fact preservation has not been measured on the budget corpora. The reviewer explicitly asked to go beyond similarity scores, so the length file alone does not answer the comment.

The eight budget corpora already exist in `output/summary-output/llm_summaries_{gpt,mistral}_{20,40,60,80}.csv`. The checklist is per file, so the same 678 facts would apply. The run would need:
- a change to the judge's corpus list;
- 8 × 150 = 1,200 extra judge calls.

**If the analysis is run later:**
- **Map the fact types onto the reviewer's four categories:**
  - pattern name and role → `role`;
  - responsibilities → `method`, `constructor`, `field`;
  - relationships → `inheritance`, `call`;
  - `kind` → identity.
- **Use actual word count, not the nominal budget.** The models undershoot: GPT-80 averages 47.9 words and Mistral-80 averages 59.0.
- **Natural length-matched pairs:**
  - SWUM (13.9 words) with GPT-20 (16.1) and Mistral-20 (18.9);
  - NLG's median (47.5) with GPT-80 (47.9) and Mistral-60 (46.0).
- **The default-length results already hint at what is lost first.** The LLMs keep inheritance and kind well but drop calls (GPT 39%, Mistral 18%) and roles (41–51%).
- **Table IV (length-matched comparison) is still pending** (`final-results/rq1/table04_length_matched_PENDING.csv`). The README calls the 20-word corpus "the closest available proxy".

**Length as a confound.** NLG summaries are much longer on average, so they have more room to keep facts. Mean and median length should sit next to each preservation rate.

### 3.4 Evidence quotes not found verbatim (comment C, point 3): automatic check done, manual review not done

Source: `fact_preservation_results.csv` and `fact_preservation_verdicts.csv`.

**Unmatched quotes:** 239 in total (NLG 114, GPT 53, Mistral 57, SWUM 15).

**What they look like:**
- **Fragments joined with "...".** This covers 164 of the 239, including 90 of NLG's 114. Example: `MacOSFactory ... overrides createButton method of GUIFactory`. Each fragment is usually verbatim.
- **Small rewordings.** For example "it implements CheckBox", or curly apostrophes.
- **A few real paraphrases or doubtful quotes.** For example "provides factory methods for creating Button objects". A few OMITTED verdicts also carry quotes.

**Strict lower bound.** Counting a fact as preserved only when its quote matched verbatim gives:

| | NLG | GPT | Mistral | SWUM |
|---|---|---|---|---|
| Strict preserved rate | 70.6% | 61.1% | 54.4% | 18.4% |

The ranking of the four pipelines does not change. These figures were computed by hand from `fact_preservation_verdicts.csv`, by keeping only the verdicts with `evidence_found` true. **They are not saved in any repo file.**

**Not done:** a manual review of the 239 quotes. One possible scheme is four groups:
1. fragments joined with "...", each verbatim (this group can be checked automatically by splitting on "...");
2. minor rewording;
3. a paraphrase the summary supports;
4. not supported by the summary.

---

## 4. Status reply to the reviewer (comments A–C)

The first draft of this reply had placeholders such as [RESULT] and claimed the budget run and the manual quote audit had been done. **It was rejected.** The reply has to be a plain status: what is done, what is missing and what is not available, with no placeholders, no invented results and no promises of new runs. The agreed version follows.

---

**Response to Reviewer**

We thank the reviewer for the suggestions. Below, for each one, we state what has been done, what has not, and what our data cannot provide.

**1. Checklist of facts from the DPS JSON, checked by an LLM**

*Done.* We built a checklist of facts for each file from the DPS JSON: 678 facts across 150 classes. An LLM judge (Llama-4-Maverick, temperature 0) labelled each fact as preserved, omitted or incorrect, and gave a supporting quote from the summary.

The pattern-specific facts the reviewer describes are included as role facts. For example: "EventManager is a publisher for EventListener in the Observer pattern", "EmailNotificationListener is a concrete observer for EventListener" and "EventListener is the observer interface".

*Not available:*
- **Role facts for every class.** The detector assigns roles in only 65 of the 150 classes (135 role facts). Some patterns are not covered at all: Facade (18 classes) has no role facts, and Observer has them for 8 of its 14 classes.
- **How notifications work, as a separate fact.** DPS does not extract this. It is covered only partly, through method-call facts such as "Time calls notifyObservers".

**2. Preservation by fact type, separating pattern roles from structural information**

*Done.* Of the 678 facts, 543 are structural facts from the parser and 135 are pattern roles from the detector.

| Preserved | NLG | GPT | Mistral | SWUM |
|---|---|---|---|---|
| Pattern roles (135) | 83.0% | 40.7% | 51.1% | 54.1% |
| Structural facts (543) | 87.3% | 75.1% | 62.9% | 11.8% |

As the reviewer suspected, NLG's high rate on structural facts comes partly from NLG describing the same parser output the facts are taken from. It preserves 100% of kind, inheritance and call facts. NLG also leads on pattern roles, but it has the highest rate of incorrectly described roles: 10.4%, against 6.7% for both LLMs. NLG rarely mentions constructors (0%) or fields (10%), while GPT preserves 87.5% and 83.3% of them.

**3. Summary lengths and preservation across the 20/40/60/80-word limits**

*Done:* We report summary lengths for all corpora, including the 20/40/60/80-word ones, together with similarity scores at each limit. The mean lengths are 108 words for NLG (median 47.5), 35 for GPT, 41 for Mistral and 14 for SWUM.

*Not done:* We have not measured fact preservation on the 20/40/60/80-word summaries. Our preservation results cover only the default summaries of each pipeline. We therefore cannot yet show:
- which information is lost first as the limit shrinks;
- whether this differs between models;
- how NLG and SWUM compare with the LLMs at similar lengths.

The summaries needed for this analysis exist; the analysis itself has not been run.

**4. Evidence quotes not found verbatim in the summaries**

*Done:* Every quote is checked automatically against its summary. 239 were not found verbatim: 114 for NLG, 53 for GPT, 57 for Mistral and 15 for SWUM. In most of these (164 of 239), the judge joined separate parts of the summary with "...". The rest are mainly rewordings or paraphrases.

As a check, we counted a fact as preserved only when its quote matched the summary verbatim. Preservation then drops to 70.6% for NLG, 61.1% for GPT, 54.4% for Mistral and 18.4% for SWUM. The ranking of the four pipelines is unchanged.

*Not done:* We have not reviewed the 239 quotes by hand to confirm whether each one is supported by its summary.

---

## 5. Annotation spreadsheet (comment D)

### 5.1 Current design of `design_pattern_preservation_annotation.xlsx`

The spreadsheet was read without changes. openpyxl is not installed, so it was read through a zip and XML parser.

- **Sheets:** Instructions, Pattern Roles, Files, F01–F30, All Annotations.
- **Sample:** 30 files, stratified by 3 repositories (AbdurRKhalid, JamesZBL, SpringFramework) and 9 patterns.
- **Code shown:** each file sheet has the GitHub link (SamSike/DPS-Corpus) and the full source code.
- **Summaries:** each file sheet shows summaries A–D in a random order per file. `evaluation-results/manual_annotation_key.csv` maps F01–F30 and A–D back to DPS_NLG, DPS_SWUM, DPS_LLM (GPT) and DPS_LLM (Mistral). The summary texts match what the judge scored; F01's GPT text was checked against `llm_summaries_gpt.csv`.
- **Annotator's own checklist.** Before reading the summaries, the annotator writes a checklist from the code: Pattern (given), Role, Responsibility 1–3, Relationship 1–3 (up to 7 items), plus an Unsupported-claim row (Yes/No).
- **Judgements.** The annotator then judges each summary A–D for each item as Preserved, Omitted, Incorrect or N/A, with notes.
- **Files sheet:** progress counters "Items listed (of 7)" and "Judgements made (of 36)".
- **Pattern Roles sheet:** GoF participant names.
- **All Annotations sheet:** File ID, Repository, Design pattern, File, Item #, Item type, Item identified in the code, Summary, Judgement, Notes.
- **Worked example:** EventManager (Observer).
- **State:** empty. There is no annotator name and every status is "None", so redesigning it loses no work.

### 5.2 The main mismatch

The annotator writes their own items, so those items don't line up with the facts the judge scored, and agreement can't be computed. The reviewer wants the student to label the same extracted facts as the judge.

| | Current spreadsheet | What the reviewer asks for |
|---|---|---|
| What gets judged | The student's own checklist: Pattern, Role, 3 Responsibilities, 3 Relationships | The extracted DPS facts for each file, the same ones the judge saw |
| Checking facts against the code | Not done; the student writes the items, so they're assumed correct | The student marks each extracted fact correct or incorrect against the source |
| Extra facts | They're the whole checklist | Kept, but recorded separately and not used for agreement |
| Comparable with the judge? | No, the items don't match the judge's fact IDs | Yes, one human label for each judge label |

### 5.3 Overlap between the annotation sample and the judge's facts

- **Coverage:** all 30 annotation files are among the 150 judged classes.
- **Facts:** 141 in total, 1–15 per file (mean 4.7, median 4).
  - By type: method 62, inheritance 24, call 19, role 16, constructor 7, field 7, kind 6.
- **Judge labels to compare against:** 564 (141 facts × 4 summaries).

**Judge verdicts on these 30 files:**

| Method | Preserved | Omitted | Incorrect |
|---|---|---|---|
| DPS_NLG | 118 | 19 | 4 |
| DPS_LLM (GPT) | 93 | 47 | 1 |
| DPS_LLM (Mistral) | 90 | 51 | 0 |
| DPS_SWUM | 26 | 112 | 3 |

**Facts per file** (call / constructor / field / inheritance / kind / method / role):

| File | call | ctor | field | inh | kind | method | role |
|---|---|---|---|---|---|---|---|
| F01 MacOSFactory | 0 | 0 | 0 | 1 | 0 | 2 | 1 |
| F02 WindowsButton | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| F03 YoungTeamFactory | 0 | 0 | 0 | 1 | 0 | 3 | 1 |
| F04 ClientHttpRequestFactory | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| F05 RoundHole | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| F06 Driver | 0 | 1 | 1 | 1 | 0 | 1 | 2 |
| F07 IsolationLevelDataSourceAdapter | 4 | 0 | 0 | 1 | 0 | 3 | 0 |
| F08 DataSourceDecorator | 0 | 1 | 1 | 1 | 0 | 2 | 0 |
| F09 CarpenterOperation | 0 | 0 | 0 | 1 | 0 | 3 | 1 |
| F10 WebSocketHandlerDecorator | 1 | 0 | 0 | 1 | 0 | 3 | 0 |
| F11 BitrateReader | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| F12 CourseFacade | 0 | 0 | 1 | 0 | 0 | 3 | 0 |
| F13 TransactionTemplate | 1 | 0 | 0 | 3 | 0 | 3 | 0 |
| F14 WindowsDialog | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| F15 FoodType | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| F16 AdaptableJobFactory | 0 | 0 | 0 | 1 | 0 | 3 | 0 |
| F17 JobDetailFactoryBean | 2 | 0 | 0 | 4 | 0 | 3 | 0 |
| F18 TextWindow | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| F19 Flower | 5 | 2 | 1 | 2 | 0 | 3 | 2 |
| F20 SavepointManager | 0 | 0 | 0 | 0 | 1 | 3 | 0 |
| F21 EventListener | 0 | 0 | 0 | 0 | 1 | 1 | 4 |
| F22 Northern | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| F23 ApplicationListener | 0 | 0 | 0 | 1 | 1 | 3 | 0 |
| F24 Singleton | 0 | 1 | 1 | 0 | 0 | 1 | 0 |
| F25 LazyInitializationDirector | 0 | 1 | 1 | 0 | 0 | 1 | 1 |
| F26 SmartInitializingSingleton | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| F27 Mouse | 1 | 0 | 0 | 2 | 0 | 1 | 1 |
| F28 BossVisitor | 0 | 0 | 0 | 1 | 0 | 3 | 0 |
| F29 Unit | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| F30 BeanDefinitionVisitor | 5 | 0 | 0 | 0 | 0 | 3 | 0 |

**Example, F01 MacOSFactory (4 facts):**
- role: "MacOSFactory is a concrete factory for GUIFactory in the Abstract Factory pattern."
- inheritance: "MacOSFactory implements GUIFactory."
- method: "MacOSFactory overrides method createButton."
- method: "MacOSFactory overrides method createCheckBox."

**Role facts are thin.** There are 16 role facts in only 11 of the 30 files. The 19 files with **no** role fact are:

AdaptableJobFactory, ApplicationListener, BeanDefinitionVisitor, BitrateReader, BossVisitor, ClientHttpRequestFactory, CourseFacade, DataSourceDecorator, FoodType, IsolationLevelDataSourceAdapter, JobDetailFactoryBean, RoundHole, SavepointManager, Singleton, SmartInitializingSingleton, TextWindow, TransactionTemplate, Unit, WebSocketHandlerDecorator.

### 5.4 Changes needed to the spreadsheet

These are not yet made.

**1. File sheets (F01–F30): add a block of extracted facts**
- **One row per extracted fact.** Show the fact_id, fact_type, source and the exact fact_text from `fact_preservation_facts.csv`.
- **A new column, "Fact correct in the code?"** Values: Correct / Incorrect / Can't tell, with a note for anything not Correct. This checks DPS's extraction separately from the summaries, including detector role errors.
- **Columns A–D labelled Preserved / Omitted / Incorrect for each fact.** Use the same three labels and the same definitions as the judge prompt in `judge_fact_preservation.py`, not the current Instructions wording; otherwise the two judge different things. Drop N/A for these rows, since every fact is real.
- **Facts marked incorrect are still judged against the summaries**, because the judge judged them too. Agreement is then reported on all facts and, separately, on the correct facts only.

**2. File sheets: turn the current checklist into "Additional facts"**
- **The existing rows move to a second block** below the extracted facts. These are the Pattern, Role, Responsibility and Relationship rows. The block records pattern information that the extracted list misses, and the student judges it against A–D the same way.
- **This block covers what comment A asked for**: publisher, observers and how notifications work. It matters because 19 of the 30 files have no role fact, including all three Facade files, Singleton.java and ApplicationListener.java.
- **"Unsupported claim" stays in this block.** The judge has no equivalent label, so it is extra information and not part of the agreement.

**3. Instructions sheet**
- **New order of steps:**
  1. Read the code.
  2. Mark each extracted fact correct or incorrect.
  3. Add missing facts.
  4. Judge A–D on both blocks.
- **Keep the existing rule** of completing steps 2–3 before reading the summaries.
- **Replace the worked example** with one built on extracted facts. EventManager's facts would work, since EventManager is not one of the 30 files.
- **Never show the judge's labels, evidence quotes or justifications** to the student.
- **Keep A–D randomised** and the methods hidden.

**4. Source code for other classes**
- **Some facts describe classes other than the file itself.** F21 EventListener.java has 4 role facts, including "EventManager is a publisher…" and "EmailNotificationListener is a concrete observer…".
- **The student cannot check these from that sheet's code alone**, so add GitHub links to the other classes a fact names.

**5. All Annotations and Files sheets**
- **All Annotations rows:** one row per (file, fact_id, summary letter), plus one fact-correctness row per fact, with additional facts flagged separately.
- **New columns:** fact_id, fact_type, source, fact correctness.
- **Linking to the judge:** joined through `manual_annotation_key.csv`, the rows link directly to `fact_preservation_verdicts.csv` on project, design_pattern, file, method and fact_id.
- **Files sheet counters:** replace "Items listed (of 7)" and "Judgements made (of 36)" with per-file counts: number of facts × 4, plus the additional facts.

### 5.5 Information needed

**Already in the repo:**

| Information | Where | What it covers |
|---|---|---|
| Extracted facts for the 30 files | `evaluation-results/fact_preservation_facts.csv` | 141 facts |
| Judge labels to compare against (hidden from the student) | `evaluation-results/fact_preservation_verdicts.csv` | 564 labels |
| Which summary is A–D | `evaluation-results/manual_annotation_key.csv` | All 30 files |
| Label definitions | Judge prompt in `python/judge_fact_preservation.py` | Copy them into the Instructions sheet word for word |
| Source code and summaries | Already in the spreadsheet | Check that all 120 summaries match the judged CSVs; only F01 GPT was checked |

**Decisions to make:**
- **One annotator or two.** One student is enough for student-vs-judge agreement. A second annotator, even on a subset, gives human–human agreement, which shows how high agreement can realistically get. Without it, a low agreement figure is hard to interpret.
- **The statistic.** Use Cohen's κ over the 564 labels, overall, per method and per fact type, plus a confusion matrix. Report raw percentage agreement as well: SWUM has 112 of its 141 labels as Omitted, and an imbalance like that can make κ misleadingly low.
- **What to report from the correctness check.** Report the share of extracted facts the student marks correct, split into detector and parser facts. This also speaks to the concern about implausible detector roles.

### 5.6 Limitation to mention

- **Agreement on pattern roles rests on few labels.** There are 16 role facts in 11 files, which gives 64 labels. Roles are the category the reviewer cares about most.
- **The rest is structural.** The other 125 facts are mostly methods and inheritance, so overall agreement will be dominated by structural facts.
- **Pattern-level facts appear only as additional facts.** This includes publisher and notification behaviour, which are not in the agreement figures.
- **Options:** state this in the response, or add files to the sample from the 65 classes that have role facts.

---

## 6. Open items

| Item | Status |
|---|---|
| Fact preservation on the 20/40/60/80-word corpora | Not run; needs a judge code change and about 1,200 judge calls |
| Manual review of the 239 unmatched quotes | Not done |
| Strict-lower-bound figures (70.6 / 61.1 / 54.4 / 18.4) | Computed by hand; not saved in a repo file |
| Table IV (length-matched comparison) | Still pending |
| Redesign of the annotation spreadsheet (Section 5.4) | Not done |
| Check that all 120 summaries in the spreadsheet match the judged CSVs | Only F01 GPT checked |
| Whether detector role facts were filtered for implausible roles | Not decided or documented |
