"""
make_part3_figures.py  --  reproduces every figure and every statistic quoted in Part 3 of the report.

INPUTS (all produced by this project):
  errors_all.csv            all errors of the final system   <- error_analysis_dump.py dump
  audit_sample.csv          seeded sample of 10 FP + 10 FN   <- error_analysis_dump.py dump (seed 42)
  part3_category_labels.csv hand-assigned categories C1-C4 for the 20 audited errors (the manual audit)
  data/dev_labeled_ChatGPT.jsonl   (optional) used only for the dataset base rates

OUTPUTS:
  part3_error_analysis_figure.png   -> Figure 3.1 (categories)
  fig_confidence_clusters.png       -> Figure 3.2 (confidence + errors per biography)
  stdout                            -> every number quoted in Sections 3.3 / 3.4

Usage:   python make_part3_figures.py [--errors errors_all.csv] [--audit audit_sample.csv] [--labels part3_category_labels.csv]
"""
import argparse, csv, json, os, re, sys
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--errors", default="errors_all.csv")
ap.add_argument("--audit", default="audit_sample.csv")
ap.add_argument("--labels", default="part3_category_labels.csv")
ap.add_argument("--labels_data", default="data/dev_labeled_ChatGPT.jsonl")
a = ap.parse_args()

rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
rows, aud, lab = rd(a.errors), rd(a.audit), rd(a.labels)
fp = [r for r in rows if r["error_type"] == "FP"]
fn = [r for r in rows if r["error_type"] == "FN"]
cat = {r["id"]: r["category"] for r in lab}
missing = [r["id"] for r in aud if r["id"] not in cat]
if missing:
    sys.exit(f"No category for audited ids {missing}; update {a.labels}")

NAMES = ["C1", "C2", "C3", "C4"]
TITLES = {"C1": "Over-inference from partial / related evidence",
          "C2": "Gold label not grounded in the passages",
          "C3": "Paraphrase / implicit-knowledge / aggregation gap",
          "C4": "Evidence lost in candidate selection"}
cnt = {c: Counter() for c in NAMES}
for r in aud:
    cnt[cat[r["id"]]][r["error_type"]] += 1

# ---------------------------------------------------------------- Figure 3.1
labs = ["C1\nOver-inference", "C2\nGold label not\ngrounded", "C3\nParaphrase /\nknowledge gap", "C4\nEvidence lost\n(candidate cap)"]
n_fp = sum(1 for r in aud if r["error_type"] == "FP"); n_fn = sum(1 for r in aud if r["error_type"] == "FN")
N = len(aud)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
x = list(range(4)); fpv = [cnt[c]["FP"] for c in NAMES]; fnv = [cnt[c]["FN"] for c in NAMES]; w = .38
b1 = ax[0].bar([i - w / 2 for i in x], fpv, w, label=f"False positives (n={n_fp})", color="#E4572E", edgecolor="k")
b2 = ax[0].bar([i + w / 2 for i in x], fnv, w, label=f"False negatives (n={n_fn})", color="#17BEBB", edgecolor="k")
for b in list(b1) + list(b2):
    ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + .1, int(b.get_height()), ha="center", fontweight="bold")
ax[0].set_xticks(x); ax[0].set_xticklabels(labs, fontsize=9); ax[0].set_ylabel("Number of audited errors")
ax[0].set_title("(A) Error categories by type"); ax[0].legend(); ax[0].set_ylim(0, max(fpv + fnv) + 1.5)
tot = [p + q for p, q in zip(fpv, fnv)]
ax[1].barh(range(4), [100 * t / N for t in tot], color="#3B5B92", edgecolor="k")
ax[1].set_yticks(range(4)); ax[1].set_yticklabels([l.replace("\n", " ") for l in labs], fontsize=9); ax[1].invert_yaxis()
for i, t in enumerate(tot):
    ax[1].text(100 * t / N + 1, i, f"{100 * t / N:.0f}% (n={t})", va="center", fontweight="bold")
ax[1].set_xlim(0, 55); ax[1].set_xlabel("Share of audited errors (%)"); ax[1].set_title(f"(B) Share of the N = {N} audited errors")
plt.tight_layout(); plt.savefig("part3_error_analysis_figure.png", dpi=200); plt.close()

# ---------------------------------------------------------------- Figure 3.2
bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].hist([float(r["p_ent"]) for r in fp], bins=bins, color="#E4572E", edgecolor="k"); ax[0].axvline(.5, color="k", ls="--")
ax[0].text(.51, ax[0].get_ylim()[1] * .92, "τ = 0.50", fontsize=9)
ax[0].set_title("(A) False positives: P(entailment)"); ax[0].set_xlabel("P(entailment) of best premise"); ax[0].set_ylabel("Number of errors")
ax[1].hist([float(r["p_ent"]) for r in fn], bins=bins, color="#17BEBB", edgecolor="k"); ax[1].axvline(.5, color="k", ls="--")
ax[1].set_title("(B) False negatives: P(entailment)"); ax[1].set_xlabel("P(entailment) of best premise")
allc, fpc, fnc = (Counter(r["topic"] for r in rows), Counter(r["topic"] for r in fp), Counter(r["topic"] for r in fn))
order = [t for t, _ in allc.most_common(8)]; y = list(range(len(order)))
ax[2].barh(y, [fpc[t] for t in order], color="#E4572E", edgecolor="k", label="FP")
ax[2].barh(y, [fnc[t] for t in order], left=[fpc[t] for t in order], color="#17BEBB", edgecolor="k", label="FN")
ax[2].set_yticks(y); ax[2].set_yticklabels([t.replace(" (inventor)", "") for t in order], fontsize=9); ax[2].invert_yaxis()
ax[2].set_xlabel("Number of errors"); ax[2].set_title(f"(C) Errors per biography (top 8 of {len(allc)})"); ax[2].legend()
plt.tight_layout(); plt.savefig("fig_confidence_clusters.png", dpi=200); plt.close()

# ---------------------------------------------------------------- numbers quoted in the report
pe_fp = [float(r["p_ent"]) for r in fp]; pe_fn = [float(r["p_ent"]) for r in fn]
print(f"Errors: {len(rows)} total = {len(fp)} FP + {len(fn)} FN")
print(f"Audit coverage: FP {n_fp}/{len(fp)} = {100*n_fp/len(fp):.0f}%   FN {n_fn}/{len(fn)} = {100*n_fn/len(fn):.0f}%")
print("\nTable 3.3 (audited errors by category):")
for c in NAMES:
    t = cnt[c]["FP"] + cnt[c]["FN"]; print(f"  {c} {TITLES[c]:<52} FP={cnt[c]['FP']} FN={cnt[c]['FN']} total={t} share={100*t/N:.0f}%")
print("\nSection 3.4 statistics (all errors):")
print(f"  FP with P(ent) >= 0.85: {sum(p>=.85 for p in pe_fp)}/{len(fp)};  >= 0.95: {sum(p>=.95 for p in pe_fp)};  between 0.50 and 0.70: {sum(.5<=p<.7 for p in pe_fp)}")
print(f"  FP that a threshold of 0.70 would remove: {sum(p<.7 for p in pe_fp)}/{len(fp)}")
print(f"  FN with P(ent) < 0.05: {sum(p<.05 for p in pe_fn)}/{len(fn)};  P(neutral) > 0.90: {sum(float(r['p_neu'])>.9 for r in fn)};  P(contradiction) > 0.5: {sum(float(r['p_con'])>.5 for r in fn)}")
top2 = fnc.most_common(2); print(f"  FN from the two largest biographies: {sum(v for _,v in top2)}/{len(fn)}  {top2}")
pron = sum("pronoun_hypothesis" in r["auto_tags"] for r in rows); num = sum("numeric_or_date" in r["auto_tags"] for r in rows)
print(f"  Errors that start with a pronoun: {pron}/{len(rows)} = {100*pron/len(rows):.1f}%;  contain digits: {num}/{len(rows)} = {100*num/len(rows):.1f}%")
print(f"  Zero-candidate errors: {sum(r['n_candidates']=='0' for r in rows)};  errors that reached the candidate cap: {Counter(r['n_candidates'] for r in rows).most_common(1)}")
if os.path.exists(a.labels_data):
    n = p = d = 0
    for l in open(a.labels_data):
        for an in json.loads(l).get("annotations") or []:
            for f in an.get("human-atomic-facts") or []:
                n += 1; p += bool(re.match(r"^(He|She|They|His|Her|Their|It|Its)\b", f["text"])); d += bool(re.search(r"\d", f["text"]))
    print(f"\nDataset base rates ({n} facts): pronoun-led {100*p/n:.1f}%, containing digits {100*d/n:.1f}%")
print("\nWrote part3_error_analysis_figure.png and fig_confidence_clusters.png")
