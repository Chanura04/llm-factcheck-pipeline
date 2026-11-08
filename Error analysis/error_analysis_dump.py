"""
error_analysis_dump.py  --  Part 3 helper (place next to factcheck.py / factchecking_main.py)

Step 1 (run once, needs the DeBERTa model):
    python error_analysis_dump.py dump [--cuda] [--strategy threshold|discrete] [--seed 42]
  -> errors_all.csv        every FP/FN with probabilities, best premise and auto-tags
  -> audit_sample.csv      10 FP + 10 FN (seeded random sample) with an EMPTY `category` column to fill by hand
  -> audit_sample.md       the same, readable, for manual reading

Step 2 (after you fill the `category` column in audit_sample.csv):
    python error_analysis_dump.py tally
  -> prints the aggregate table (FP / FN / total / %) ready to paste into the report.

`--dry-run` replaces DeBERTa by a random stub purely to test the plumbing (numbers are meaningless).
"""
import sys, json, csv, re, random, argparse, types
from collections import Counter, defaultdict

PRON = re.compile(r"^(He|She|They|His|Her|Their|It|Its)\b")
NUM = re.compile(r"\d")
NEG = re.compile(r"\b(not|never|no|neither|nor|did not|denied|false(ly)?|claimed)\b", re.I)


def load_examples(labels_path, passages_path):
    from factchecking_main import read_passages, read_fact_examples
    return read_fact_examples(labels_path, read_passages(passages_path))


def scored_predict(checker, fact, passages, strategy):
    """Re-implements EntailmentFactChecker.predict_* but keeps the evidence for analysis."""
    cands = checker._extract_candidates(fact, passages)
    info = dict(n_cand=len(cands), best_premise="", p_ent=0.0, p_neu=0.0, p_con=0.0,
                max_ent_any=0.0, pred="NS")
    if not cands:
        return info
    best = None
    for prem in cands:
        pe, pn, pc = checker.ent_model.check_entailment_raw(prem, fact)
        info["max_ent_any"] = max(info["max_ent_any"], pe)
        if strategy == "discrete":
            if pe > pn and pe > pc:
                info.update(pred="S", best_premise=prem, p_ent=pe, p_neu=pn, p_con=pc)
                return info
            if best is None or pe > best[1]:
                best = (prem, pe, pn, pc)
        else:
            if pe > pc and (best is None or pe > best[1]):
                best = (prem, pe, pn, pc)
            if best and best[1] >= 0.95:
                break
    if best is None:  # every premise was contradiction-dominated: keep the highest-entailment one for display
        best = max(((p,) + checker.ent_model.check_entailment_raw(p, fact) for p in cands), key=lambda x: x[1])
    info.update(best_premise=best[0], p_ent=best[1], p_neu=best[2], p_con=best[3])
    if strategy == "threshold":
        info["pred"] = "S" if (info["p_ent"] >= checker.threshold and info["p_ent"] > info["p_con"]) else "NS"
    return info


class StubModel:
    def __init__(self, seed=0): self.r = random.Random(seed)
    def check_entailment_raw(self, p, h):
        a = [self.r.random() for _ in range(3)]; s = sum(a); return tuple(x / s for x in a)


def dump(args):
    from factcheck import EntailmentFactChecker, EntailmentModel
    exs = load_examples(args.labels_path, args.passages_path)
    if args.dry_run:
        model = StubModel()
    else:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        tok = AutoTokenizer.from_pretrained(name); m = AutoModelForSequenceClassification.from_pretrained(name)
        if args.cuda: m.to("cuda")
        model = EntailmentModel(m, tok, args.cuda)
    checker = EntailmentFactChecker(model, strategy=args.strategy)

    rows, conf = [], Counter()
    for i, ex in enumerate(exs):
        gold = "NS" if ex.label == "IR" else ex.label
        info = scored_predict(checker, ex.fact, ex.passages, args.strategy)
        conf[(gold, info["pred"])] += 1
        if gold != info["pred"]:
            ptxt = " ".join(p["text"] for p in ex.passages)
            tags = []
            if PRON.match(ex.fact): tags.append("pronoun_hypothesis")
            if NUM.search(ex.fact): tags.append("numeric_or_date")
            if info["n_cand"] == 0: tags.append("pruned_to_zero_candidates")
            if NEG.search(ptxt): tags.append("negation_cue_in_passages")
            if ex.label == "IR": tags.append("gold_is_IR")
            rows.append(dict(id=i, topic=ex.passages[0]["title"] if ex.passages else "", fact=ex.fact,
                             gold_raw=ex.label, gold=gold, pred=info["pred"],
                             error_type="FP" if info["pred"] == "S" else "FN",
                             p_ent=round(info["p_ent"], 3), p_neu=round(info["p_neu"], 3), p_con=round(info["p_con"], 3),
                             n_candidates=info["n_cand"], auto_tags=";".join(tags),
                             best_premise=info["best_premise"], category=""))
    n = len(exs); correct = conf[("S", "S")] + conf[("NS", "NS")]
    print(f"Confusion (gold,pred): {dict(conf)}")
    print(f"Accuracy {correct}/{n} = {correct/n:.4f}  |  FP={conf[('NS','S')]}  FN={conf[('S','NS')]}")

    fields = list(rows[0].keys()) if rows else []
    with open("errors_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    rng = random.Random(args.seed)
    fp = [r for r in rows if r["error_type"] == "FP"]; fn = [r for r in rows if r["error_type"] == "FN"]
    sample = rng.sample(fp, min(10, len(fp))) + rng.sample(fn, min(10, len(fn)))
    with open("audit_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(sample)
    with open("audit_sample.md", "w", encoding="utf-8") as f:
        for k, r in enumerate(sample, 1):
            f.write(f"### {k}. [{r['error_type']}] {r['topic']}\n- **Fact:** {r['fact']}\n- **Gold:** {r['gold_raw']} -> {r['gold']} | **Pred:** {r['pred']}\n"
                    f"- **P(ent/neu/con):** {r['p_ent']} / {r['p_neu']} / {r['p_con']}  | candidates: {r['n_candidates']}\n"
                    f"- **Auto-tags:** {r['auto_tags']}\n- **Best premise:** {r['best_premise']}\n- **Your category:** ________\n\n")
    print(f"Wrote errors_all.csv ({len(rows)} errors), audit_sample.csv/.md ({len(sample)} rows)")


def tally(_):
    rows = list(csv.DictReader(open("audit_sample.csv", encoding="utf-8")))
    cats = defaultdict(lambda: Counter())
    for r in rows:
        if not r["category"].strip(): sys.exit("Fill every `category` cell first.")
        cats[r["category"].strip()][r["error_type"]] += 1
    N = len(rows)
    print(f"| Error category | FP | FN | Total (N={N}) | % |\n|---|---|---|---|---|")
    for c, k in sorted(cats.items(), key=lambda x: -sum(x[1].values())):
        t = sum(k.values()); print(f"| {c} | {k['FP']} | {k['FN']} | {t} | {100*t/N:.1f}% |")
    print(f"| **Total** | {sum(1 for r in rows if r['error_type']=='FP')} | {sum(1 for r in rows if r['error_type']=='FN')} | {N} | 100% |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "tally"])
    ap.add_argument("--labels_path", default="data/dev_labeled_ChatGPT.jsonl")
    ap.add_argument("--passages_path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    ap.add_argument("--strategy", default="threshold", choices=["threshold", "discrete"])
    ap.add_argument("--cuda", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dump(a) if a.cmd == "dump" else tally(a)
