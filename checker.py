import csv
from collections import defaultdict

for path in ["exp_linucb_cold.csv", "exp_linucb_warm.csv"]:
    conf_by_model = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if float(row["cpu_ema"]) >= 40:  # match your current stressed threshold
                conf_by_model[row["model_used"]].append(float(row["confidence"]))
    print(path)
    for model, vals in conf_by_model.items():
        print(f"  {model}: n={len(vals)}  avg_confidence={sum(vals)/len(vals):.3f}")