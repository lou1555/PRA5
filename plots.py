import glob, csv
import matplotlib.pyplot as plt

def load_latencies(csv_path):
    vals = []
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            vals.append(float(row[3]))  # elapsed_ms
    return vals

datasets, labels = [], []
for path in sorted(glob.glob("*_latency.csv")):
    datasets.append(load_latencies(path))
    labels.append(path.replace("_latency.csv",""))

plt.figure()
plt.boxplot(datasets, labels=labels)
plt.title("PRA5 API Latency (100 calls per case)")
plt.ylabel("Latency (ms)")
plt.xlabel("Test Case")
plt.tight_layout()
plt.savefig("latency_boxplots.png")
print("Saved latency_boxplots.png")
