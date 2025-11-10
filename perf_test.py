import sys, time, csv, statistics, json, random, urllib.request, urllib.error

BASE = "http://serve-sentiment-env.eba-2kt4nxup.us-east-2.elasticbeanstalk.com"
OUT_PREFIX = "pra5"

TEST_CASES = [
    ("fake1", "BREAKING: Government confirms lizard people control the stock market."),
    ("fake2", "Scientists say chocolate cures all diseases in new groundbreaking study."),
    ("real1", "The central bank announced an interest rate decision on Thursday."),
    ("real2", "Local council approves new housing plan after public consultation.")
]

def post_json(url, data, timeout=10):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type":"application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def run_case(name, text, n=100):
    latencies = []
    rows = [("call_idx","start_ts","end_ts","elapsed_ms","http_status")]
    # warmup (not recorded)
    try:
        post_json(f"{BASE}/predict", {"text": text}, timeout=5)
    except Exception:
        pass
    for i in range(n):
        t0 = time.time()
        status = "ERR"
        try:
            _ = post_json(f"{BASE}/predict", {"text": text}, timeout=10)
            status = 200
        except urllib.error.HTTPError as e:
            status = e.code
        except Exception:
            status = "ERR"
        t1 = time.time()
        ms = (t1 - t0) * 1000.0
        rows.append((i, t0, t1, ms, status))
        latencies.append(ms)
        time.sleep(0.05 + random.random()*0.05)
    csv_name = f"{OUT_PREFIX}_{name}_latency.csv"
    with open(csv_name, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    avg = statistics.mean(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]
    p99 = statistics.quantiles(latencies, n=100)[98]
    print(f"{name:>5}: avg={avg:.2f} ms  p95={p95:.2f} ms  p99={p99:.2f} ms  -> {csv_name}")

if __name__ == "__main__":
    print(f"Target base URL: {BASE}")
    for nm, txt in TEST_CASES:
        run_case(nm, txt, n=100)
    print("Done.")
