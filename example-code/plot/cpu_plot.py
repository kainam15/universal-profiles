#cpu_plot
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("result.csv")

df = df[df["status"] == "ok"]
df = df[df["warmup"] == 0]

# 只选 Mem16
df = df[df["mem_cap_gb"] == 16]

plt.figure(figsize=(7,5))

for cpu in sorted(df["cpu_cores"].unique()):
    sub = df[df["cpu_cores"] == cpu]
    g = sub.groupby("context_length", as_index=False)["latency_s"].mean()
    plt.plot(
        g["context_length"],
        g["latency_s"],
        marker="o",
        linewidth=2,
        label=f"CPU{cpu}"
    )

plt.xlabel("Context Length")
plt.ylabel("Latency (s)")
plt.title("Latency vs Context Length (Mem=16GB)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("latency_cpu_scaling.png", dpi=300)
plt.show()