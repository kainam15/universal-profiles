import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("result.csv")

# 只保留成功数据
d = df[df["status"] == "ok"].copy()

# 每个资源组合 cold_start 取平均
g = d.groupby(
    ["gpu_mode", "cpu_cores", "mem_cap_gb"],
    as_index=False
)["cold_start_s"].mean()

# 指定顺序（非常关键）
cpu_order = [1, 2, 4, 8]
mem_order = [2, 4, 8, 16]

g["cpu_cores"] = pd.Categorical(g["cpu_cores"], categories=cpu_order, ordered=True)
g["mem_cap_gb"] = pd.Categorical(g["mem_cap_gb"], categories=mem_order, ordered=True)

g = g.sort_values(["gpu_mode", "cpu_cores", "mem_cap_gb"])

# 分开 on / off
g_on  = g[g["gpu_mode"] == "on"]
g_off = g[g["gpu_mode"] == "off"]

labels_on  = [f"GPU+CPU{int(c)}+Mem{int(m)}" for c, m in zip(g_on["cpu_cores"], g_on["mem_cap_gb"])]
labels_off = [f"CPU{int(c)}+Mem{int(m)}"      for c, m in zip(g_off["cpu_cores"], g_off["mem_cap_gb"])]

values_on  = g_on["cold_start_s"].to_list()
values_off = g_off["cold_start_s"].to_list()

# 合并
labels = labels_on + labels_off
values = values_on + values_off

plt.figure(figsize=(14,6))
plt.bar(labels, values)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Cold Start Time (s)")
plt.title("Cold Start Time (GPU ON vs OFF)")
plt.tight_layout()

plt.savefig("cold_start_on_off.png", dpi=200)
plt.show()