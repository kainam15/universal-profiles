"""资源配置的颜色和绘图样式。"""

import colorsys

import pandas as pd
import matplotlib.pyplot as plt

from acprof.plotting.config import (
    CPU_FIXED_COLORS,
    GPU_GREEN,
    MEM_FIXED_COLORS,
)


def _sort_key(config: tuple[int, int, bool]) -> tuple[int, int, int]:
    cpu, mem, gpu_on = config
    return 0 if gpu_on else 1, cpu, mem


def build_cpu_base_colors(cpu_values: list[int]) -> dict[int, tuple[float, float, float]]:
    color_map = plt.get_cmap("tab10")
    cpu_colors = {}
    for index, cpu in enumerate(cpu_values):
        cpu_colors[cpu] = CPU_FIXED_COLORS.get(cpu, color_map(index % color_map.N)[:3])
    return cpu_colors


def shade_for_mem(
    base_rgb: tuple[float, float, float],
    mem_rank: int,
    mem_count: int,
) -> tuple[float, float, float]:
    red, green, blue = base_rgb
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)

    if mem_count <= 1:
        target_lightness = lightness
    else:
        min_lightness = 0.30
        max_lightness = 0.78
        ratio = mem_rank / (mem_count - 1)
        target_lightness = max_lightness - ratio * (max_lightness - min_lightness)

    return colorsys.hls_to_rgb(hue, target_lightness, saturation)


def shade_for_cpu(
    base_rgb: tuple[float, float, float],
    cpu_rank: int,
    cpu_count: int,
) -> tuple[float, float, float]:
    """Keep the base hue while making larger CPU configurations darker."""
    if cpu_count <= 1:
        return base_rgb
    return shade_for_mem(base_rgb, cpu_rank, cpu_count)


def build_gpu_mixed_colors(
    configs: list[tuple[int, int, bool]],
) -> dict[tuple[int, int, bool], tuple[float, float, float]]:
    gpu_configs = [config for config in configs if config[2]]
    gpu_cpus = sorted({cpu for cpu, _, gpu_on in gpu_configs if gpu_on})
    if not gpu_configs:
        return {}

    cpu_rank_map = {cpu: index for index, cpu in enumerate(gpu_cpus)}
    return {
        config: shade_for_mem(GPU_GREEN, cpu_rank_map[config[0]], len(gpu_cpus))
        for config in gpu_configs
    }


def color_for_mem(mem: int, mem_rank: int) -> tuple[float, float, float]:
    if mem in MEM_FIXED_COLORS:
        return MEM_FIXED_COLORS[mem]

    color_map = plt.get_cmap("tab10")
    return color_map(mem_rank % color_map.N)[:3]


def _configurations_with_colors(
    agg_df: pd.DataFrame,
) -> tuple[
    list[tuple[int, int, bool]],
    dict[tuple[int, int, bool], tuple[float, float, float]],
]:
    """Return sorted configurations and the standard color for each one."""
    cpu_values = sorted(int(value) for value in agg_df["cpu_cores"].unique())
    mem_values = sorted(int(value) for value in agg_df["mem_cap_gb"].unique())
    cpu_colors = build_cpu_base_colors(cpu_values)
    mem_rank_map = {mem: index for index, mem in enumerate(mem_values)}
    configs = sorted(
        {
            (int(row.cpu_cores), int(row.mem_cap_gb), bool(row.gpu_on))
            for row in agg_df.itertuples(index=False)
        },
        key=_sort_key,
    )
    has_cpu_series = any(not gpu_on for _, _, gpu_on in configs)
    gpu_mixed_colors = build_gpu_mixed_colors(configs) if has_cpu_series else {}
    colors = {}
    for cpu, mem, gpu_on in configs:
        if gpu_on and has_cpu_series:
            colors[(cpu, mem, gpu_on)] = gpu_mixed_colors[(cpu, mem, gpu_on)]
        else:
            colors[(cpu, mem, gpu_on)] = shade_for_mem(
                cpu_colors[cpu],
                mem_rank_map[mem],
                len(mem_values),
            )
    return configs, colors
