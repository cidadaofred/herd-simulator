import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "chapter7_results" / "assets"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def points(values, x_min, x_max, y_min, y_max, left=70, top=45, width=760, height=360):
    result = []
    for x, y in values:
        px = left + (x - x_min) / (x_max - x_min) * width
        py = top + height - (y - y_min) / (y_max - y_min) * height
        result.append(f"{px:.1f},{py:.1f}")
    return " ".join(result)


def chart(title, series, x_min, x_max, y_min, y_max, x_label, y_label, path):
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">',
        '<rect width="900" height="500" fill="#ffffff"/>',
        f'<text x="450" y="26" text-anchor="middle" font-family="Arial" font-size="19" font-weight="bold">{title}</text>',
    ]
    left, top, width, height = 70, 45, 760, 360
    for index in range(6):
        y = top + height - index * height / 5
        value = y_min + index * (y_max - y_min) / 5
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="60" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')
    for index in range(6):
        x = left + index * width / 5
        value = x_min + index * (x_max - x_min) / 5
        lines.append(f'<text x="{x:.1f}" y="425" text-anchor="middle" font-family="Arial" font-size="11">{value:.0f}</text>')
    lines.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#111827"/>',
        f'<text x="450" y="455" text-anchor="middle" font-family="Arial" font-size="13">{x_label}</text>',
        f'<text x="18" y="230" text-anchor="middle" transform="rotate(-90 18 230)" font-family="Arial" font-size="13">{y_label}</text>',
    ])
    legend_x = 90
    for index, (label, values) in enumerate(series.items()):
        color = colors[index]
        dash = ' stroke-dasharray="8 5"' if index == 2 else ""
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3"{dash} points="{points(values, x_min, x_max, y_min, y_max)}"/>')
        lines.append(f'<line x1="{legend_x}" y1="478" x2="{legend_x + 24}" y2="478" stroke="{color}" stroke-width="3"{dash}/>')
        lines.append(f'<text x="{legend_x + 30}" y="482" font-family="Arial" font-size="12">{label}</text>')
        legend_x += 220
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def day_summaries(campaign):
    directory = DATA / "users" / "web" / "campaigns" / campaign / "day_summaries"
    return [read_json(path) for path in sorted(directory.glob("day_*.json"))]


base = day_summaries("exp_1")
large = day_summaries("branch_continuacao_exp_1")
small = day_summaries("branch_continuacao_exp_2")


def mean_quality(item, field="pasture_after"):
    values = item[field].values()
    return sum(values) / len(item[field])


initial_mean = mean_quality(base[0], "pasture_before")
base_series = [(0, initial_mean)] + [(item["day"], mean_quality(item)) for item in base]
large_series = [(10, mean_quality(base[-1]))] + [
    (item["day"], mean_quality(item)) for item in large
]
small_series = [(10, mean_quality(base[-1]))] + [
    (item["day"], mean_quality(item)) for item in small
]
chart(
    "Qualidade média das pastagens por cenário",
    {
        "Cenário-base (40 animais)": base_series,
        "Expansão para 100 animais": large_series,
        "Redução para 10 animais": small_series,
    },
    0,
    20,
    0,
    0.8,
    "Dia da campanha",
    "Qualidade média (0–1)",
    OUTPUT / "grafico_qualidade_pastagem.svg",
)


analysis_rows = []
for dataset_root in sorted((DATA / "analysis_runs").iterdir()):
    run_root = next(path for path in dataset_root.iterdir() if path.is_dir())
    evaluation = read_json(run_root / "evaluation.json")
    analysis_rows.append(
        {
            "dataset": dataset_root.name,
            "frames": read_json(DATA / "datasets" / dataset_root.name / "observable_manifest.json")["frame_range"]["end"],
            "tp": evaluation["true_positives"],
            "fp": evaluation["false_positives"],
            "fn": evaluation["false_negatives"],
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1": evaluation["f1_score"],
        }
    )

with open(OUTPUT / "metricas_analiticas.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=analysis_rows[0].keys())
    writer.writeheader()
    writer.writerows(analysis_rows)


def bar_chart(rows, path):
    width, height = 900, 500
    left, top, chart_width, chart_height = 70, 50, 760, 330
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="900" height="500" fill="#ffffff"/>',
        '<text x="450" y="28" text-anchor="middle" font-family="Arial" font-size="19" font-weight="bold">Desempenho do agente analítico</text>',
    ]
    for index in range(6):
        y = top + chart_height - index * chart_height / 5
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="60" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{index / 5:.1f}</text>')
    colors = {"precision": "#2563eb", "recall": "#16a34a", "f1": "#f97316"}
    group_width = chart_width / len(rows)
    bar_width = 52
    for group, row in enumerate(rows):
        center = left + group_width * (group + 0.5)
        for offset, metric in enumerate(("precision", "recall", "f1")):
            value = row[metric]
            x = center + (offset - 1) * (bar_width + 6) - bar_width / 2
            y = top + chart_height * (1 - value)
            h = chart_height * value
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" fill="{colors[metric]}"/>')
            lines.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.2f}</text>')
        lines.append(f'<text x="{center:.1f}" y="405" text-anchor="middle" font-family="Arial" font-size="12">{row["dataset"]}</text>')
    legend_x = 260
    for metric, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        lines.append(f'<rect x="{legend_x}" y="448" width="16" height="16" fill="{colors[metric]}"/>')
        lines.append(f'<text x="{legend_x + 22}" y="461" font-family="Arial" font-size="12">{label}</text>')
        legend_x += 140
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


bar_chart(analysis_rows, OUTPUT / "grafico_metricas_analiticas.svg")


summary = {
    "datasets": analysis_rows,
    "pasture": {
        "initial_mean": round(initial_mean, 4),
        "base_day_10_mean": round(mean_quality(base[-1]), 4),
        "large_day_20_mean": round(mean_quality(large[-1]), 4),
        "small_day_13_mean": round(mean_quality(small[-1]), 4),
    },
}
(OUTPUT / "resumo_numerico.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
