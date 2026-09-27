"""rev34 - The two schematic figures of the manuscript, drawn from the model definition.

figR00_screening_concept.pdf   forward-looking screening under a fixed alert budget
figR00_aeris_architecture.pdf  the AERIS network, its three heads and the density layer

Both are vector drawings whose labels follow the notation of Section 3 and the
deployed configuration (window length T, number of inputs, latent dimension m,
attention heads), so that the figures cannot disagree with the text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_FIG_DIR, REV_TABLE_DIR  # noqa: E402

NAVY, TEAL, GREY, LIGHT, ORANGE = "#1F4E79", "#2A7F8E", "#6B6B6B", "#EEF2F6", "#D55E00"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})


def box(ax, x, y, w, h, title, body="", face=LIGHT, edge=NAVY, title_color=NAVY,
        fs_title=7.6, fs_body=6.5, line=0.052):
    """Rounded panel with a (possibly two-line) bold title and a centred body."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.016",
                                facecolor=face, edgecolor=edge, linewidth=1.1))
    top = y + h - 0.03
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=fs_title,
            fontweight="bold", color=title_color, linespacing=1.15)
    if body:
        n_title = title.count("\n") + 1
        ax.text(x + w / 2, top - n_title * line - 0.025, body, ha="center", va="top",
                fontsize=fs_body, color="#222222", linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, color=NAVY, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 color=color, linewidth=lw))


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def concept() -> None:
    fig, ax = canvas(7.2, 2.9)
    w, h, y = 0.225, 0.62, 0.34
    xs = [0.0, 0.258, 0.516, 0.775]
    box(ax, xs[0], y, w, h, "Operating state\nup to $t$",
        "wind and photovoltaic\nmeasurements, forecasts\nand 10-90% bands;\n"
        "imbalance, ACE, prices\n(Elia open data, 15 min)",
        face="#F4F4F4", edge=GREY, title_color="#333333")
    box(ax, xs[1], y, w, h, "AERIS\n ",
        "temporal encoder and\nvariational risk state\nof the last $T$ intervals\n"
        "→ risk score $s_t$ for\nthe interval $t{+}h$")
    box(ax, xs[2], y, w, h, "Ranked alert list\n ",
        "intervals ordered by $s_t$;\nthe top 2% are flagged\n(one to two per day),\n"
        "each with a density\nconfidence $c_t$")
    box(ax, xs[3], y, w, h, "Operator action\n ",
        "15 to 60 min ahead:\nreposition reserve,\nschedule flexible units,\n"
        "activate demand\nresponse", face="#FFF4EC", edge=ORANGE, title_color=ORANGE)
    for i in range(3):
        arrow(ax, xs[i] + w + 0.006, y + h / 2, xs[i + 1] - 0.008, y + h / 2, lw=1.4)
    # decision timeline under the panels
    ty = 0.13
    ax.plot([0.03, 0.97], [ty, ty], color=GREY, lw=1.0)
    ax.add_patch(FancyBboxPatch((0.14, ty - 0.012), 0.23, 0.024, boxstyle="square,pad=0",
                                facecolor="#C9D6E3", edgecolor="none"))
    for x, lab in ((0.14, "$t-T+1$"), (0.37, "$t$ (decision)"), (0.63, "$t{+}h$ (target)")):
        ax.plot([x, x], [ty - 0.025, ty + 0.025], color=GREY, lw=1.0)
        ax.text(x, ty - 0.045, lab, ha="center", va="top", fontsize=6.8, color="#333333")
    ax.text(0.255, ty + 0.035, "observation window", ha="center", va="bottom", fontsize=6.4,
            color=NAVY)
    arrow(ax, 0.375, ty + 0.035, 0.625, ty + 0.035, color=ORANGE, lw=1.0)
    ax.text(0.50, ty + 0.05, "horizon $h$ = 15 to 60 min", ha="center", va="bottom",
            fontsize=6.4, color=ORANGE)
    fig.savefig(REV_FIG_DIR / "figR00_screening_concept.pdf", bbox_inches="tight")
    plt.close(fig)


def architecture(cfg: dict, n_inputs: int) -> None:
    fig, ax = canvas(7.4, 4.1)
    T, m, H = cfg["seq_len"], cfg["latent_dim"], cfg["attention_heads"]
    yb, hb = 0.53, 0.42
    main = [(0.000, 0.125, "Input\nwindow",
             f"$\\mathbf{{X}}_t\\in\\mathbb{{R}}^{{T\\times d}}$\n$T={T}$, $d={n_inputs}$\n"
             "selected\ninputs", "#F4F4F4", GREY, "#333333"),
            (0.145, 0.165, "Temporal\nencoder",
             "linear projection;\n5 residual blocks,\ndilations 1,2,4,1,2,\n"
             "mixing $t-\\delta$, $t$, $t+\\delta$", LIGHT, NAVY, NAVY),
            (0.330, 0.180, "Pooling\nand gate",
             f"attention pooling\n({H} heads, one query)\nand terminal state $\\mathbf{{h}}_T$,\n"
             "fused by a gate", LIGHT, NAVY, NAVY),
            (0.530, 0.150, "Variational\nlayer",
             f"$\\boldsymbol{{\\mu}},\\boldsymbol{{\\sigma}}\\in\\mathbb{{R}}^{{{m}}}$\n"
             "$\\mathbf{z}=\\boldsymbol{\\mu}+\\boldsymbol{\\sigma}\\odot\\boldsymbol{\\epsilon}$\n"
             "(posterior mean\nat inference)", LIGHT, NAVY, NAVY)]
    for x, w, title, body, face, edge, tc in main:
        box(ax, x, yb, w, hb, title, body, face=face, edge=edge, title_color=tc)
    for x0, x1 in ((0.125, 0.145), (0.310, 0.330), (0.510, 0.530)):
        arrow(ax, x0 + 0.004, yb + hb / 2, x1 - 0.005, yb + hb / 2, lw=1.4)
    # three heads on the latent state
    hx, hw, hh = 0.720, 0.280, 0.125
    heads = [(0.825, "Auxiliary head", "$\\hat{\\mathcal{E}}_{t+h}$ → ranking score $s_t$",
              ORANGE, "#FFF4EC"),
             (0.670, "Task head", "logit $a_t$,  $r_t=\\sigma(a_t)$", NAVY, LIGHT),
             (0.515, "Decoder", "window mean $\\hat{\\mathbf{x}}$", TEAL, "#EAF5F6")]
    for yh, title, body, edge, face in heads:
        ax.add_patch(FancyBboxPatch((hx, yh), hw, hh,
                                    boxstyle="round,pad=0.008,rounding_size=0.014",
                                    facecolor=face, edgecolor=edge, linewidth=1.1))
        ax.text(hx + 0.014, yh + hh - 0.022, title, ha="left", va="top", fontsize=7.4,
                fontweight="bold", color=edge)
        ax.text(hx + 0.014, yh + 0.022, body, ha="left", va="bottom", fontsize=6.8,
                color="#222222")
        arrow(ax, 0.684, yb + hb / 2, hx - 0.006, yh + hh / 2, lw=1.1, color=edge)
    # training objective and density layer
    box(ax, 0.145, 0.10, 0.365, 0.31, "Training objective",
        "$\\mathcal{L}=\\mathcal{L}_{\\mathrm{cls}}+\\lambda_{\\mathrm{aux}}\\mathcal{L}_{\\mathrm{aux}}"
        "+\\lambda_{\\mathrm{rec}}\\mathcal{L}_{\\mathrm{rec}}+\\beta\\,\\mathcal{L}_{\\mathrm{KL}}$\n"
        "focal-weighted classification of $y_{t+h}$,\nregression of the future score "
        "$\\mathcal{E}_{t+h}$,\nreconstruction and divergence terms",
        face="#FAFAFA", edge=GREY, title_color="#333333")
    box(ax, 0.530, 0.10, 0.470, 0.31, "Latent-density confidence",
        "training means $\\boldsymbol{\\mu}_i$ → principal projection → KDE\n"
        "$c_t\\in[0,1]$: share of training states less typical than $t$\n"
        "decision index $D_t=s_t\\,[1+\\alpha(1-c_t)]$\n"
        "(projection, bandwidth and $\\alpha$ chosen on validation)",
        face="#EAF5F6", edge=TEAL, title_color=TEAL)
    arrow(ax, 0.605, yb - 0.006, 0.605, 0.41 + 0.008, color=TEAL, lw=1.1)
    ax.text(0.614, 0.47, "$\\boldsymbol{\\mu}$", fontsize=7.5, color=TEAL, va="center")
    fig.savefig(REV_FIG_DIR / "figR00_aeris_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = json.loads((REV_TABLE_DIR / "rev10_selected_summary.json").read_text(encoding="utf-8"))
    cfg = summary["selected_config"]
    n_inputs = int(cfg.get("n_inputs") or len(summary.get("selected_inputs") or []) or 146)
    concept()
    architecture(cfg, n_inputs)
    print("schematics written to", REV_FIG_DIR)


if __name__ == "__main__":
    main()
