"""matplotlib のグローバル設定を集約するモジュール。

旧 `util.py` の冒頭で行われていた rcParams 設定をここにまとめる。
import するだけで rcParams が更新されるため、可視化関数を含むモジュール側で
`import plot_style  # noqa: F401` の形で呼ぶ。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "Arial"
