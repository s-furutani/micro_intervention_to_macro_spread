"""共有ユーティリティのファサード（後方互換）。

このモジュールは旧 `util.py` が提供していた API を、責務ごとに分割した
新モジュールから re-export するだけの薄いファサードにゃん。

実体:
- seed 関連:        `seed_users.py` / `paths.py`
- ファイル名構築:    `paths.py`
- violin/box plot:   `plot_violin.py`
- ヒートマップ:      `plot_heatmap.py`
- matplotlib 共通:   `plot_style.py`

旧コードの `import util` / `from util import xxx` をそのまま動かせるよう、
ここで全名前をエクスポートしている。
"""

from __future__ import annotations

# matplotlib rcParams（旧 util.py の冒頭設定相当）
import plot_style  # noqa: F401

# seed_mode 関連（paths.py が一次定義、seed_users 側で再エクスポート）
from paths import (  # noqa: F401
    SEED_MODE_ALIASES,
    SEED_MODES,
    normalize_seed_mode,
    seed_mode_filename_suffix,
)
from seed_users import get_seed_users  # noqa: F401

# violin / boxplot 系
from plot_violin import (  # noqa: F401
    boxplot_jitter_multiple_plots,
    boxplot_jitter_plot,
    violin_plot,
)

# ヒートマップ系
from plot_heatmap import plot_multiple_heatmaps, relative_suppression  # noqa: F401
