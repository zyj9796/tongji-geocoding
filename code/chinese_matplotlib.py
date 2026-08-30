"""Matplotlib 可见文字中文化适配器。

用于重跑历史绘图代码时统一翻译标题、坐标轴、图例和标注，
不改动数据、图形对象、坐标范围或统计数值。
"""

from __future__ import annotations

from functools import wraps

import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

from organize_picall_chinese import translated_text


def _translate(value):
    if isinstance(value, str):
        return translated_text(value)
    if isinstance(value, tuple):
        return tuple(_translate(item) for item in value)
    if isinstance(value, list):
        return [_translate(item) for item in value]
    return value


def _patch_positional(cls, method_name: str, index: int) -> None:
    original = getattr(cls, method_name)
    if getattr(original, "_picall_chinese", False):
        return

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        values = list(args)
        if len(values) > index:
            values[index] = _translate(values[index])
        return original(self, *values, **kwargs)

    wrapped._picall_chinese = True
    setattr(cls, method_name, wrapped)


def _patch_label_keyword(method_name: str) -> None:
    original = getattr(Axes, method_name)
    if getattr(original, "_picall_chinese", False):
        return

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        if "label" in kwargs:
            kwargs["label"] = _translate(kwargs["label"])
        return original(self, *args, **kwargs)

    wrapped._picall_chinese = True
    setattr(Axes, method_name, wrapped)


def install_chinese_labels() -> None:
    """安装一次全局适配，使后续新建图件的可见文字使用中文。"""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Source Han Sans CN", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    # 历史模块会在运行期重置 rcParams，因此在最终 Text 对象层强制使用中文字库。
    original_set_text = Text.set_text
    if not getattr(original_set_text, "_picall_chinese", False):
        chinese_font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

        @wraps(original_set_text)
        def set_text_chinese(self, value):
            result = original_set_text(self, _translate(value))
            self.set_fontproperties(chinese_font)
            return result

        set_text_chinese._picall_chinese = True
        Text.set_text = set_text_chinese
    for cls, name, index in [
        (Axes, "set_title", 0), (Axes, "set_xlabel", 0), (Axes, "set_ylabel", 0),
        (Axes, "text", 2), (Axes, "annotate", 0),
        (Axes, "set_xticklabels", 0), (Axes, "set_yticklabels", 0),
        (Figure, "suptitle", 0), (Figure, "text", 2),
        (Colorbar, "set_label", 0), (Artist, "set_label", 0),
    ]:
        _patch_positional(cls, name, index)
    for name in ["plot", "scatter", "bar", "barh", "step", "fill", "fill_between", "axhline", "axvline"]:
        _patch_label_keyword(name)
