"""翻译模块。

导入各翻译器以触发 ``register_translator`` 的注册副作用，
保证运行时（以及 PyInstaller 打包）能拿到完整服务列表。
"""

from services.translation import mock_translator  # noqa: E402,F401  (注册副作用；離線翻譯由 app.application 載入)
from services.translation.base import TranslationError, Translator

__all__ = ["TranslationError", "Translator"]
