from chatglm.ui import on_ui_tabs
from chatglm.util import on_app_started

from modules import script_callbacks

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_app_started(on_app_started)