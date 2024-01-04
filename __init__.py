from . import api_chineseocr_lite
from . import chineseocr_lite_config

# 插件信息
PluginInfo = {
    # 插件组别
    "group": "ocr",
    # 全局配置
    "global_options": chineseocr_lite_config.globalOptions,
    # 局部配置
    "local_options": chineseocr_lite_config.localOptions,
    # 接口类
    "api_class": api_chineseocr_lite.Api,
}
