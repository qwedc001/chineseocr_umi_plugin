import os
import sys
import site
import base64
from PIL import Image
from io import BytesIO

# 当前目录
CurrentDir = os.path.dirname(os.path.abspath(__file__))
# 依赖包目录
SitePackages = os.path.join(CurrentDir, "site-packages")

"""
TODO 1 :
优化git结构。插件git仓库中，应该只存放必须的代码文件，减小git管理的体积。
如 models 、 site-packages 这些存放二进制文件及第三方库的目录，从git中移除。
将完整的插件打个压缩包放在github的Release里。
即：git只存放插件代码，Release存放完整的插件包。
附：从git记录中永久删除指定目录或文件以释放空间：
https://www.cnblogs.com/shines77/p/3460274.html
https://www.jianshu.com/p/d333ab0e6818
谨慎操作！

TODO 2 :
在本模块的run方法中，img直接转换为cv2对象，跳过转为PIL的步骤。
在 model.py 和 dbnet_infer.py 的相应接口中，直接传递cv2对象。
这样可以去除 PIL → np.asarray(BGR) → RGB 的繁琐转换步骤，减少无谓的开销。

TODO 3 :
将更多可设置参数添加到面板。如 config.py 中的 angle_detect 之类。

TODO 4 :
精度问题。chineseocr主页有一些测试图片，它展示的识别结果是正确的，但本插件的结果存在一些误差。
https://github.com/DayBreak-u/chineseocr_lite/tree/master
需要验证：是否为参数设置不当导致？是否为onnx模型导致？使用别的方法部署（如pytorch）的结果是否一致？

onnx库有一些警告Warning日志信息。这些Warning可以忽视吗？是否为精度问题的原因？
如果可以忽视，则下面47行 onnxruntime.set_default_logger_severity(3) 可以屏蔽日志。
"""


class Api:
    def __init__(self, globalArgd):
        self.chineseocr = None

    # 获取OcrHandle 实例
    def start(self, argd):
        self.short_size = argd["short_size"]  # 记录最新 short_size 参数
        if self.chineseocr:  # 引擎已启动，则跳过再启动
            return ""
        site.addsitedir(SitePackages)  # 依赖库到添加python搜索路径
        try:
            from .model import OcrHandle

            # import onnxruntime
            # 设置onnx日志级别为3级-Error。忽视Warning以下级别的信息。
            # onnxruntime.set_default_logger_severity(3)
            # 启动引擎
            self.chineseocr = OcrHandle()
            return ""
        except Exception as e:
            self.chineseocr = None
            err = str(e)
            print(err)
            return f"[Error] Error on loading:{err}"

    def stop(self):
        self.chineseocr = None

    # 借用自plugins-P2Tocr,进行了修改
    def _standardized(self, res):
        datas = []

        for item in res:
            text = item[1].split(" ")[1]
            accuracy = item[2]
            position = item[0]
            datas.append(
                {
                    "text": text,
                    "score": float(accuracy.__str__()),
                    "box": position.tolist(),
                }
            )
        if datas:
            out = {"code": 100, "data": datas}
        else:
            out = {"code": 101, "data": ""}
        return out

    def _run(self, img: Image):
        if not self.chineseocr:
            res = {"code": 201, "data": "chineseocr not initialized."}
        else:
            try:
                res = self.chineseocr.text_predict(img, self.short_size)
                res = self._standardized(res)
            except Exception as e:
                return {"code": 202, "data": f"chineseocr recognize error:{e}"}
        return res

    def runPath(self, imgPath: str):
        return self._run(Image.open(imgPath))

    def runBytes(self, imgBytes: bytes):
        return self._run(Image.open(BytesIO(imgBytes)))

    def runBase64(self, imgBase64: str):
        return self._run(Image.open(BytesIO(base64.b64decode(imgBase64))))
