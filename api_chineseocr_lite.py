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


class Api:
    def __init__(self, globalArgd):
        self.chineseocr = None

    # 获取OcrHandle 实例
    def start(self, argd):
        # 引擎已启动，且 short_size 一致，则跳过再启动
        if self.chineseocr and self.short_size == argd["short_size"]:
            return ""
        site.addsitedir(SitePackages)  # 依赖库到添加python搜索路径
        try:
            from .model import OcrHandle

            self.chineseocr = OcrHandle()
            self.short_size = argd["short_size"]
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
