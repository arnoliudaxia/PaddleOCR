# PP-OCRv6_small Android 原生迁移开发文档

> 目标：将当前基于 Python/PaddleOCR 的手机网页 OCR，迁移为 Android 14+ 原生应用，并优先使用 Snapdragon 8 Gen 3 的硬件能力。
>
> 本文针对当前项目的实际功能：后置摄像头、相册图片、实时预览、OCR 叠加显示、数字/符号过滤、四位数字目标搜索、匹配后冻结画面。

---

## 1. 目标与非目标

### 1.1 目标

Android 应用需要实现：

1. 使用后置摄像头采集实时画面；
2. 保留高分辨率原始画面用于显示；
3. 使用 PP-OCRv6_small 检测和识别文字；
4. 在画面中绘制检测框和识别结果；
5. 只对数字和指定符号进行文字展示；
6. 非数字文字保留灰色检测框；
7. 支持从相册选择图片；
8. 支持输入 4 位数字作为目标；
9. 识别到目标后使用红色半透明填充框选；
10. 目标命中后冻结主画面并停止 OCR；
11. 再次设置目标后恢复识别；
12. 优先测试 CPU、NNAPI、Qualcomm GPU 和 Qualcomm HTP/NPU。

### 1.2 非目标

第一阶段不要求：

- 将完整 PaddlePaddle Python 运行时打包进 APK；
- 在手机上运行当前 HTTP API；
- 让每一帧都达到相机原始 30 fps 的 OCR 频率；
- 让所有图像预处理、后处理都运行在 NPU 上；
- 同时支持所有 Android 厂商和所有 Snapdragon 型号。

推荐的移动端结构是直接在本地运行 ONNX 模型，而不是继续调用局域网服务器。

---

## 2. 推荐总体架构

```text
┌──────────────────────────────────────────────┐
│ Android App                                   │
│                                              │
│  CameraX Preview ───────────────┐            │
│       │                          │            │
│       │ YUV/RGBA frame           │            │
│       ▼                          ▼            │
│  Capture/Preprocess          Display Canvas   │
│       │                          ▲            │
│       ▼                          │            │
│  PP-OCRv6_small_det.onnx        │            │
│       │                          │            │
│       ▼                          │            │
│  DB postprocess                  │            │
│       │                          │            │
│  crop/rotate text regions        │            │
│       ▼                          │            │
│  PP-OCRv6_small_rec.onnx        │            │
│       │                          │            │
│       ▼                          │            │
│  CTC decode + filter + target ───┘            │
└──────────────────────────────────────────────┘
```

模型分为两个独立 ONNX session：

```text
det session: PP-OCRv6_small_det.onnx
rec session: PP-OCRv6_small_rec.onnx
```

如果启用文字行方向分类，还需要单独的方向模型；第一版可以保持当前网页逻辑，只启用检测和识别，确认基础流程后再加入方向模型。

---

## 3. 模型和运行时选择

## 3.1 推荐模型

当前已验证最适合项目的是：

```text
PP-OCRv6_small_det
PP-OCRv6_small_rec
```

Android 端应准备：

```text
models/
├── det/
│   └── inference.onnx
├── rec/
│   ├── inference.onnx
│   ├── inference.yml
│   └── rec_char_dict.txt
└── config.json
```

识别字典必须与识别模型匹配。不能只复制 `inference.onnx` 而忽略字典，否则模型输出的类别索引无法还原成文字。

## 3.2 第一阶段运行时

建议依次验证：

| 阶段 | 运行时 | 目的 |
|---|---|---|
| 1 | ONNX Runtime CPU | 先确认模型和后处理正确 |
| 2 | ONNX Runtime NNAPI | 尝试系统提供的 GPU/NPU 加速 |
| 3 | ONNX Runtime QNN GPU | 使用 Qualcomm GPU |
| 4 | ONNX Runtime QNN HTP | 尝试 Hexagon NPU |

第一版不要一开始就做 QNN HTP。QNN 的模型兼容性、库文件、设备驱动和 HTP 架构匹配问题，会显著增加排错成本。

## 3.3 Snapdragon 8 Gen 3 的现实情况

Snapdragon 8 Gen 3 通常包含 Adreno GPU 和 Hexagon AI 加速单元，但应用能否使用某个后端取决于：

- 手机厂商 ROM；
- Android 系统版本；
- Qualcomm 驱动；
- QAIRT/QNN 版本；
- ONNX 算子是否被目标后端支持；
- 模型是否进行了适合后端的 FP16/INT8 转换。

因此不能仅凭芯片型号断言模型一定运行在 NPU 上。必须通过运行时日志、ORT profiling 或 QNN 日志确认。

---

## 4. Android 工程初始化

## 4.1 推荐工程参数

建议使用：

```text
Android Studio Ladybug 或更新版本
JDK 17
Kotlin 2.x
compileSdk 35 或当前稳定版本
minSdk 26 或更高
ABI: arm64-v8a
```

Snapdragon 8 Gen 3 只需要优先支持：

```text
arm64-v8a
```

不要在第一版同时打包 `armeabi-v7a`，否则会增加 APK 体积和测试矩阵。

## 4.2 基础依赖

示例依赖：

```kotlin
dependencies {
    implementation("androidx.camera:camera-camera2:<version>")
    implementation("androidx.camera:camera-lifecycle:<version>")
    implementation("androidx.camera:camera-view:<version>")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:<version>")

    implementation("androidx.lifecycle:lifecycle-runtime-ktx:<version>")
    implementation("androidx.activity:activity-compose:<version>")
}
```

PP-OCRv6 官方 Android 示例采用 ONNX Runtime 和 OpenCV，并提供 SDK / AAR 结构。实际项目可以从官方示例开始，再替换为当前项目的 UI 和识别规则。

官方 Android 部署文档：

<https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/inference_deployment/cross_platform/android_deployment.html>

## 4.3 权限

`AndroidManifest.xml`：

```xml
<uses-permission android:name="android.permission.CAMERA" />
```

如果只使用本地模型，不需要网络权限。若保留模型下载或日志上传功能，才添加：

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

相册访问建议使用 Android Photo Picker，不要在 Android 14+ 默认申请广泛存储权限。

---

## 5. 模型准备和 ONNX 导出

## 5.1 优先使用官方 Android 模型

优先从 PaddleOCR 官方 Android 部署示例获取 PP-OCRv6 ONNX 模型。如果官方示例中的模型名称、输入配置与当前 small 模型一致，直接使用并进行结果对照。

检查模型输入输出：

```bash
python - <<'PY'
import onnx
m = onnx.load("inference.onnx")
for x in m.graph.input:
    print("input", x.name)
for x in m.graph.output:
    print("output", x.name)
PY
```

也可以使用 ONNX Runtime 的模型检查工具或 Netron 查看输入布局。

## 5.2 从 Paddle 模型导出

如果官方没有提供所需的 small 模型，可以使用 Paddle2ONNX/PaddleX 导出。具体命令需要以当前安装版本的帮助信息为准：

```bash
paddlex --help
paddlex --paddle2onnx --help
```

典型形式：

```bash
paddlex --paddle2onnx \
  --paddle_model_dir /path/to/PP-OCRv6_small_det \
  --onnx_model_dir /path/to/PP-OCRv6_small_det_onnx

paddlex --paddle2onnx \
  --paddle_model_dir /path/to/PP-OCRv6_small_rec \
  --onnx_model_dir /path/to/PP-OCRv6_small_rec_onnx
```

导出后必须在 Python 端用 ONNX Runtime 跑一张相同图片，与 PaddlePaddle 输出对比：

- 检测框数量；
- 检测框位置；
- 识别文字；
- 识别置信度；
- 旋转文字结果。

不要在没有对照的情况下直接把导出的模型放进 APK。

## 5.3 模型精度策略

第一阶段使用 FP32 或 FP16 ONNX：

```text
优先保证结果一致性
```

第二阶段再测试：

```text
FP16
INT8
```

INT8 需要校准数据，建议使用当前仓库中实际拍摄的快递标签图片，而不是只使用通用图片。校准数据至少应覆盖：

- 远近不同的标签；
- 旋转标签；
- 反光和阴影；
- 模糊画面；
- 不同数字字体；
- 目标数字和相似数字。

---

## 6. OCR 预处理和后处理

## 6.1 检测输入

PP-OCR 的检测模型通常接收 `NCHW` 浮点张量。具体尺寸和归一化方式必须以模型配置为准，不应猜测。

典型流程：

```text
Android Bitmap / ImageProxy
    ↓
旋转到显示坐标
    ↓
缩放到检测模型输入范围
    ↓
RGB/BGR 转换
    ↓
归一化
    ↓
HWC → CHW
    ↓
FloatBuffer
```

建议保留两个尺寸：

```text
原始图：用于主画面显示和最终框绘制
缩放图：用于 OCR 推理
```

例如相机实际为 `3200×2400` 时，检测副本可以限制长边在 `1280～1920`。检测框得到后再按缩放比例映射回原图。

## 6.2 DB 检测后处理

检测模型输出通常不是最终四边形，需要实现：

1. 概率图阈值处理；
2. 可选的 box score 过滤；
3. 连通域/轮廓提取；
4. unclip 扩张；
5. 最小边长过滤；
6. 四边形排序；
7. 坐标映射回原图。

当前 Python 脚本中的关键阈值由 PaddleOCR/PaddleX 默认配置提供。Android 迁移时应先保持相同默认值，再单独调节：

```text
thresh
box_thresh
unclip_ratio
max_candidates
```

提高阈值不一定让检测网络更快，只可能减少后续识别框数量；检测网络本身仍然要运行。

## 6.3 文字裁剪

每个检测四边形需要进行透视变换，将文字区域拉正后送入识别模型：

```text
四边形 polygon
    ↓
计算目标宽高
    ↓
透视变换
    ↓
按识别模型高度缩放
    ↓
按比例 padding 或 batch
```

识别模型通常可以对多个文字框组成 batch，但实时模式第一版建议使用 batch=1 或小 batch，先保证内存和延迟稳定。

## 6.4 CTC 解码

识别模型输出类别序列后，需要：

1. 对每个时间步取最大类别；
2. 去除重复 token；
3. 去除 blank；
4. 使用 `rec_char_dict.txt` 转换字符；
5. 计算平均置信度。

Android 端的识别结果必须与 Python 端逐字对照。尤其需要检查：

- 全角数字；
- 半角/全角横线；
- 中文冒号；
- 斜杠；
- 识别字典末尾是否包含 blank token。

---

## 7. 业务过滤规则

当前规则可以抽象为 Kotlin：

```kotlin
private val numericPattern = Regex(
    "[0-9０-９+＋\\-－−./／:：%％#＃×*＝=()（）\\[\\]【】,，]+"
)

fun isNumericOrSymbol(text: String): Boolean {
    return text.isNotEmpty() && numericPattern.matches(text.trim())
}
```

四位数字：

```kotlin
fun isFourDigits(text: String): Boolean {
    return Regex("[0-9]{4}").matches(text)
}
```

数字-数字前缀：

```kotlin
fun startsWithDigitHyphenDigit(text: String): Boolean {
    return Regex("^[0-9]-[0-9]").containsMatchIn(text)
}
```

目标匹配：

```kotlin
fun isTarget(text: String, target: String): Boolean {
    return target.matches(Regex("[0-9]{4}")) && text.trim() == target
}
```

显示规则：

| 识别结果 | 框 | 填充 | 文字 |
|---|---|---|---|
| 目标四位数字 | 红色粗框 | 红色半透明 | 白色 |
| 普通四位数字 | 蓝色 | 蓝色半透明 | 白色 |
| 数字-数字 | 绿色 | 绿色半透明 | 白色 |
| 其他数字/符号 | 红色 | 无或透明 | 白色 |
| 非数字文字 | 灰色 | 无 | 不显示 |

目标命中后：

```text
保留命中帧
停止提交新 OCR 请求
停止/暂停主画面更新
暂停视频预览，或保留独立实时小窗
```

推荐使用状态机，而不是多个 Boolean 分散控制：

```kotlin
enum class OcrState {
    STOPPED,
    RUNNING,
    REQUESTING,
    TARGET_FOUND,
    ERROR
}
```

---

## 8. CameraX 设计

## 8.1 两个画面

当前网页的交互可以原样映射到 Android：

```text
PreviewView：实时相机小窗
Canvas/Overlay View：最近一次 OCR 结果主画面
```

主画面不能直接每帧刷新，否则 OCR 框会被下一帧相机画面擦掉。正确结构是：

```text
CameraX Preview ──实时小窗
ImageAnalysis ────取帧并送 OCR
Overlay View ─────只在 OCR 返回后刷新
```

## 8.2 图像分析策略

不建议对每个 30 fps 帧都执行 OCR。推荐：

```text
相机预览：30 fps
OCR：上一帧完成后才提交下一帧
```

伪代码：

```kotlin
if (ocrBusy || state != OcrState.RUNNING) {
    imageProxy.close()
    return
}

ocrBusy = true
val bitmap = imageProxy.toBitmap()
imageProxy.close()

ocrExecutor.execute {
    try {
        val result = ocr.run(bitmap)
        mainExecutor.execute {
            overlay.update(result)
            if (result.targetFound) {
                state = OcrState.TARGET_FOUND
            }
        }
    } finally {
        ocrBusy = false
    }
}
```

必须保证每个 `ImageProxy` 最终调用 `close()`，否则相机会因为分析缓冲区耗尽而卡住。

## 8.3 分辨率

相机的最大分辨率不一定是最适合 OCR 的分辨率。建议保存：

```text
Preview：设备支持的高分辨率
Analysis：适合 OCR 的分辨率
```

可以通过 CameraX 查询支持的输出尺寸，并记录实际选中的：

```text
width
height
rotationDegrees
format
frame rate
```

第一阶段建议：

```text
Preview：设备高分辨率
Analysis：1280～1920 长边
```

之后再测试 3200×2400 是否能提高目标数字识别率。

---

## 9. ONNX Runtime CPU 集成

## 9.1 Session 初始化

两个模型各创建一个 session，并在应用启动时加载一次：

```kotlin
val env = OrtEnvironment.getEnvironment()
val detSession = env.createSession(detModelBytes, sessionOptions)
val recSession = env.createSession(recModelBytes, sessionOptions)
```

不要每帧重新创建 session。

## 9.2 线程策略

第一版建议：

```text
OCR executor：单线程
CPU intra-op：2～4
CPU inter-op：1
```

手机上线程越多不一定越快，可能导致：

- UI 卡顿；
- 发热；
- 功耗增加；
- 相机帧队列积压。

## 9.3 内存策略

避免每帧重复分配大量对象：

- 复用输入 FloatBuffer；
- 复用 Bitmap 或临时数组；
- 限制同时处理的文字区域数量；
- 每次 OCR 返回后释放临时裁剪图；
- 不要保存所有历史帧。

---

## 10. NNAPI 加速

NNAPI 是 Android 系统层的硬件抽象接口。它可能把部分算子交给 Qualcomm 驱动，但实际后端由设备系统决定。

测试流程：

```text
同一模型
同一输入
同一手机
分别测试 CPU 和 NNAPI
```

记录：

```text
session 创建时间
首帧时间
稳定帧平均时间
P50/P90 延迟
结果差异
温度和耗电
```

如果 NNAPI 速度没有提高，可能原因是：

- 图没有成功分区到加速器；
- 算子不支持；
- 模型输入尺寸动态导致编译开销；
- 后处理耗时占主要比例；
- 驱动将模型回退到 CPU。

因此 NNAPI 只能作为“尝试路径”，不能默认等于 NPU。

---

## 11. Qualcomm QNN GPU / HTP 部署

## 11.1 QNN 组成

QNN 路径大致是：

```text
ONNX Runtime
    ↓ QNN Execution Provider
Qualcomm QNN
    ↓
GPU 或 HTP backend
    ↓
Adreno GPU / Hexagon HTP
```

需要的文件和组件取决于 QAIRT 版本，通常包括：

```text
ONNX Runtime Android arm64
QNN Execution Provider
QNN backend library
Qualcomm runtime libraries
HTP stub/skel libraries（HTP）
```

不要从不同 QAIRT 版本混拷 `.so` 文件。

## 11.2 QNN GPU

QNN GPU 通常比 HTP 更容易先验证，但仍需要：

- 对应的 QNN GPU backend；
- OpenCL/Vulkan 相关运行时；
- 正确的 ABI；
- 模型算子支持。

先将 detection 和 recognition 分别测试，不要一上来测试完整流水线。

## 11.3 QNN HTP/NPU

HTP 部署需要特别关注：

1. SoC model 和 HTP arch；
2. QAIRT 与手机驱动兼容性；
3. skel library 是否能被 DSP 侧找到；
4. FP16/INT8 支持情况；
5. 模型是否含不支持算子；
6. 是否发生 CPU fallback；
7. 首次 context 编译时间。

在 Android 上，QNN HTP 失败时常见日志包括：

```text
failed to load HTP backend
remote_handle64_open failed
skel library not found
contextCreate failed
unsupported op
```

这类问题通常不是 Kotlin 代码错误，而是 QNN SDK、驱动、库路径或模型后端兼容性问题。

## 11.4 QNN 的实际建议

建议采用以下顺序：

```text
CPU 基线
→ NNAPI
→ QNN GPU
→ QNN HTP
```

只有当 QNN HTP 的端到端耗时明显低于 CPU/NNAPI，并且结果稳定，才值得将它作为正式发布后端。

---

## 12. 性能测试规范

不能只测一次首帧。每个后端至少测试：

```text
模型加载时间
首次推理时间
预热后 20～50 帧
P50
P90
最大延迟
内存
温度
电池消耗
```

测试输入必须固定：

- 同一张图片；
- 同一组视频帧；
- 同一分辨率；
- 同一目标区域；
- 同一阈值；
- 同一 OCR 后处理逻辑。

推荐记录：

```text
det_ms
crop_ms
rec_ms
postprocess_ms
draw_ms
total_ms
boxes_count
numeric_count
target_found
```

结果表：

| 后端 | det P50 | rec P50 | 总 P50 | 首帧 | 结果一致性 | 温度 |
|---|---:|---:|---:|---:|---|---:|
| CPU |  |  |  |  |  |  |
| NNAPI |  |  |  |  |  |  |
| QNN GPU |  |  |  |  |  |  |
| QNN HTP |  |  |  |  |  |  |

如果总耗时没有下降，不要只优化模型推理，要检查：

```text
YUV/RGB 转换
Bitmap 拷贝
透视变换
Java/Kotlin ↔ JNI 拷贝
DB 后处理
Canvas 绘制
```

---

## 13. 与当前 Python 版本的验收对照

准备一组固定图片，建议包括：

```text
images/image.png
视频每秒抽样帧
不同旋转角度的标签
模糊和反光样本
包含中文和数字的混合图片
```

每张图片生成 Python 基准 JSON：

```json
{
  "image": "frame_000000.png",
  "results": [
    {
      "text": "6028",
      "score": 0.99,
      "polygon": [[...], [...], [...], [...]]
    }
  ]
}
```

Android 输出同样的结构，然后比较：

- 文字完全一致率；
- 目标数字召回率；
- 误匹配率；
- 检测框 IoU；
- 坐标旋转方向；
- 低置信度结果差异。

验收建议：

```text
目标数字召回率优先于平均速度
```

因为本项目最终功能是“找到目标后冻结”，漏掉目标比延迟增加几十毫秒更严重。

---

## 14. 推荐开发阶段

### 阶段 A：Android UI 和相机

完成：

- CameraX 后置相机；
- PreviewView 实时预览；
- 可拖动小窗；
- 高分辨率显示；
- 相册图片选择；
- 目标输入框。

此阶段先用假 OCR 框验证 UI。

### 阶段 B：ONNX CPU OCR

完成：

- det session；
- DB 后处理；
- rec session；
- CTC 解码；
- 坐标映射；
- 数字/符号过滤；
- 红/蓝/绿/灰标注。

要求与 Python 版本对照通过。

### 阶段 C：实时闭环

完成：

- 单线程 OCR 队列；
- 上一帧完成后提交下一帧；
- 主画面只在结果返回后更新；
- 目标命中后冻结；
- 重新设置目标后恢复。

### 阶段 D：硬件加速

依次测试：

```text
CPU → NNAPI → QNN GPU → QNN HTP
```

每一步都保留 CPU fallback，不能因为硬件后端不可用而让应用无法启动。

### 阶段 E：发布优化

完成：

- 模型压缩；
- FP16/INT8 对照；
- APK 只保留 arm64-v8a；
- 16 KB page size 兼容性检查；
- 崩溃和后端初始化日志；
- 温度和功耗测试；
- 离线运行测试。

---

## 15. 推荐的回退策略

正式版本建议有明确的后端回退链：

```text
QNN HTP
  ↓ 失败
QNN GPU
  ↓ 失败
NNAPI
  ↓ 失败
ONNX Runtime CPU
```

伪代码：

```kotlin
fun createOcrEngine(): OcrEngine {
    return tryCreateQnnHtp()
        ?: tryCreateQnnGpu()
        ?: tryCreateNnapi()
        ?: createCpuEngine()
}
```

每个后端启动时记录：

```text
selected_backend
fallback_reason
model_version
runtime_version
device_model
android_version
```

不能静默地声称“使用 NPU”。只有后端确认成功且 profiling 证明确实有算子下发时，才显示硬件加速状态。

---

## 16. 当前项目迁移时应保留的逻辑

Python 版本中的以下逻辑应在 Android 端保持一致：

```text
device-independent:
- NUMERIC_TEXT 过滤规则
- 四位数字判断
- 数字-数字判断
- 目标数字严格匹配
- 原图坐标标注
- 红色/蓝色/绿色半透明填充
- 非数字文字灰色框
- 目标命中后冻结
```

以下内容不应直接照搬到移动端：

```text
- PaddleOCR Python Pipeline
- PIL 绘制
- HTTP 上传到本机服务
- 每帧创建 PaddleOCR 对象
- 每帧读取模型文件
```

---

## 17. 参考资料

1. PaddleOCR Android PP-OCRv6 部署：
   <https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/inference_deployment/cross_platform/android_deployment.html>

2. PaddleOCR PP-OCRv6：
   <https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html>

3. ONNX Runtime QNN Execution Provider：
   <https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html>

4. Qualcomm AI Runtime / QAIRT：
   <https://docs.qualcomm.com/doc/80-80022-15B/topic/qairt.html>

5. Paddle2ONNX：
   <https://github.com/PaddlePaddle/Paddle2ONNX>

6. PaddleOCR Android 示例源码：
   <https://github.com/PaddlePaddle/PaddleOCR/tree/main/deploy/ppocr-android>

---

## 18. 最终建议

对于当前项目，最稳妥的实施顺序是：

```text
官方 PP-OCRv6 Android ONNX 示例
        ↓
替换为 PP-OCRv6_small
        ↓
移植当前过滤和可视化逻辑
        ↓
CameraX 实时闭环
        ↓
CPU 基准
        ↓
NNAPI
        ↓
QNN GPU
        ↓
QNN HTP/NPU
```

第一版产品目标应是：

```text
PP-OCRv6_small ONNX
+ ONNX Runtime Android
+ CameraX
+ CPU/NNAPI 回退
+ 稳定的目标数字检测和冻结
```

等功能和结果稳定后，再投入 QNN HTP/NPU 优化。这样即使 Snapdragon 驱动或 QNN 后端不可用，应用仍然可以通过 CPU 正常工作。
