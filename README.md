# sd-webui-ChatGLM

这个扩展将 [ChatGLM](https://github.com/THUDM/ChatGLM2-6B) 整合到了 [Stable Diffusion WebUi](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 内部

## 安装

你可能需要先手动安装 `requirments.txt` 中的依赖库，将扩展文件夹放置在 `extensions` 文件夹下即可。

然后，下载所有的 [ChatGLM 模型文件](https://huggingface.co/THUDM/chatglm2-6b)，将模型文件放置在扩展文件夹中的 `model` 文件夹中即可

使用 pip 安装 `icetk` 时，可能提示需要 `protobuf<3.20.1`，但经过测试，`protobuf==3.20.3` 仍然能够正常的运行。

最后，启动 WebUI 即可。

## 功能

- 与 AI 模型对话
- 修改自己发送的内容或 AI 回复的内容
- 保存对话内容

## 提示

[ChatGLM 语言模型](https://huggingface.co/THUDM/chatglm2-6b)需要消耗较多显存，建议显存为 8G 以上的设备使用。

请使用 [WebUI v1.6.0](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/5ef669de080814067961f28357256e8fe27544f4) 及以上的版本。以前版本的 WebUI 需要 `Transformer==4.25.1`，与 [ChatGLM](https://github.com/THUDM/ChatGLM2-6B) 需求的 Transformer 版本相互冲突，导致无法运行。
