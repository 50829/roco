# 模型部署与 Prompt 策略

## 1. 模型约束

项目要求使用 80B 以下大模型。可选模型包括：

- Llama 3.3 70B Instruct；
- Qwen 2.5 72B Instruct；
- 其他 80B 以下指令模型。

## 2. 推荐部署方式

推荐使用 Ollama / vLLM / SGLang，并暴露 OpenAI-compatible API。

不建议使用 `ollama-python`，因为它可能引入 `pydantic > 2`，与当前代码中的 `pydantic==1.10.4` 冲突。

## 3. API 接入位置

集中式规划：

```text
prompting/plan_prompter.py::query_once
```

对话式规划：

```text
prompting/dialog_prompter.py::query_once
```

## 4. 推荐输出格式约束

在 prompt 中强制加入：

```text
You must output exactly one EXECUTE block.
Do not output explanations.
Do not output markdown.
Do not use Chinese.
Each robot must have exactly one line.
Use only valid robot names and object names from the observation.
Use only valid actions from the action instruction.
If a robot should do nothing, output WAIT.
```

标准格式：

```text
EXECUTE
NAME Alice ACTION ...
NAME Bob ACTION ...
NAME Chad ACTION ...
```

## 5. 通用规划原则

- 一次只让必要机器人行动，其他机器人 `WAIT`；
- 不要让两个机器人抓同一个物体；
- 不要让两个机器人同时穿过同一区域；
- 不确定时先 `MOVE` 到安全位置；
- 对需要 PATH 的任务，路径点要高一点、平滑一点、避开桌面和其他机器人；
- 如果反馈出现 IK 或 collision，下一次规划应更保守。

## 6. Response 清洗建议

模型输出后可做轻量清洗：

1. 去掉 ```markdown 代码块；
2. 截取第一个 `EXECUTE` 之后的文本；
3. 去掉多余首尾空白；
4. 如果没有 `EXECUTE`，可以在日志中记录并触发重试；
5. 谨慎自动补 `WAIT`，只在缺失机器人且其他动作格式完全合法时使用。
