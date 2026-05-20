  ### code/

  code/ 是基于官方版本修改过的工作版本。主要变化：

  #### 1. 已增加 Ollama 调用封装

  新增文件：

  code/prompting/llm_client.py

  它通过 Ollama 原生 HTTP 接口调用模型：

  POST http://127.0.0.1:11434/api/chat

  好处是不用安装 ollama-python，避免和项目固定的：

  pydantic==1.10.4

  冲突。

  ———

  #### 2. plan_prompter.py 已接入模型

  code/ 版本中：

  from .llm_client import query_ollama_chat

  并在 query_once() 里实际调用：

  response, usage = query_ollama_chat(...)

  所以集中式规划 --comm_mode plan 可以走 Ollama。

  ———

  #### 3. dialog_prompter.py 也已接入模型

  多机器人对话模式 --comm_mode dialog 也改成了调用：

  query_ollama_chat(...)

  ———

  #### 4. 默认运行模式改了

  code_v1_0/run_dialog.py 默认：

  --comm_mode chat
  --llm_source llama3.3:latest
  --data_dir data

  code/run_dialog.py 默认：

  --comm_mode plan
  --llm_source llama3.3:70b
  --data_dir /inspire/.../rocobench_runs/data

  也就是说 code/ 更适合当前环境直接跑。

  ———

  #### 5. utils.py 默认输出路径改了

  code/ 默认把评测输出放到共享目录：

  /inspire/qb-ilm2/project/26summer-camp-09/public/inspire_shared/mount/26220478/rocobench_runs/output

  也可以用环境变量覆盖：

  export ROCO_EVAL_OUTPUT_DIR=/your/path/output
  export ROCO_DATA_DIR=/your/path/data

  ———

  #### 6. code/ 里已有运行痕迹

  code/ 额外有：

  code/__pycache__/
  code/data/

  这些是之前运行产生的缓存和数据。提交代码时建议不要带这些。