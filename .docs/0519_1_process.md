可以，按 conda 的 roco 环境跑即可。我看了一下仓库，当前环境里已经是：

  /root/miniconda3/envs/roco/bin/python

  而且核心依赖能 import：

  import mujoco, dm_control, rocobench.envs

  ## 1. evaluate 入口在哪里

  代码目录：

  cd /inspire/hdd/project/26summer-camp-09/26220478/code

  主入口：

  python evaluator.py

  evaluator.py 会依次跑 6 个任务，每个任务默认 5 次：

  sort
  cabinet
  rope
  sweep
  sandwich
  pack

  底层实际调用的是：

  python run_dialog.py --task <task> --num_runs 5 --skip_display ...

  LLM 调用位置已经接好了 Ollama：

  prompting/llm_client.py
  prompting/plan_prompter.py
  prompting/dialog_prompter.py

  默认模型名来自：

  ROCO_LLM_MODEL

  如果不设，默认是：

  llama3.3:70b

  ———

  ## 2. 推荐运行方式：GPU 机器跑 Ollama，评测机器通过 HTTP 调用

  因为代码里调用 Ollama 的 HTTP API：

  /api/chat

  所以共享磁盘只能共享代码和日志，不能通过磁盘本身调用模型。两种方案：

  ### 方案 A：evaluate 和 Ollama 都在 GPU 机器跑

  最简单、最稳。

  ### 方案 B：evaluate 在当前机器跑，Ollama 在 GPU 机器跑

  需要当前机器能访问 GPU 机器的 11434 端口。

  ———

  ## 3. GPU 机器上启动 Ollama

  在 GPU 机器上：

  conda activate roco  # 如果只跑 ollama，其实不一定需要 roco
  export OLLAMA_HOST=0.0.0.0:11434
  export OLLAMA_KEEP_ALIVE=24h
  ollama serve

  另开一个终端检查模型：

  ollama list
  ollama ps
  nvidia-smi

  当前磁盘里我看到已有：

  llama3.3:70b    42 GB

  如果 GPU 机器也能看到同一个 OLLAMA_MODELS 目录，应该不用重新 pull。

  测试 Ollama API：

  curl http://127.0.0.1:11434/api/tags

  从评测机器测试远程 GPU Ollama：

  curl http://<GPU机器IP>:11434/api/tags

  ———

  ## 4. 评测机器上设置环境变量

  在跑 evaluate 的机器上：

  conda activate roco

  cd /inspire/hdd/project/26summer-camp-09/26220478/code

  export ROCO_LLM_MODEL=llama3.3:70b
  export OLLAMA_BASE_URL=http://<GPU机器IP>:11434

  如果 evaluate 和 Ollama 在同一台 GPU 机器：

  export OLLAMA_BASE_URL=http://127.0.0.1:11434

  建议显式设置输出目录到共享盘：

  export SHARED_RUN_DIR=/inspire/hdd/project/26summer-camp-09/rocobench_runs
  export ROCO_DATA_DIR=$SHARED_RUN_DIR/data
  export ROCO_EVAL_OUTPUT_DIR=$SHARED_RUN_DIR/output
  export MPLCONFIGDIR=$SHARED_RUN_DIR/mplconfig

  mkdir -p "$ROCO_DATA_DIR" "$ROCO_EVAL_OUTPUT_DIR" "$MPLCONFIGDIR"

  ———

  ## 5. 先跑单任务 smoke test

  不要一上来跑完整 evaluator，先跑一个 sort：

  python run_dialog.py \
    --task sort \
    --num_runs 1 \
    --run_name smoke_sort_llama70b \
    --data_dir "$ROCO_DATA_DIR" \
    --skip_display \
    --comm_mode plan \
    --llm_source "$ROCO_LLM_MODEL" \
    --tsteps 3 \
    --num_replans 1 \
    --run_timeout 600

  如果无显示环境有 MuJoCo/OpenGL 问题，用：

  xvfb-run -a python run_dialog.py \
    --task sort \
    --num_runs 1 \
    --run_name smoke_sort_llama70b \
    --data_dir "$ROCO_DATA_DIR" \
    --skip_display \
    --comm_mode plan \
    --llm_source "$ROCO_LLM_MODEL" \
    --tsteps 3 \
    --num_replans 1 \
    --run_timeout 600

  ———

  ## 6. 跑完整 evaluate

  确认单任务能跑后：

  python evaluator.py

  无显示服务器建议：

  xvfb-run -a python evaluator.py

  后台跑：

  nohup xvfb-run -a python evaluator.py > eval_stdout.log 2>&1 &

  查看日志：

  tail -f eval_stdout.log

  ———

  ## 7. 输出在哪里看

  如果设置了：

  export ROCO_EVAL_OUTPUT_DIR=/inspire/hdd/project/26summer-camp-09/rocobench_runs/output

  那么完整评测输出在：

  /inspire/hdd/project/26summer-camp-09/rocobench_runs/output/run_YYYYMMDD_HHMMSS/

  里面有：

  evaluator.log
  summary.json
  tasks/
    01_sort/
    02_cabinet/
    ...

  每个 task 里有：

  stdout.log
  stderr.log
  summary.json
  runs/run_0/
  runs/run_1/
  ...

  ———

  ## 8. 最关键注意点

  1. evaluator.py 不显式传 --llm_source，所以要靠环境变量：

     export ROCO_LLM_MODEL=llama3.3:70b

  2. 客户端必须设置 Ollama 地址：

     export OLLAMA_BASE_URL=http://<GPU机器IP>:11434

  3. 如果没设，代码默认连：

     http://127.0.0.1:11434

     也就是当前机器本地 Ollama。

  4. 当前机器我看到 NVIDIA_VISIBLE_DEVICES= 是空的，说明这里大概率没有 GPU。70B 要在 GPU 机器上跑 Ollama。
  5. llama3.3:70b 约 42GB，最好用 A100/H100 80GB 级别 GPU。跑的时候看：

     ollama ps
     nvidia-smi

     确认不是 CPU 在硬跑。