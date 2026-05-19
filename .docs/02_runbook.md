# 运行手册 Runbook

## 1. 环境激活

官方建议使用已经配置好的 `roco` conda 环境：

```bash
conda activate roco
```

如果手动配置：

```bash
conda create -n robot python=3.8
conda activate robot
pip install -r requirements.txt
```

## 2. 环境检查

```bash
python -c "import mujoco; print('mujoco ok')"
python -c "import dm_control; print('dm_control ok')"
python -c "import rocobench.envs; print('rocobench envs ok')"
```

如果 `rocobench.envs` import 失败，说明当前代码包不完整，需要补齐官方任务环境。

## 3. 单任务运行

```bash
python run_dialog.py --task sort --num_runs 1 --skip_display --comm_mode plan
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--task` | 任务名：sort/cabinet/rope/sweep/sandwich/pack |
| `--num_runs` | 运行次数 |
| `--skip_display` | 不弹出显示窗口，适合服务器 |
| `--comm_mode` | 规划模式：plan/chat/dialog |
| `--llm_source` | 模型名 |
| `--tsteps` | 每个 episode 最大高层规划步数 |
| `--run_timeout` | 单个 run 超时时间，默认 600 秒 |

## 4. 无显示服务器运行

如果 MuJoCo 或 OpenGL 需要虚拟显示器：

```bash
xvfb-run python run_dialog.py --task sort --num_runs 1 --skip_display
```

## 5. 全量评测

```bash
python evaluator.py
```

输出通常在：

```text
output/run_YYYYMMDD_HHMMSS/
├── evaluator.log
├── summary.json
└── tasks/
```

## 6. 打包代码

```bash
chmod +x pack_code.sh
./pack_code.sh
```

注意不要提交：

- `output/`
- `data/`
- 大模型权重
- 大型日志文件
