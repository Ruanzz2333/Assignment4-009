# 大作业任务梳理

## 🎯 核心目标

在 **RoboTwin 2.0** 仿真平台上，为 **Galbot 双臂机器人**训练一个**扩散策略（Diffusion Policy, DP）**，使其能完成 `beat_block_hammer`（拿锤敲木块）任务。采用**仿真+真实联合训练（sim-and-real co-training）**的范式。

---

## 📦 任务本身：`beat_block_hammer`

分两步：

1. 用右夹爪抓起桌上的锤子
2. 把锤子移到木块上并**敲击**一次（不需要抬起、不需要松开夹爪归位）

---

## 📂 工作区各模块解析

| 模块 | 路径 | 作用 |
|---|---|---|
| **环境定义** | `envs/` | 基于 SAPIEN 物理引擎的仿真环境，包含机器人模型、相机系统、动作控制、物体生成等 |
| **具体任务** | `envs/beat_block_hammer.py` | 继承基类 `Base_Task`，实现锤子抓取、敲击的物理逻辑 |
| **全局配置** | `envs/_GLOBAL_CONFIGS.py` | 路径、抓取方向、世界坐标系等全局常量 |
| **机器人控制** | `envs/robot/` | 双臂运动学/逆运动学、夹爪控制、路径规划（Mplib/CuRoBo） |
| **相机系统** | `envs/camera/` | D435 深度相机，RGB+深度+点云采集 |
| **扩散策略** | `policy/DP/` | DP 模型训练、部署、数据预处理的核心代码 |
| **数据收集** | `script/collect_galbot_beat_block_hammer_dataset.py` | 在仿真中收集演示数据（遥操作轨迹） |
| **策略评估** | `script/eval_policy.py` | 加载训练好的 checkpoint，在仿真/真实环境中跑若干 episode 计算成功率 |
| **任务配置** | `task_config/` | 定义「干净」（`galbot_demo_clean`）和「随机增强」（`galbot_demo_randomized`）两种数据采集配置 |
| **代码生成** | `code_gen/` | 用 LLM 辅助生成任务的 `play_once` 控制代码（非本项目直接要求） |
| **文本描述** | `description/` | 生成任务指令、对象描述的 prompt 模板 |
| **数据处理** | `data/` | 存放采集到的 HDF5 轨迹数据以及卡顿检测脚本 `process_stuck.py` |

---

## 🔄 完整流水线（四步）

```
┌─────────────────────────────────────────────────────┐
│ ① 数据采集（Data Collection）                        │
│   仿真中运行遥操作，收集 50 条右臂演示轨迹 → HDF5    │
│   参数：--clean-episodes 50 --force-arm-tag right    │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ ② 数据处理（Process Data）                          │
│   提取单头部相机 + 仅右臂 8D 动作 → Zarr 格式        │
│   python policy/DP/process_data.py ... --right-arm-only│
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ ③ 训练（Train）                                     │
│   用 Diffusion Policy 训练 600 epochs                │
│   批次大小 48，设备 cuda:0                           │
│   产出：checkpoint .ckpt 文件                        │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ ④ 评估（Evaluate）                                  │
│   加载 checkpoint，仿真中跑 50 个测试 episode        │
│   计算成功率 → ≥70% 即满分                           │
│   生成评估视频                                       │
└─────────────────────────────────────────────────────┘
```

---

## ✅ 各阶段要求

### 当前阶段 —— Part I：仿真评估（25%）

1. **配好环境**：装 RoboTwin + 下载 Galbot 资产
2. **跑通基线流水线**：把 README 里的四步命令执行一遍，确认能跑通
3. **提升成功率到 ≥70%**：允许修改两个东西 ——
   - **数据生成方法**：增加仿真初始状态的随机性（锤子/木块初始位姿、机器人初始关节角、桌面大小、背景纹理、光照等），生成更多样化的训练数据
   - **训练策略**：调整训练超参数等，但不能改模型架构
4. **提交 checkpoint**：把最终的 `.ckpt` 文件提交到 `course.pku.edu.cn`

### 后续阶段（2026年6月8日发布）

- **Part II（50%）**：真实世界分布内（in-distribution）评估 —— 用少量真实遥操作数据 + 大量仿真数据联合训练
- **Part III（25%）**：真实世界分布外（out-of-distribution）评估 —— 测试泛化到仿真见过但真实没见过的场景

---

## ⚡ 关键约束切记

- ❌ **不能修改扩散策略的模型架构**（DP 的 transformer/网络结构不能动）
- ❌ **不能用任何其他类型的模型**（必须是 Diffusion Policy）
- ✅ 可以改数据生成（加随机化、改场景配置）
- ✅ 可以改训练策略（超参数、联合训练方式等）
- ⚠️ 真实数据发布前（6月8日前），不建议大改轨迹规划方法

---

## 🚀 Part I 改进历程

### 实验 1：基线（h8o3a6，50 条干净数据）

直接使用 README 提供的默认配置跑通流水线，模型完成训练后在干净场景评估。

- **现象**：大部分 episode 能抓起锤子，但第一下锤到木块附近后出现水平方向抽搐，最终步数耗尽失败。
- **分析**：扩散策略为纯模仿学习，训练数据全是完美轨迹，模型从未见过"偏移了几厘米"的状态。一旦首次落点有偏差就进入分布外，无法自我纠偏。
- **成功率**：~80%（干净场景）

### 实验 2：尝试随机场景数据（100 条随机数据）

采集 100 条随机场景数据（含随机背景、杂物桌面、随机光照），单独训练。

- **现象**：随机环境下能抓起锤子但定位不准；在干净场景评估反而表现下降。
- **分析**：随机数据中的杂物和光照变化增加了噪声，ResNet18 特征提取被干扰，定位精度反而不如纯干净数据训出的模型。
- **结论**：Part I 只需要干净场景 ≥70%，随机数据留到 Part II/III 使用。

### 实验 3：扩大木块位置范围（100 条干净数据，h16o4a8）

核心思路：让专家轨迹覆盖更多起始位置，模型学会"从不同距离接近木块"。

**改了什么：**

1. `envs/beat_block_hammer.py`：扩大木块初始位置范围
   ```python
   block_x_offset = [0.03, 0.35]        # 原 [0.05, 0.25]，范围翻倍
   block_y_from_near_edge = [0.10, 0.50] # 原 [0.15, 0.45]
   ```
2. `task_config/galbot_demo_clean.yml`：桌子小幅扩大
   ```yaml
   table_length: 0.95   # 原 0.90
   table_width: 0.60    # 原 0.55
   ```
3. 训练超参数：
   ```
   horizon=16, n_obs_steps=4, n_action_steps=8
   100 条干净数据, 800 epochs
   ```
   更长的 horizon（16 vs 原始 8）让模型更早规划敲击路径；更多观测步数（4 vs 3）增强运动趋势感知。

**评估时恢复到初始配置，额外加 `--n_action_steps 6`** 提升纠偏频率。

- **最终成功率**：**92%（干净场景）**
- **关键收获**：对模仿学习来说，数据覆盖的**起始条件多样性**比数据条数更重要。扩大 block offset 范围让专家轨迹覆盖了不同距离的接近路径，模型学会了适应偏差而非死记固定模式。

### 失败路径总结

| 尝试 | 结果 | 原因 |
|---|---|---|
| 纯随机数据训练 → 干净评估 | 定位精度下降 | 噪声干扰特征提取 |
| 混合训练（干净50+随机100） | 干净评估尚可，随机评估抽搐 | 随机场景分布太广，模型无法兼顾 |
| 桌子改得太大（1.10×0.70） | 全部采集失败 | 超出机器人臂展范围 |
| 改 n_action_steps=2 | 建议放弃 | 训练碎片化严重 |

### 最终配置速查

| 项目 | 值 |
|---|---|
| 训练数据 | 100 条干净（galbot_demo_clean） |
| 训练参数 | h16o4a8, batch=48, epochs=800 |
| 评估参数 | `--n_obs_steps 4 --n_action_steps 6` |
| block_x_offset | `[0.03, 0.35]` |
| block_y_from_near_edge | `[0.10, 0.50]` |
| table_length | 0.95 |
| table_width | 0.60 |
| ✅ Part I 成功率 | **92%** |

---

## 📖 命令参数速查

### 训练命令参数

| 参数 | 基线值 | 含义 |
|---|---|---|
| `--config-name` | `robot_dp_8.yaml` | Hydra 主配置，定义 DP 模型结构 |
| `task.name` | `beat_block_hammer` | 任务名称 |
| `task.dataset.zarr_path` | `.zarr 路径` | 训练数据集位置 |
| `training.num_epochs` | `600` | 训练轮次 |
| `training.seed` | `0` | 随机种子 |
| `training.device` | `cuda:0` | 训练设备 |
| `dataloader.batch_size` | `48` | 训练批次大小 |
| `val_dataloader.batch_size` | `48` | 验证批次大小 |
| `head_camera_type` | `D435` | 头部相机型号 |
| `expert_data_num` | `50` | 数据总量（用于统计归一化） |
| `setting` | `galbot_demo_clean` | 训练 setting 标识 |
| `exp_name` | `galbot_clean50_head_8d` | 实验名（决定 checkpoint 目录） |
| `logging.mode` | `offline` | 离线模式（不上传 wandb） |

### checkpoint路径命名构成
e.g. beat_block_hammer-galbot_demo_clean-8d-50-galbot_demo_clean-galbot_clean50_head_8d_h8o3a6-0
任务名称-数据集名称-setting标识-实验名-随机种子

### 评估命令参数

| 参数 | 基线值 | 含义 |
|---|---|---|
| `--config` | `policy/DP/deploy_policy.yml` | 部署推理配置 |
| `--overrides` | — | 允许命令行覆盖 YAML 配置 |
| `--policy_name` | `DP` | 使用 Diffusion Policy |
| `--task_name` | `beat_block_hammer` | 任务名称 |
| `--task_config` | `galbot_demo_clean` | 场景配置（干净/随机化） |
| `--ckpt_setting` | `galbot_demo_clean` | 匹配训练时的 `setting` |
| `--seed` | `0` | 起始种子（评估时偏移为 100000+） |
| `--instruction_type` | `unseen` | 使用未见过的语言指令 |
| `--expert_data_num` | `50` | 匹配训练数据量（归一化统计） |
| `--checkpoint_num` | `600` | 加载第几个 epoch 的权重 |
| `--ckpt_file` | `路径/600.ckpt` | checkpoint 完整路径 |
| `--action_dim` | `8` | 动作维度（右臂 7D + 夹爪 1D） |
| `--eval_video_log` | `True` | 保存评估视频 |
| `--eval_test_num` | `50` | 测试 episode 数量 |
| `--eval_step_lim` | `200` | 单 episode 最大步数 |
| `--force_arm_tag` | `right` | 强制右臂执行 |
| `--force_block_arm_tag` | `right` | 阻挡左臂 |

### ⚠️ 必须保持一致的关键参数

| 参数 | 原因 |
|---|---|
| `task_name` | 训练和评估必须是同一个任务 |
| `action_dim` | 必须匹配训练时的维度（8D） |
| `expert_data_num` | 必须等于训练数据量，否则归一化参数不匹配 |
| `ckpt_setting` / `setting` | 评估 `ckpt_setting` = 训练 `setting` |
| `head_camera_type` | 相机型号必须一致 |

---

### 虚拟机（Linux）— 跑训练

**首次 setup：**
```bash
# 克隆仓库
git clone https://github.com/Ruanzz2333/Assignment4-009.git
cd Assignment4-009

# 配环境（只需一次）
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
bash script/_install.sh
bash script/_download_assets.sh

# 下载 Galbot 资产，解压到 assets/embodiments/
```

**每次本地 push 后，虚拟机上：**
```bash
cd ~/Assignment4-009
conda activate RoboTwin

# 1. 暂存虚拟机上的临时文件（如果有）
git stash

# 2. 拉取最新
git pull

# 3. 恢复自己的临时文件
git stash pop

# 4. 跑训练/评估
ROOT=$(pwd)
python script/collect_galbot_beat_block_hammer_dataset.py \
    --clean-episodes 50 --skip-randomized --save-path data \
    --overwrite --skip-render-test --force-arm-tag right

# 后续：处理数据 → 训练 → 评估
```

### 文件传输

| 场景 | 命令 |
|---|---|
| 下载 checkpoint 到本地 | `scp user@vm_ip:~/Assignment4-009/policy/DP/checkpoints/xxx.ckpt .` |
| 上传文件到虚拟机 | `scp local_file user@vm_ip:~/Assignment4-009/` |
| 批量下载评估视频 | `scp -r user@vm_ip:~/Assignment4-009/eval_video/ .` |

### 一句话总结

> **本地改 → push → 虚拟机 pull → 跑实验 → scp 拿结果**

---

## 📋 快速启动清单

- [✅️] 克隆仓库到本地和虚拟机
- [✅️] 配好 RoboTwin conda 环境
- [✅️] 下载 Galbot 资产放到 `assets/embodiments/`
- [✅️] 跑通基线 50 条干净数据 → 训练 → 评估流水线
- [✅️] 修改数据生成加随机化，提升数据多样性
- [✅️] 调优训练策略，目标成功率 ≥ 70%
- [] 提交最终 checkpoint 到 course.pku.edu.cn
