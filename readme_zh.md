# 小组项目

在本项目中，我们将使用 [RoboTwin 2.0](https://robotwin-platform.github.io/doc/) 平台在 Galbot 上为 `beat_block_hammer` 任务训练一个扩散策略（diffusion policy）。该任务包含两个步骤：

1. 使用右侧夹爪（gripper）抓取锤子。
2. 将锤子移动至木块处，并用锤子敲击木块。

<video controls width="600">
  <source src="assets/files/episode0.mp4" type="video/mp4">
</video>

本仓库已经提供了数据生成、数据处理、策略训练以及仿真评估（simulation evaluation）流水线的基础实现。

本项目将采用**仿真与真实联合训练（sim-and-real co-training）**。在仿真环境（simulation environment）中可以收集大量且多样的仿真数据，同时我们也将会提供少量覆盖范围较为有限的真实世界遥操作（teleoperation）数据。真实世界数据将于 **2026年6月8日** 发布。

训练完成的策略将在以下设定中进行评估：

1. 仿真评估
2. 真实分布内（in-distribution）评估
3. 真实分布外（out-of-distribution）（但属于仿真分布内）评估

允许你修改仿真数据生成方法与仿真与真实联合训练策略。然而，**不允许**修改扩散策略的模型架构，也不得使用任何其他类型的模型。

## 环境

你可以通过遵循 [RoboTwin 2.0 官方文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) 中的说明来配置环境。主要步骤如下：

```bash
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
```

此外，你还需要下载并解压 [Galbot 资产](https://github.com/PKU-EPIC/Intro2EAI_2026/blob/master/static_files/assignments/galbot-one-golf.tar.gz)，然后将它们放置在 `assets/embodiments` 目录下。

## 第一部分：仿真评估 (25%)

仿真评估能够帮助验证训练和评估流水线的正确性，并对策略的能力提供初步估计。

在这一部分中，你可以修改数据生成方法和训练方法，但不可修改模型架构。我们将评估你的模型在仿真环境中的成功率。如果成功率大于或等于 **70%**，你将获得该部分的满分。

即使你的成功率未能达到 70%，只要你的实现基本正确，仍能获得该部分的大部分分数。

各步骤的命令列出如下：

```bash
ROOT=$(pwd)

# 1. 收集 50 条干净的右臂演示数据
rm -rf data/beat_block_hammer/galbot_demo_clean
python script/collect_galbot_beat_block_hammer_dataset.py \
    --clean-episodes 50 \
    --skip-randomized \
    --save-path data \
    --overwrite \
    --skip-render-test \
    --force-arm-tag right

# 2. 将数据处理为包含单头部相机、仅有右臂的 8D zarr 数据集
rm -rf policy/DP/data/beat_block_hammer-galbot_demo_clean-8d-50.zarr
python policy/DP/process_data.py beat_block_hammer galbot_demo_clean 50 \
    --load-dir data/beat_block_hammer/galbot_demo_clean \
    --save-dir policy/DP/data/beat_block_hammer-galbot_demo_clean-8d-50.zarr \
    --right-arm-only

# 3. 训练 600 个轮次
cd $ROOT/policy/DP
python train.py --config-name=robot_dp_8.yaml \
    task.name=beat_block_hammer \
    task.dataset.zarr_path=$ROOT/policy/DP/data/beat_block_hammer-galbot_demo_clean-8d-50.zarr \
    training.num_epochs=600 \
    training.seed=0 \
    training.device=cuda:0 \
    dataloader.batch_size=48 \
    val_dataloader.batch_size=48 \
    head_camera_type=D435 \
    expert_data_num=50 \
    setting=galbot_demo_clean \
    exp_name=galbot_clean50_head_8d \
    logging.mode=offline

# 4. 评估已训练的策略
cd $ROOT
CKPT=$ROOT/policy/DP/checkpoints/beat_block_hammer-galbot_demo_clean-8d-50-galbot_demo_clean-galbot_clean50_head_8d-0/600.ckpt
python script/eval_policy.py \
    --config policy/DP/deploy_policy.yml \
    --overrides \
    --policy_name DP \
    --task_name beat_block_hammer \
    --task_config galbot_demo_clean \
    --ckpt_setting galbot_demo_clean \
    --seed 0 \
    --instruction_type unseen \
    --expert_data_num 50 \
    --checkpoint_num 600 \
    --ckpt_file $CKPT \
    --action_dim 8 \
    --eval_video_log True \
    --eval_test_num 50 \
    --eval_step_lim 200 \
    --force_arm_tag right \
    --force_block_arm_tag right
```

为了生成更具多样性的数据，你可以增加仿真中初始状态的随机性，例如锤子和木块的初始位姿（poses）、机器人的初始位姿和关节位置（qpos）、桌子的尺寸，以及其他重要参数。你也可以使用 RoboTwin 已提供的接口来修改桌子和背景的纹理以及光照。

在真实世界数据发布之前，我们不建议对轨迹规划（trajectory planning）方法进行大幅修改。

## 第二部分：真实分布内评估 (50%)

此部分将于 **2026年6月8日** 发布。

## 第三部分：真实分布外评估 (25%)

此部分将于 **2026年6月8日** 发布。

## 提交

对于仿真评估，请在 `course.pku.edu.cn` 上提交你训练后的策略的检查点（checkpoint）。

关于真实世界评估的详细信息将于 **2026年6月8日** 发布。