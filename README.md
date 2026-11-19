# TMM: Multi-Modal 3D Trajectory Prediction

Trajectory Mapping Model（TMM）的离线合成轨迹实验代码。包含 **TMM-F**、**TMM-N** 和 **CV / CA** 两个运动学基线，不需要下载外部数据或预训练权重。

![TMM-F、TMM-N、CV 和 CA 的 5 cm 加噪测试结果](assets/benchmark_5cm.png)

## 模型

```text
50 点三维观测 → 坐标处理（F：原始 / N：归一化）
  → 离散 Fréchet 距离 + 贪心代表轨迹 + RBF 映射
  → 双层 MLP-Mixer + 混合密度网络（MDN）
  → 采样 RBF 轨迹系数 → 解码 25 个未来点 → 末点 GMM 聚类
  → 多条候选轨迹及其概率（N 输出还原到原始坐标）
```

TMM-F 和 TMM-N 使用相同网络结构，但分别选择代表轨迹、独立训练并各自选择最佳轮次。TMM-F 保留原始坐标；TMM-N 用观测段确定平移、旋转和尺度，输出再换回米制坐标。

## 1. 配置环境

打开终端，进入包含 `train.py` 的仓库根目录。已安装项目依赖的 `ktm` 环境可以直接使用：

```sh
conda activate ktm
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

如果从零开始，先准备 Conda，再新建 Python 3.10 环境：

```sh
conda create -n tmm python=3.10 pip -y
conda activate tmm
```

Windows / Linux 下选择**一条** PyTorch 安装命令（见 [PyTorch 2.7.0 官方安装说明](https://pytorch.org/get-started/previous-versions/#v270)）。无 NVIDIA GPU 时安装 CPU 版；使用 NVIDIA GPU 时需要支持 CUDA 11.8 的驱动：

```sh
# CPU
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
# 或 CUDA 11.8
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

安装其余依赖并检查环境：

```sh
python -m pip install -r requirements.txt
python -c "import numpy, numba, sklearn, matplotlib, torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

图中结果实测于 Windows、Python 3.10、PyTorch 2.7.0 + CUDA 11.8 和 RTX 4060 Laptop GPU；CPU 也可运行。固定种子控制数据生成、划分、模型初始化、训练批次和轨迹采样，不保证跨硬件或依赖版本的结果逐位一致。

## 2. 准备数据

仓库内的 `data/trajectories.npz` 可直接使用。重新生成同一份数据：

```sh
python generate_trajectories.py --count 10000 --seed 42 --noise-std 0.05
```

上述参数也是默认值。数据包含 10,000 条直飞、缓转或急转的合成 3D 轨迹，点间隔 **0.2 秒**。观测阶段每个点的 X、Y、Z 分别加入独立的零均值高斯噪声，**每轴标准差为 0.05 米**；未来真值不加噪。

| 数组 | 形状 | 内容 |
| --- | --- | --- |
| `observations` | `(10000, 50, 3)` | 带噪历史坐标，单位米 |
| `predictions` | `(10000, 25, 3)` | 不带噪的未来真值，预测时长 5 秒 |
| `kinds` | `(10000,)` | 运动类别 |

## 3. 训练与评估

**快速检查**：用 512 条轨迹训练 3 轮，验证环境和输出，不用于对比下方表格：

```sh
python train.py --variant both --seed 42 --epochs 3 --max-trajectories 512 --val-cases 12 --val-samples 48 --samples 80 --output-dir runs/smoke
```

**运行表中实验**：两版分别训练 50 轮，使用相同的 7,000 / 1,000 / 2,000 条训练、验证、测试划分：

```sh
python train.py --variant both --seed 42 --epochs 50 --output-dir runs/benchmark
```

也可以独立运行：

```sh
python train.py --variant f --seed 42 --epochs 50 --output-dir runs/only_f
python train.py --variant n --seed 42 --epochs 50 --output-dir runs/only_n
```

- 不指定 `--device` 时自动选择 CUDA 或 CPU。可加 `--device cpu` 强制 CPU，或加 `--device cuda` 检查 CUDA 是否可用。
- `--data` 可指定由本仓库生成的其他 `.npz`。代表轨迹仅从训练集选取。完整规模的 Fréchet 距离计算在 CPU 上预处理，每版距离矩阵约 374 MiB。
- 默认每条测试轨迹采样 300 次，再按末点进行 GMM 聚类。每版独立选取验证集概率加权 DF（`M-DF`）最低的轮次，然后在留出的测试集上评估。

## 4. 阅读日志和结果

终端依次显示设备、种子、样本划分、两版的预处理和每轮训练信息。`train NLL` / `val NLL` 是训练集 / 验证集上的混合密度负对数似然，`val M-DF` 是验证集概率加权离散 Fréchet 距离（米）。最后会打印最佳轮次、测试表格和结果目录。

| 文件 | 内容 |
| --- | --- |
| `runs/benchmark/metrics.json` | 实验配置、划分规模、各版最佳轮次和测试指标 |
| `runs/benchmark/comparison.png` | 各版 top-1、CV、CA 的测试指标对比 |
| `runs/benchmark/tmm-f/` | TMM-F 的 `history.csv`、`learning_curves.png`、`trajectory.png` |
| `runs/benchmark/tmm-n/` | TMM-N 对应的训练记录、曲线和轨迹示例 |

### 指标

| 指标 | 含义 |
| --- | --- |
| **ADE** | 25 个未来点逐步对应的欧氏距离取平均 |
| **ED / FDE** | 只算最后一个未来点的欧氏距离。论文 ED 与 FDE 同义，JSON 键为 `ED` |
| **DF** | 整段预测与真值之间的离散 Fréchet 距离，比较有序轨迹的整体偏差 |

三项均以米计、越小越好。指标在测试样本间取平均。TMM 的多模态结果分三种口径：

- **top-1**：只评价概率最高的候选轨迹。
- **weighted**：先在每条样本内按候选概率加权，再跨样本平均。ED / DF 对应论文 M-ED / M-DF。
- **oracle**：对每项指标分别选择与真值误差最小的候选。ED / DF 对应论文 B-ED / B-DF；用于评估候选质量，不是实际预测时可用的选模规则。

### 示例结果：5 cm 观测噪声

两版各训练 50 轮，最佳轮次分别为 50（TMM-F）和 14（TMM-N）。下表是同一批 2,000 条测试轨迹上的结果：

| 方法 | ADE (m) ↓ | ED / FDE (m) ↓ | DF (m) ↓ |
| --- | ---: | ---: | ---: |
| TMM-F top-1 | 3.41 | 7.58 | 7.64 |
| **TMM-N top-1** | **3.30** | **7.20** | **7.41** |
| CV | 3.58 | 8.99 | 8.99 |
| CA | 22.93 | 63.60 | 63.60 |

本组数据上，TMM-N 的 top-1 三项误差最低。CA 未对观测做平滑，末三帧的二阶差分估计对噪声敏感，误差随预测时长放大。

| 变体 | 预测轨迹 | 训练曲线 |
| --- | --- | --- |
| TMM-F | [轨迹示例](assets/tmm_f_example.png) | [学习曲线](assets/tmm_f_learning_curves.png) |
| TMM-N | [轨迹示例](assets/tmm_n_example.png) | [学习曲线](assets/tmm_n_learning_curves.png) |

两张轨迹图使用同一条测试样本，展示带噪观测、无噪声真值、TMM 的候选轨迹与概率，以及 CV / CA 对照。[完整实测指标（JSON）](assets/benchmark_5cm.json) 包含 top-1、weighted 和 oracle 的所有分数。

## 代码结构

| 路径 | 用途 |
| --- | --- |
| `generate_trajectories.py` | 生成固定种子的合成观测和未来真值 |
| `train.py` | 数据划分、独立训练、验证选模、测试评估 |
| `tmm.py` | MLP-Mixer、MDN、核岭回归系数和多模态采样解码 |
| `geometry.py` | 坐标变换、离散 Fréchet、代表轨迹选择和 CV / CA |
| `report.py` | 输出学习曲线、轨迹示例和测试对比图 |
| `data/` | 发布用合成轨迹 |
| `assets/` | 已完成的完整实验：指标 JSON、对比图、轨迹图和学习曲线 |

`runs/` 为本地输出，不进入版本控制；本项目不分发旧数据、预处理缓存或模型权重。若要保存终端日志，可在 Bash 或 PowerShell 中给训练命令追加 `2>&1 | tee benchmark.log`。

## 引用

J. Ma et al., *TMM: Trajectory Mapping Model Framework for Multi-Modal Probabilistic UAV Motion Prediction*, IEEE Robotics and Automation Letters. DOI: [10.1109/LRA.2026.3748666](https://doi.org/10.1109/LRA.2026.3748666).
