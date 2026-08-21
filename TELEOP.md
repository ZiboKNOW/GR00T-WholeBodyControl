# PICO 遥操与 VLA 数据录制指南

面向本仓库本地使用的操作手册（从零配置 PICO → 遥操 → 录数据）。官方文档：

- [VR Teleop Setup (PICO)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)
- [PICO VR Whole-body Teleop](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html)
- [Data Collection for VLA](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html)

> **安全**：全身遥操动作快、幅度大。务必保持安全区，键盘操作员随时准备急停。  
> 急停：C++ 终端按 **`O`**，或手柄同时按 **A+B+X+Y**。  
> 穿紧身裤，保证脚部 tracker 可见。

---

## 目录

1. [前置条件总览](#1-前置条件总览)
2. [从零配置 PICO（一次性）](#2-从零配置-pico一次性)
3. [软件环境安装](#3-软件环境安装)
4. [只遥操（不录数据）](#4-只遥操不录数据)
5. [遥操 + 录数据（推荐）](#5-遥操--录数据推荐)
6. [手动多终端录数据](#6-手动多终端录数据备选)
7. [首次遥操流程](#7-首次遥操流程)
8. [手柄速查](#8-手柄速查)
9. [数据流架构](#9-数据流架构简图)
10. [录后处理](#10-录后处理)
11. [最短路径 checklist](#11-最短路径-checklist)
12. [常见问题](#12-常见问题)

---

## 1. 前置条件总览

| 项 | 说明 |
|---|---|
| Quick Start | 能跑通 sim2sim，checkpoint 已下载 |
| PICO 硬件 + XRoboToolkit | 见第 2 节（头显 / 手柄 / 脚部 tracker / PC Service / App） |
| `.venv_teleop` | `bash install_scripts/install_pico.sh` |
| `.venv_data_collection` | 仅录数据时需要：`bash install_scripts/install_data_collection.sh` |
| 真机相机 | 机器人端 camera server（仿真不需要） |

若 **PICO 已配好**，可跳过第 2 节，从第 3 / 4 节开始。

---

## 2. 从零配置 PICO（一次性）

内容对齐官方 [VR Teleop Setup (PICO)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)。

### 2.1 所需硬件

- PICO 4 / PICO 4 Pro 头显
- 2× PICO 手柄
- 2× PICO Motion Tracker（绑脚踝）
- **高速低延迟 Wi-Fi**（遥操对网络质量很敏感）

### 2.2 Step 1：安装 XRoboToolkit

XRoboToolkit = **PC Service**（工作站）+ **PICO App**（头显）。必须先装好并启动 PC Service，头显才能连上。

#### PC Service（工作站）

**Ubuntu 22.04 (x86_64)：**

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

**Ubuntu 24.04 (x86_64)：**

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
```

**Jetson (aarch64，机载)：**

```bash
sudo dpkg -i gear_sonic_deploy/thirdparty/roboticsservice_1.0.0.0_arm64.deb
```

其他平台 / 新版本见 [XRoboToolkit-PC-Service releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)。

#### PICO App（头显内安装）

1. 戴上头显，完成 PICO 初始设置。
2. 确认头显已连 Wi-Fi。
3. 打开头显浏览器，搜索 **xrobotoolkit**，进入 [XR-Robotics GitHub](https://github.com/XR-Robotics)。
4. 开启 **Developer Mode**（Settings → Developer）。
5. 在 GitHub 页面向下找到 APK，用扳机下载（推荐 [XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk)；[其他版本](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases)）。
6. 浏览器右上角管理下载 → 打开 APK → **Install**。
7. 应用会出现在 Library 的 **Unknown** 分区。

### 2.3 Step 2：Motion Tracker 配对与标定

#### 配对

1. 左右脚踝各绑一个 tracker；**卷起宽松裤腿**，指示灯朝上。
2. PICO Settings → 左侧最后一项 **Developer** → 关闭 **Safeguard**。  
   - 若没有 Developer：连点 **Software** 直到出现。
3. 点菜单里的 **Wi-Fi 图标** → 头显图上方的小圆标进入 Motion Tracker（没有则直接开 **Motion Tracker** App）。
4. 每个 tracker 旁点 **i** → **unpair** 全部清除。
5. 右上角点 **Pair**。
6. 每个 tracker **顶部按钮长按约 6 秒**进入配对（红蓝闪烁）。

#### 标定

1. 头显戴在眼前。
2. 点蓝色 **Calibrate**，完成两段：
   - **Sequence 1**：站直，双手柄自然垂在身侧。
   - **Sequence 2**：低头看脚部 tracker，直到头显摄像头识别到。
3. 标定完成后，可将头显戴在额头（仍朝前，以便继续看到脚部 tracker）。

### 2.4 Step 3：连接 PICO ↔ 工作站

1. **PC 与 PICO 必须同一 Wi-Fi**；记下工作站 IPv4。
2. 打开头显里的 **XRoboToolKit** App。
3. 在 **PC Service:** 旁点 **Enter**，填入工作站 IP。  
   - Status 显示 **WORKING** 即连接成功。  
   - IP 已填好可点 **Reconnect**。
4. 勾选 / 设置（与官方截图一致）：
   - Tracking：**Head**、**Controller**
   - Data/Control：选 **Send**
   - Pico Motion Tracker：选 **Full body**

至此硬件链路就绪。接下来装软件环境并开始遥操。

---

## 3. 软件环境安装

在仓库根目录：

```bash
# PICO 遥操 Python 环境 → .venv_teleop
# 含 teleop / sim extras、XRoboToolkit SDK、Unitree SDK2 等
bash install_scripts/install_pico.sh
source .venv_teleop/bin/activate   # 提示符: (gear_sonic_teleop)

# 数据录制环境 → .venv_data_collection（仅录数据时需要）
bash install_scripts/install_data_collection.sh
```

真机相机服务（在机器人 Jetson 上）：

```bash
bash install_scripts/install_camera_server.sh
sudo systemctl status composed_camera_server.service
```

G1 默认相机/机器人 IP 常为 `192.168.123.164`，端口 `5555`。

---

## 4. 只遥操（不录数据）

### 4.1 仿真

**终端 1 — MuJoCo**

```bash
source .venv_teleop/bin/activate   # 或 .venv_sim
python gear_sonic/scripts/run_sim_loop.py
```

**终端 2 — C++ Deploy**

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim
# 等到 "Init done"
```

**终端 3 — PICO Streamer**

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --vis_vr3pt --vis_smpl
```

### 4.2 真机

先停掉任何 `run_sim_loop.py`。

**终端 1 — C++ Deploy**

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real
# 自动检测失败时：./deploy.sh --input-type zmq_manager <G1-IP>
```

**终端 2 — PICO Streamer**

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --vis_vr3pt --vis_smpl
```

若无可视化窗口 / 无 pose：回到第 2.4 节检查 XRoboToolkit Status 是否为 **WORKING**。

---

## 5. 遥操 + 录数据（推荐）

一键 tmux 启动（无需先 activate venv）：

```bash
# 仿真
python gear_sonic/scripts/launch_data_collection.py --sim \
    --task-prompt "pick up the cup"

# 真机
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host 192.168.123.164 \
    --task-prompt "pick up the cup"

# 同时录腕部相机
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host 192.168.123.164 \
    --task-prompt "pick up the cup" \
    --record-wrist-cameras
```

tmux 布局：

```
┌───────────────────────┬───────────────────────┐
│ Pane 0: C++ Deploy    │ Pane 2: Data Exporter │
├───────────────────────┼───────────────────────┤
│ Pane 1: PICO Teleop   │ Pane 3: Camera Viewer │
└───────────────────────┴───────────────────────┘
```

| 操作 | 命令 |
|---|---|
| 切窗格 | `Ctrl+b` + 方向键（支持鼠标点选） |
| Detach | `Ctrl+b` 再 `d` |
| 重连 | `tmux attach -t sonic_data_collection` |
| 结束 | 任意窗格 `Ctrl+\`，或 `tmux kill-session -t sonic_data_collection` |

常用参数：

| Flag | 默认 | 说明 |
|---|---|---|
| `--task-prompt` | `"demo"` | 任务语言描述 |
| `--dataset-name` | 时间戳 | 省略则自动生成 |
| `--sim` / `--no-sim` | 真机 | 仿真模式 |
| `--camera-host` | `localhost` | 真机相机 IP |
| `--camera-port` | `5555` | 相机端口 |
| `--record-wrist-cameras` | off | 录左右腕相机 |
| `--data-exporter-frequency` | `50` | 录制频率 Hz |
| `--no-camera-viewer` | viewer on | 关掉预览窗格 |
| `--no-text-to-speech` | TTS on | 关掉语音提示 |

数据输出目录：`outputs/<dataset-name>/`（LeRobot v2.1）。

---

## 6. 手动多终端录数据（备选）

若不用 launcher，按顺序开：

1. **仿真**：`run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port 5555`（真机跳过）
2. **Deploy**：`./deploy.sh --input-type zmq_manager sim|real`
3. **PICO**：`pico_manager_thread_server.py --manager`
4. **Exporter**：

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the cup" \
    --camera-host 192.168.123.164 --camera-port 5555
```

5. **可选 Viewer**：

```bash
python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host 192.168.123.164 --camera-port 5555
```

---

## 7. 首次遥操流程

1. **标定姿势**：直立、脚并拢、目视前方；上臂贴身下垂；前臂前屈 90°（肘成 L）；掌心向内。
2. 同时按 **A+B+X+Y** → 启动 policy + `CALIB_FULL`。
3. 身体对齐机器人当前姿态后，按 **A+X** → 进入 **POSE**（全身 SMPL 遥操）。
4. 再按 **A+X** → 回到 **PLANNER**。
5. 再按 **A+B+X+Y** → 停机（OFF）。

**进入 POSE / VR_3PT 前必须先对齐身体**，否则会出现猛甩或失控。

---

## 8. 手柄速查

### 模式与标定

| 动作 | 按键 |
|---|---|
| 启动 / 急停 policy | **A+B+X+Y** |
| PLANNER ↔ POSE | **A+X** |
| POSE ↔ PLANNER_FROZEN_UPPER | **B+Y** |
| Planner ↔ VR_3PT（进时做手腕 CALIB） | **Left Stick Click** |
| 抓取 | 对应手 **Trigger** |

| 模式 | 含义 |
|---|---|
| OFF | 未跑 policy |
| POSE | 全身跟随你的 SMPL 姿态 |
| PLANNER | 下肢/全身由 planner；摇杆走位 |
| PLANNER_FROZEN_UPPER | planner 行走，上半身冻结 |
| VR_3PT | planner 行走，上半身跟头+双手 3 点追踪 |

### Planner 摇杆

| 输入 | 功能 |
|---|---|
| 左摇杆 | 平移（前后左右） |
| 右摇杆水平 | 航向 yaw |
| **A+B** | 下一个 locomotion 模式 |
| **X+Y** | 上一个 locomotion 模式 |

Locomotion 示例：0 Idle，1 Slow Walk，2 Walk，3 Run，4 Squat，…（完整列表见官方 teleop 教程）。

### 录制控制

| 动作 | 手柄 | 键盘 |
|---|---|---|
| 开始 / 结束一集（toggle） | **Left Grip + A** | `c` |
| 丢弃当前集 | **Left Grip + B** | `x` |

建议：先进入 POSE → 确认画面与跟踪正常 → **Left Grip + A** 开录 → 做完再按一次保存。

---

## 9. 数据流架构（简图）

```
Workstation                          Robot
┌─────────────┐  ┌──────────────┐   ┌──────────────┐
│ C++ deploy  │  │ pico_manager │   │ Camera server│
│ :5557       │  │ :5556 pose   │   │ :5555 JPEG   │
│ g1_debug    │  │              │   │              │
└──────┬──────┘  └──────┬───────┘   └──────┬───────┘
       └────────┬───────┴──────────────────┘
                ▼
        run_data_exporter.py
        → outputs/...  (parquet + mp4)
```

| 源 | 端口 | 内容 |
|---|---|---|
| C++ deploy | 5557 | 关节状态、IMU、`robot_config` |
| PICO | 5556 | SMPL teleop 目标 |
| Camera | 5555 | ego / wrist JPEG |

---

## 10. 录后处理

```bash
source .venv_data_collection/bin/activate

# 清理 discard episode + 陈旧 SMPL 帧
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/my_dataset \
    --output-path outputs/my_dataset_cleaned

# VR_3PT 录制时 SMPL 常为全零，必须关掉 SMPL 清洗
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/my_dataset \
    --output-path outputs/my_dataset_cleaned \
    --no-remove-stale-smpl

# 合并多次录制
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/s1 outputs/s2 outputs/s3 \
    --output-path outputs/merged_dataset
```

---

## 11. 最短路径 checklist

**从零配置 PICO**

1. 装 PC Service deb → 头显装 XRoboToolkit APK
2. 脚部 tracker 配对 + Calibrate
3. 同 Wi-Fi → App 填 PC IP → Status **WORKING** → Head/Controller + Send + Full body
4. `bash install_scripts/install_pico.sh`

**只遥操（仿真）**

1. `run_sim_loop.py`
2. `deploy.sh --input-type zmq_manager sim`
3. `pico_manager_thread_server.py --manager --vis_vr3pt --vis_smpl`
4. 标定姿势 → **A+B+X+Y** → 对齐 → **A+X**

**录数据（真机）**

1. 确认机器人 camera server 正常；工作站有 `.venv_data_collection`
2. `python gear_sonic/scripts/launch_data_collection.py --camera-host <IP> --task-prompt "..."`
3. **A+B+X+Y** → **A+X** → **Left Grip + A** 开录
4. 结束：再按 **Left Grip + A**；失败用 **Left Grip + B**
5. `process_dataset.py` 清洗

---

## 12. 常见问题

| 现象 | 排查 |
|---|---|
| Status 不是 WORKING | 同 Wi-Fi？PC Service 已装并运行？IP 填的是工作站而非头显？点 Reconnect |
| Tracker 配对不上 | 先全部 unpair；长按 6 秒到红蓝闪；Safeguard 已关；穿紧身裤露出发光面 |
| 无可视化窗口 / 无 pose | 第 2.4 节 XRoboToolkit 勾选；确认 `.venv_teleop`；PC Service 先于 App 启动 |
| 相机黑屏 / exporter 无图 | `camera-host`、端口、机器人 `composed_camera_server` |
| 进 POSE 猛甩 | 切换前未对齐身体；先回 PLANNER 再对齐 |
| VR_3PT 乱动 | 进模式前对齐手臂；Left Stick 退回 → 对齐 → **A+X** 回 POSE |
| 清洗后数据被清空 | 若用了 VR_3PT，加 `--no-remove-stale-smpl` |
| launcher 报缺依赖 | `bash install_scripts/install_data_collection.sh` |

更多排障见 [Troubleshooting](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/troubleshooting.html) 与 [Whole-body Teleoperation Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/teleoperation.html)。
