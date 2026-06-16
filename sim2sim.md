# sim2sim — unitree_g1_sonic_no_hand（29-DOF rubber-hand + suitcase）

embodiment-tag：`unitree_g1_sonic_no_hand`  
checkpoint 示例：`/home/ubuntu/DATA4/zzb/chekpoint_suitcase/checkpoint-5000`（只输出 64-dim `motion_token`，无手部 action）

MuJoCo 场景：`g1_29dof_rubberhand_suitcase_sim.xml`（由 `build_rubberhand_suitcase_scene.py` 生成）  
- 机器人本体与 git `scene_suitcase_43dof` 相同（Inspire DFQ 29-DOF 身体 + 关节阻尼）  
- 手：Inspire 手移除，换 rubber_hand 视觉 + eef_box 接触  
- 行李箱：训练用 `suitcase-simplified_training.xml`（非 git 的 `suitcase_scene.xml`）

```bash
cd GR00T-WholeBodyControl
python gear_sonic/scripts/build_rubberhand_suitcase_scene.py
```

---

## 数据流（全程无手部 I/O）

```
┌─────────────────┐     ZMQ REQ/REP      ┌──────────────────┐
│ PolicyServer    │◄────────────────────►│ run_vla_inference │
│ (Isaac-GR00T)   │  obs: video(ego+     │ embodiment:       │
│ checkpoint-5000 │      wrist)×state     │ unitree_g1_sonic_ │
│                 │      29-DOF body only │ no_hand           │
│ 输出: motion_   │  action: motion_token │ 不读 left/right_  │
│ token (64)      │  only (无 hand)       │ hand_q；不发 hand  │
└─────────────────┘                       └────────┬─────────┘
                                                   │ ZMQ PUB :5556
                                                   │ latent v4 (token only)
                                                   ▼
┌─────────────────┐     DDS lowcmd         ┌──────────────────┐
│ MuJoCo sim      │◄────────────────────►│ gear_sonic_deploy │
│ run_sim_loop    │  rt/lowstate          │ C++ WBC (sim)     │
│ wbc: nohand_    │  29 body motors       │ sim_hands 占位，   │
│ suitcase        │  NUM_HAND=0           │ 不接收 hand cmd   │
│ rubber-hand XML │                       │ ZMQ state :5557   │
└────────┬────────┘                       └──────────────────┘
         │ ZMQ camera :5555
         │ ego_view / left_wrist / right_wrist / global_view
         ▼
┌─────────────────┐
│ run_data_exporter│  录制视频（可选 --no-record-wrist-cameras）
└─────────────────┘
```

| 环节 | 手部相关 | no-hand 行为 |
|------|----------|--------------|
| VLA 训练/推理 | checkpoint 无 hand action | Policy 只返回 `motion_token` |
| `run_vla_inference` | `no_hand=True` | 观测无 `left_hand`/`right_hand`；ZMQ 只发 token |
| C++ deploy | `sim_hands` | sim 模式不驱动 Inspire；hand state 恒为 0 |
| MuJoCo | rubber_hand 仅为碰撞网格 | 无 L_/R_ 关节；`NUM_HAND_*=0` |
| DDS bridge | `GetAction` | `num_hand_motor==0` 时仅需 `lowcmd` |

初始化（与 git `sonic_model12` / `scene_suitcase_43dof` 一致）：
- 机器人：Inspire DFQ 身体 + 关节阻尼（`damping=0.05`）；手腕为 rubber_hand + eef_box（无 Inspire 手）
- 站姿：C++ deploy `InitControl` 在弹力带下 ramp（终端 3 出现 `Init Done`）
- 手腕初始角：`left_wrist_yaw=-0.4`，`right_wrist_yaw=0.4`（与 `move_suitcase_sonic.yaml` 一致）
- 行李箱：`suitcase-simplified_training.xml` 竖立放置（freejoint 在 motion.npz 的 suitcase link 原点；geom offset 使 20×30 cm 面贴地、高 40 cm；与 `sub1_suitcase_011` 第 0 帧 pelvis-yaw 相对位姿一致）

修改 deploy 默认关节角后需重新编译：
```bash
cd gear_sonic_deploy && source scripts/setup_env.sh && cmake --build build -j$(nproc)
```

---

## 一共开 6 个终端

### 1. PolicyServer

```bash
cd Isaac-GR00T
export CUDA_VISIBLE_DEVICES=0

uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /home/ubuntu/DATA4/zzb/chekpoint_suitcase/checkpoint-5000 \
    --embodiment-tag unitree_g1_sonic_no_hand \
    --device cuda:0 \
    --port 5550
```

### 2. MuJoCo sim（必须用 nohand_suitcase 配置）

`Channel factory init error` 是 DDS 重复初始化提示，可忽略。若之前有 deploy 残留，先 `pkill -f g1_deploy` 再启动。

```bash
cd GR00T-WholeBodyControl
source .venv_sim/bin/activate
export MUJOCO_GL=egl

python gear_sonic/scripts/run_sim_loop.py \
    --wbc-version nohand_suitcase \
    --no-with-hands \
    --enable-offscreen \
    --enable-image-publish \
    --no-enable-onscreen \
    --camera-port 5555
```

### 3. C++ Whole-Body Deploy（sim）

```bash
cd GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
printf '\n' | ./deploy.sh --input-type zmq_manager sim
```

### 4. 键盘控制

```bash
cd GR00T-WholeBodyControl
source .venv_inference/bin/activate
python gear_sonic/scripts/keyboard_publisher.py
```

看到 `Keyboard publisher ready` 后，**先确认终端 3 deploy 已 `Init Done`**，再依次输入：`k` → `i` → `c` → `p`（录视频时）

- `k`：释放弹力带 + 启动 C++ 控制环（PLANNER）
- `i`：发送初始站姿并切到 POSE 模式（`k` 后尽快按，避免自由落体）
- `c`：开始录制（data exporter；**须在 `p` 之前**，否则策略先跑会把 suitcase 碰倒）
- `p`：恢复/暂停 VLA 策略循环（`i` 后等 2～3 秒站稳，且 `c` 已开始录后再按）

注意：`run_vla_inference.py` 和 C++ deploy 也要在跑，按键才有效果。

### 5. VLA Inference

```bash
cd GR00T-WholeBodyControl
source .venv_inference/bin/activate

python gear_sonic/scripts/run_vla_inference.py \
    --host localhost --port 5550 \
    --embodiment-tag unitree_g1_sonic_no_hand \
    --prompt "Pick up the suitcase in front of you and move it." \
    --camera-host localhost --camera-port 5555 \
    --action-publish-rate 50 --action-horizon 40
```

### 6. 录视频

```bash
cd GR00T-WholeBodyControl
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "Pick up the suitcase in front of you and move it." \
  --camera-host localhost \
  --camera-port 5555 \
  --dataset-name vla_sim2sim_$(date +%Y%m%d_%H%M%S) \
  --root-output-dir /home/ubuntu/DATA4/zzb/HDMI/vla_sim_videos
```

默认录制 4 路：`ego_view`、`global_view`、`left_wrist`、`right_wrist`。  
手腕相机仅用于 VLA 视觉输入；若不需要录制手腕视角，加 `--no-record-wrist-cameras`。

一键启动（含自动按键与录制）：`python gear_sonic/scripts/launch_sim2sim_record.py`

---

## 推荐操作顺序

1. 终端 1：PolicyServer 出现 `Server ready — listening on ...:5550`
2. 终端 2：MuJoCo sim 稳定、相机 ZMQ 有帧
3. 终端 3：`deploy.sh sim` 运行至出现 **`Init Done`**（deploy 在弹力带下完成站姿 ramp）
4. 终端 4/5/6：键盘、VLA、录制就绪
5. 键盘（终端 4）：`k` → `i` → 等 2～3 秒站稳 → `c`（开始录）→ `p`（开策略）
6. 执行任务 … → `s`（结束录制）
7. 结束：`p` 暂停 → `k` 停控制环 → 各终端 Ctrl+C
8. 查看视频：`/home/ubuntu/DATA4/zzb/HDMI/vla_sim_videos/`

**不要**在录视频时用 `k → i → p → c`：`p` 会立刻发 VLA token，机器人乱动把 suitcase 碰倒后，`c` 才开始录，第一帧就是倒地的箱子。
