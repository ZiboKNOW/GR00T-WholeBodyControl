# sim2sim — unitree_g1_sonic_no_hand_wo_wrist（29-DOF rubber-hand + suitcase）

embodiment-tag：`unitree_g1_sonic_no_hand_wo_wrist`（与 Isaac `render_vla` 一致，**仅 ego_view**）  
checkpoint：`/home/ubuntu/DATA4/zzb/chekpoint_suitcase_final/checkpoint-25000`（只输出 64-dim `motion_token`，无手部 action）

MuJoCo 场景：`g1_29dof_rubberhand_suitcase_sim.xml`（由 `build_rubberhand_suitcase_scene.py` 生成）  
- 机器人本体与 git `scene_suitcase_43dof` 相同（Inspire DFQ 29-DOF 身体 + 关节阻尼）  
- 手：Inspire 手移除，换 rubber_hand 视觉 + eef_box 接触  
- 行李箱：训练用 `suitcase-simplified_training.xml`（非 git 的 `suitcase_scene.xml`）

```bash
cd GR00T-WholeBodyControl
python gear_sonic/scripts/build_rubberhand_suitcase_scene.py
```

---

## 数据流（全程无手部 I/O，VLA 仅 ego_view）

```
┌─────────────────┐     ZMQ REQ/REP      ┌──────────────────┐
│ PolicyServer    │◄────────────────────►│ run_vla_inference │
│ (Isaac-GR00T)   │  obs: ego_view×state │ embodiment:       │
│ checkpoint-25000│  29-DOF body groups  │ unitree_g1_sonic_ │
│                 │  (robot_model 映射)  │ no_hand_wo_wrist  │
│ 输出: motion_   │  action: motion_token│ 不读 left/right_  │
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
│ --ego-view-only │                       │ ZMQ state :5557   │
└────────┬────────┘                       └──────────────────┘
         │ ZMQ camera :5555
         │ ego_view (640×480 head_camera)
         ▼
┌─────────────────┐
│ run_data_exporter│  录制视频（可选 global / wrist）
└─────────────────┘
```

| 环节 | 手部相关 | no-hand wo-wrist 行为 |
|------|----------|------------------------|
| VLA 训练/推理 | checkpoint 无 hand action | Policy 只返回 `motion_token`；视频仅 `ego_view` |
| `run_vla_inference` | `no_hand=True` | `body_q` → `robot_model` 分组 state；无 wrist 相机要求 |
| C++ deploy | `sim_hands` | sim 模式不驱动 Inspire；hand state 恒为 0 |
| MuJoCo | rubber_hand 仅为碰撞网格 | 无 L_/R_ 关节；`NUM_HAND_*=0` |
| DDS bridge | `GetAction` | `num_hand_motor==0` 时仅需 `lowcmd` |

与 Isaac `render_vla` 对齐要点：
- embodiment：`unitree_g1_sonic_no_hand_wo_wrist`
- 相机：仅 `ego_view`（640×480）；`run_sim_loop.py --ego-view-only`
- 状态：`get_configuration_from_actuated_joints` + `get_joint_group_indices`（与 `run_data_exporter` 一致）

初始化（向 Isaac `render_vla` 的 `robot_init_override` 对齐）：
- 机器人：Inspire DFQ 身体 + 关节阻尼（`damping=0.05`）；手腕为 rubber_hand + eef_box（无 Inspire 手）
- MuJoCo reset：`g1_29dof_sonic_nohand_suitcase.yaml` 的 `DEFAULT_DOF_ANGLES` / `DEFAULT_MOTOR_ANGLES` 使用 `/home/ubuntu/DATA4/zzb/HDMI/data/motion/initial_pose/analysis/non_hand_state_29dof.npz` 的 498 帧均值
- C++ deploy：`InitControl` 在弹力带下 ramp 到同一套 render 初始姿态（终端 3 出现 `Init Done`）；`default_angles` 对齐 render 的 `sonic_g1_model_12` low-level policy 基准（wrist yaw 基准为 0，不是初始姿态）
- 右臂关键初始角：`right_shoulder_roll≈-0.261`，`right_shoulder_yaw≈0.643`，`right_elbow≈1.119`，`right_wrist_yaw≈-0.251`
- 行李箱：`suitcase-simplified_training.xml` 竖立放置（freejoint 在 motion.npz 的 suitcase link 原点；geom offset 使 20×30 cm 面贴地、高 40 cm；与 `sub1_suitcase_011` 第 0 帧 pelvis-yaw 相对位姿一致）
- 可选 40 cm 立方体：`run_sim_loop.py --object-type cube40` 加载 `cube-40cm_training.xml`，初始相对位姿对齐 `walk_with_suitcase_g1_arm_aligned/motion.npz` 第 0 帧（pelvis-yaw 水平偏移约 `[1.035, -0.161]` m）

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
export CUDA_VISIBLE_DEVICES=10
export HF_HOME=/home/ubuntu/DATA4/zzb/hf_cache
export HF_HUB_CACHE=/home/ubuntu/DATA4/zzb/hf_cache/hub
export GROOT_HF_LOCAL_FIRST=1
export GROOT_PATCH_MISTRAL=1
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /home/ubuntu/DATA4/zzb/checkpoint_suitcase_slow_1_5/checkpoint-50000 \
    --embodiment-tag unitree_g1_sonic_no_hand_wo_wrist \
    --device cuda:0 \
    --port 5550
```

### 2. MuJoCo sim（必须用 nohand_suitcase 配置 + ego_view only）

`Channel factory init error` 是 DDS 重复初始化提示，可忽略。若之前有 deploy 残留，先 `pkill -f g1_deploy` 再启动。

默认使用行李箱（`suitcase-simplified_training.xml`）。若要把物体换成 **40 cm 立方体**（初始相对位姿对齐 `walk_with_suitcase_g1_arm_aligned/motion.npz` 第 0 帧），在下方命令加 `--object-type cube40`；其余终端与 prompt 不变。

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
    --ego-view-only \
    --camera-port 5555
```

40 cm 立方体示例（仅终端 2 多一个参数）：

```bash
python gear_sonic/scripts/run_sim_loop.py \
    --wbc-version nohand_suitcase \
    --no-with-hands \
    --enable-offscreen \
    --enable-image-publish \
    --no-enable-onscreen \
    --ego-view-only \
    --camera-port 5555 \
    --object-type cube40
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

注意：`run_vla_inference.py` 和 C++ deploy 也要在跑，按键才有效果。i之后等待3秒再c。

### 5. VLA Inference

```bash
cd GR00T-WholeBodyControl
source .venv_inference/bin/activate

python gear_sonic/scripts/run_vla_inference.py \
    --host localhost --port 5550 \
    --embodiment-tag unitree_g1_sonic_no_hand_wo_wrist \
    --prompt "Pick up the suitcase in front of you and move it." \
    --camera-host localhost --camera-port 5555 \
    --action-publish-rate 50 --action-horizon 40 \
    --inference-rate-hz 2.5 \
    --async-timing-mode sim
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
  --root-output-dir /home/ubuntu/DATA4/zzb/HDMI/vla_sim_videos \
  --no-record-wrist-cameras
```

默认 VLA 仅用 `ego_view`。若 MuJoCo 未加 `--ego-view-only`，可加 `--record-wrist-cameras` 额外录手腕视角（**不会**送入 VLA）。

一键启动（含自动按键与录制）：`python gear_sonic/scripts/launch_sim2sim_record.py`

---

## 推荐操作顺序

1. 终端 1：PolicyServer 出现 `Server ready — listening on ...:5550`
2. 终端 2：MuJoCo sim 稳定、相机 ZMQ 有 `ego_view` 帧
3. 终端 3：`deploy.sh sim` 运行至出现 **`Init Done`**（deploy 在弹力带下完成站姿 ramp）
4. 终端 4/5/6：键盘、VLA、录制就绪
5. 键盘（终端 4）：`k` → `i` → 等 2～3 秒站稳 → `c`（开始录）→ `p`（开策略）
6. 执行任务 … → `s`（结束录制）
7. 结束：`p` 暂停 → `k` 停控制环 → 各终端 Ctrl+C
8. 查看视频：`/home/ubuntu/DATA4/zzb/HDMI/vla_sim_videos/`

**不要**在录视频时用 `k → i → p → c`：`p` 会立刻发 VLA token，机器人乱动把 suitcase 碰倒后，`c` 才开始录，第一帧就是倒地的箱子。
