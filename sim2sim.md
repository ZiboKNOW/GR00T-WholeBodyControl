# sim2sim 一共开6个终端

## 载入刚刚保存的ckpt，开PolicyServer

cd Isaac-GR00T
export CUDA_VISIBLE_DEVICES=0

uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /home/ubuntu/DATA4/zzb/groot_ckpt/wid_6/checkpoint-80000 \
    --embodiment-tag unitree_g1_sonic_inspire \
    --device cuda:0 \
    --port 5550

## 开mujoco
cd GR00T-WholeBodyControl
#source /home/ubuntu/DATA2/zzb/Isaac-GR00T/.venv/bin/activate
source .venv_sim/bin/activate
export MUJOCO_GL=egl

python gear_sonic/scripts/run_sim_loop.py \
    --enable-offscreen \
    --enable-image-publish \
    --no-enable-onscreen \
    --camera-port 5555

## C++ Whole-Body Deploy（sim）
cd GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
printf '\n' | ./deploy.sh --input-type zmq_manager sim

## 键盘控制
cd GR00T-WholeBodyControl
source .venv_inference/bin/activate
python gear_sonic/scripts/keyboard_publisher.py
  看到 Keyboard publisher ready 后，在同一终端依次输入：
  k i p

  注意：run_vla_inference.py 和 C++ deploy 也要在跑，按键才有效果；只开键盘脚本、其它服务没起来时，只会显示 Sent: k 但机器人不会动。

## VLA Inference
cd GR00T-WholeBodyControl
source .venv_inference/bin/activate

python gear_sonic/scripts/run_vla_inference.py \
  --host localhost \
  --port 5550 \
  --embodiment-tag unitree_g1_sonic_inspire \
  --prompt "Lift the shelf and walk a few steps forward, then put it down" \
  --camera-host localhost \
  --camera-port 5555 \
  --action-publish-rate 100 \
  --action-horizon 40

## 录视频
cd GR00T-WholeBodyControl
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "Lift the shelf and walk a few steps forward, then put it down" \
  --camera-host localhost \
  --camera-port 5555 \
  --dataset-name vla_sim2sim_$(date +%Y%m%d_%H%M%S)

  默认录制 4 路视频：ego_view、global_view（上帝视角）、left_wrist、right_wrist。
  若不需要手腕相机，加 --no-record-wrist-cameras。

  - 推荐操作顺序
    1. 终端 1：PolicyServer 出现 Server ready — listening on ...:5550
    2. 终端 2：MuJoCo sim 稳定、相机 ZMQ 有帧
    3. 终端 3：deploy.sh sim 确认并运行
    4. 终端 4/5/6：键盘、VLA、录制 就绪
    5. 键盘（terminal 4）：k → i → p（开始策略）
    6. 录制：c … 执行任务 … s（data exporter）或跑 headless 录制脚本
    7. 结束：p 暂停 → k 停控制环 → 各终端 Ctrl+C
    8. 查看视频 outputs/vla_sim_videos/（工作区根目录下）