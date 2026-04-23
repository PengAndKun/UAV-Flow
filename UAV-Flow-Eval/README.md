# UAV-Flow-Eval

UAV-Flow-Eval is a simulation environment for UAV evaluation tasks, built on top of [UnrealZoo Gym](https://github.com/UnrealZoo/unrealzoo-gym).


### 1. Set Up the Environment
Create a conda environment, then install this repository in editable mode:
```bash
conda create -n unrealcv python=3.11
conda activate unrealcv
pip install -e .
```

### 2. Download the Simulation Environment
We tested on Windows using the packaged UnrealZoo environment:
[Collection_WinNoEditor_0424_25.zip](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE4/file/view/master/Collection_WinNoEditor_0424_25.zip?id=77779&status=2). Download and extract the archive to a local directory.

### 3. Configure the Environment Path
We use the DowntownWest as the test campus environment.
You need to update the configuration file:

/gym_unrealcv/envs/setting/Track/DowntownWest.json

Change the env_bin_win field to the actual path of your extracted simulation environment.

### 4. Run Evaluation

Set up your model on the server side and expose it through a specific port.
Then run:

```bash
python batch_run_act_all.py
```

You can modify arguments either in the batch_run_act_all.py or directly via the command line.
For example, to change the inference server port to 5006:

```bash
python batch_run_act_all.py --server_port 5006
```
After inference finishes, compute the NDTW metric by running:
```bash
python metric.py
```

### 5. Memory-Aware Collection in Basic Controller

The basic control server/panel now supports a lightweight memory-aware collection loop for entry-search data.

Server:
- `uav_control_server_basic.py`
- new arguments:
  - `--memory_collection_root`
  - `--entry_search_memory_path`

Panel:
- `uav_control_panel_basic.py`
- new `Memory Collection` section:
  - `Start Episode`
  - `Stop Episode`
  - `Reset Memory`
  - `Snapshot Now`
  - `Open Memory Window`

What gets collected when an episode is active:
- `episode_id`
- `step_index`
- shared `entry_search_memory.json`
- per-capture:
  - `*_entry_search_memory_snapshot_before.json`
  - `*_entry_search_memory_snapshot_after.json`

Additional integration:
- `Phase1 Scan x12` now records scan-level memory start/end snapshots and uses episode-relative step indices.
- `Execute Sequence` now saves quiet sequence boundary snapshots:
  - `sequence_start`
  - `sequence_end`
  - `sequence_stop`
  - `sequence_failed`

The memory snapshots are written alongside capture metadata so later dataset export can recover:
- the state before capture
- the state after capture
- the active target/current house context

Typical usage:
```bash
python uav_control_server_basic.py --capture_dir ./captures_remote
python uav_control_panel_basic.py --host 127.0.0.1 --port 5020
```

Recommended workflow:
1. Select target house and task.
2. Start a memory collection episode from the panel.
3. Move/capture as usual.
4. Stop the episode after finishing the sequence.
