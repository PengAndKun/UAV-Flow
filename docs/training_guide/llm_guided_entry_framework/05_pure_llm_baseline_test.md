# Pure LLM Baseline Test

## 1. 目的

这一步对应 `Step C / C-1：standalone 验证`。

目标不是训练策略，也不是在线控飞，而是先回答一个最基础的问题：

- **只给 LLM / VLM 看 `RGB + depth preview`，它能不能对入口场景做出基本正确的判断？**

这里的 `Pure LLM baseline` 只作为：

- 一个对照实验
- 一个能力上限 / 下限参考
- 后续 `Fusion + LLM guidance` 的比较基线

它**不是最终部署方案**。

---

## 2. 当前应该测什么

第一轮 `Pure LLM baseline` 先只看下面 5 件事：

1. `scene_state`
   - `outside_house / threshold_zone / inside_house / unknown`

2. `active_stage`
   - 例如 `entry_search / approach_entry / cross_entry`

3. `entry_door_visible`

4. `entry_door_traversable`

5. `scene_description`

也就是说，这一轮先验证：

- 它能不能看懂当前画面
- 它能不能区分“看到了门”和“门是否能进去”
- 它会不会把窗户误判成门
- 它会不会在障碍重的时候乱给乐观判断

这一轮**先不要求**输出最终 `Step C teacher schema` 的：

- `subgoal`
- `action_hint`
- `waypoint_hint`

这些放到下一轮 `Fusion + LLM guidance` 里再测试更稳。

---

## 3. 输入样本用什么

推荐直接使用现有 `Phase2 Fusion` 结果目录中的 `labeling/` 样本包。

每个样本至少会有：

- `rgb.png`
- `depth_preview.png`
- `labeling_summary.txt`
- `fusion_result.json`

其中 `Pure LLM baseline` 实际喂给模型的图像只需要：

- `rgb.png`
- `depth_preview.png`

注意：

- 这里优先用 `depth_preview.png`
- 不直接把 `fusion_result.json` 喂给模型
- 因为这一组实验的定义就是：**纯靠 LLM/VLM 看图判断**

---

## 4. 推荐样本选择

第一轮先不要跑太多，建议：

- `20 ~ 30` 个样本

优先覆盖 4 类：

1. `enterable_open_door`
2. `front_blocked_detour`
3. `window_visible_keep_search`
4. `no_entry_confirmed`

如果样本够，再补：

5. `visible_but_blocked_entry`
6. `geometric_opening_needs_confirmation`
7. `enterable_door`

这样第一轮最容易看出：

- LLM 会不会把窗户当门
- 会不会忽视前障碍
- 会不会在证据不足时胡乱给乐观判断

---

## 5. 推荐脚本入口

当前仓库里已经有两个 standalone 入口：

- [anthropic_vlm_scene_descriptor.py](/E:/github/UAV-Flow/UAV-Flow-Eval/anthropic_vlm_scene_descriptor.py)
- [vlm_scene_descriptor.py](/E:/github/UAV-Flow/UAV-Flow-Eval/vlm_scene_descriptor.py)

推荐优先使用：

- `anthropic_vlm_scene_descriptor.py`

适合模型：

- `claude-sonnet-4-6`
- `qwen3-coder-next`

如果你要测试 Gemini，则使用：

- `vlm_scene_descriptor.py`

---

## 6. 实际操作步骤

### Step 1：准备一个样本

假设你要测试的样本目录是：

```text
E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\results\fusion_xxx\labeling
```

确认里面有：

- `rgb.png`
- `depth_preview.png`

### Step 2：设置 API 环境变量

以 Anthropic 路线为例：

```powershell
$env:ANTHROPIC_BASE_URL="http://你的base_url"
$env:ANTHROPIC_AUTH_TOKEN="你的token"
```

### Step 3：运行 standalone baseline

以 `claude-sonnet-4-6` 为例：

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\anthropic_vlm_scene_descriptor.py `
  --model claude-sonnet-4-6 `
  --task_label "search the house for people" `
  --rgb_path E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\results\fusion_xxx\labeling\rgb.png `
  --depth_path E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\results\fusion_xxx\labeling\depth_preview.png `
  --output_json E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\results\fusion_xxx\labeling\pure_llm_baseline_result.json
```

说明：

- `--depth_path` 这里故意使用 `depth_preview.png`
- 因为这个 baseline 的定义是“人可读多模态图像判断”

### Step 4：查看输出

重点看：

- `parsed.scene_state`
- `parsed.active_stage`
- `parsed.entry_door_visible`
- `parsed.entry_door_traversable`
- `parsed.scene_description`
- `latency_ms`

同时记录：

- 它是不是把窗户当门
- 它是不是忽略了前障碍
- 它是不是明显比人工判断更乐观或更保守

---

## 7. 推荐保存方式

建议每个样本目录下新增：

- `pure_llm_baseline_result.json`

如果要比较多个模型，可以改成：

- `pure_llm_claude_result.json`
- `pure_llm_qwen_result.json`
- `pure_llm_gemini_result.json`

这样后面非常方便和：

- `fusion_result.json`
- 人工标注结果

做对比。

---

## 8. 推荐人工记录项

建议你对每个样本额外记一张简表，至少包含：

```json
{
  "sample_id": "",
  "model_name": "",
  "gt_entry_state": "",
  "llm_scene_state": "",
  "llm_active_stage": "",
  "llm_entry_door_visible": false,
  "llm_entry_door_traversable": false,
  "judgement_ok": false,
  "error_type": "",
  "notes": ""
}
```

其中 `error_type` 推荐只先记这几类：

- `window_as_door`
- `blocked_but_says_enterable`
- `missed_entry`
- `uncertain_but_overconfident`
- `other`

---

## 9. 第一轮怎么判定“有效”

第一轮不用太严格，先看下面 4 条：

1. 大多数 `open_door` 正样本里，能识别：
   - `entry_door_visible = true`

2. 大多数 `window` 样本里，不会误判成明显可进入入口

3. 大多数前障碍很强的样本里，不会给出明显激进的进入判断

4. 同类样本输出比较稳定，不会一会儿 `approach_entry` 一会儿 `inside_house`

如果这 4 条大体成立，就说明：

- `Pure LLM baseline` 是可跑通的
- 下一步值得进入 `Fusion + LLM guidance`

---

## 10. 这一轮完成后下一步做什么

当你跑完 `20 ~ 30` 个样本后，下一步再进入：

- `Fusion + LLM guidance`

那一轮再把输入扩展成：

- RGB
- depth preview
- YOLO 检测摘要
- 深度分析摘要
- 融合判断结果
- 当前位姿 / 历史

并把输出固定成：

- `subgoal`
- `action_hint`
- `waypoint_hint`
- `risk_level`
- `reason`
- `confidence`

---

## 11. 一句话执行建议

先不要训练，先做这件事：

- 从现有 `Phase2 Fusion labeling` 样本包里挑 `20 ~ 30` 个样本
- 用 standalone descriptor 跑 `Pure LLM baseline`
- 把输出和人工真值做一轮对比

只要这一步跑通，你后面再做 `Fusion + LLM guidance` 和 teacher 数据导出就会稳很多。
