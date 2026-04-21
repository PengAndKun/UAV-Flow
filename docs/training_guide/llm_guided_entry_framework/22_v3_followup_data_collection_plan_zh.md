# V3 后续补采计划（中文）

## 文档目标

这份文档把 `pilot_distill_v3_reviewed` 的评估结果转成一份可以直接执行的中文补采清单。

这一步的目标不是“继续泛泛地多采一些数据”，而是：

- 针对 `v3` 暴露出来的薄弱类别，
- 有计划地补 target-conditioned 样本，
- 为下一轮 `v4` 训练提供更稳的监督。

## V3 当前结论

`v3` 说明一件很重要的事：

- 人工审核后的 target house 标签是有效的；
- 模型在 `target_conditioned_state` 和 `target_conditioned_subgoal` 上明显更稳了；
- 但在更细粒度的动作决策上仍然不够稳。

也就是说，模型现在更擅长判断：

- 当前看到的是不是目标房屋的入口；
- 是不是应该忽略非目标房屋入口；
- 目标入口是被挡住了，还是只是还没完全确认。

但它还不够擅长判断：

- 现在到底该不该直接前进；
- 左绕还是右绕；
- 是否应该短暂停住确认；
- 目标入口已经“可接近”还是仅仅“可见”。

## 当前不需要优先补的类

下一轮不要优先继续补这些类：

- `target_house_not_in_view`
- `ignore_non_target_entry`
- `reorient_to_target_house`
- `yaw_left`
- `yaw_right`

原因很简单：

- 这些类在当前导出数据里已经很多；
- 再继续补，只会让数据分布更偏；
- 而不会明显提升我们现在真正薄弱的能力。

## 最该优先补的类别

### 第一优先级：`target_house_entry_approachable`

这是目前最缺、也最关键的 target-conditioned 状态。

当前问题：

- 训练里这个类几乎没有；
- 模型还不能稳定区分：
  - “目标房屋入口已经可以靠近”
  - 和
  - “目标房屋入口只是可见，但还没到该前进的时候”

建议新增样本：

- `20 ~ 30` 条

建议场景条件：

- 目标房屋就是当前正确目标；
- 目标入口清楚可见；
- 前方通道没有明显大障碍；
- 门还没有到“立刻穿门”的距离；
- 门最好在图像中心附近，或者只轻微偏左/偏右。

希望得到的标签：

- `target_conditioned_state = target_house_entry_approachable`
- `target_conditioned_subgoal = approach_target_entry`
- `target_conditioned_action_hint = forward`

### 第二优先级：`approach_target_entry`

这是当前最重要、但支持数还太小的 target-conditioned 子任务。

当前问题：

- 模型容易把它和下面几类搞混：
  - `detour_right_to_target_entry`
  - `ignore_non_target_entry`
  - 或者其他非目标入口行为

建议新增样本：

- `+25`

建议场景条件：

- 目标房屋已经判断正确；
- 当前门属于目标房屋；
- 通道前方没有明显强障碍；
- 这时候正确动作应当是“往前靠近”，而不是原地转头。

希望得到的标签：

- `target_conditioned_subgoal = approach_target_entry`
- `target_conditioned_action_hint = forward`

### 第三优先级：`detour_left_to_target_entry`

当前左右绕行仍然不平衡。

现在的问题：

- 左绕样本偏少；
- 模型对左右绕行的区分还不稳定；
- 容易更偏向某一侧模式。

建议新增样本：

- 左绕：`+15`
- 右绕：`+8`（作为平衡检查）

建议场景条件：

- 当前入口是目标房屋入口；
- 前方确实有障碍；
- 障碍物要足够明确，让“左绕更合理”或“右绕更合理”非常清楚。

希望得到的标签：

- `target_conditioned_state = target_house_entry_blocked`
- `target_conditioned_subgoal = detour_left_to_target_entry` 或 `detour_right_to_target_entry`
- `target_conditioned_action_hint = left` 或 `right`

### 第四优先级：`hold`

当前模型还不会近距离稳定确认。

当前问题：

- 几乎没有“该停一下看清楚”的样本；
- 所以模型容易继续左右动，而不是短暂停住确认。

建议新增样本：

- `8 ~ 12`

建议场景条件：

- 目标入口已经比较近；
- 当前已经确认是目标房屋；
- 暂时不需要继续大动作；
- 更合理的动作是先稳一下再判断下一步。

希望得到的标签：

- `target_conditioned_action_hint = hold`

## 推荐采集模板

### 模板 A：目标入口可接近

用来采：

- `target_house_entry_approachable`
- `approach_target_entry`

采集设置：

- 明确指定一个目标房屋；
- 让 UAV 面向正确入口；
- 门清楚可见；
- 通道前方不要有大障碍。

建议距离分层：

- 远：`6m ~ 8m`
- 中：`4m ~ 6m`
- 近：`2.5m ~ 4m`

每个距离层次都建议采：

- 门居中
- 门偏左
- 门偏右

理想动作：

- `forward`

不要混入这些情况：

- 其他房屋的门占主导；
- 前方有很大遮挡导致其实该绕行；
- 当前正确动作其实是转头而不是前进。

### 模板 B：目标立面继续搜索

用来采：

- `target_house_entry_visible`
- `keep_search_target_house`

采集设置：

- 目标房屋已经在画面里；
- 但还不能稳定确认当前候选就是目标入口；
- 画面里可能有：
  - 立面、
  - 柱子、
  - 窗户、
  - 门廊边缘、
  - 侧墙。

建议采集动作：

- 缓慢横向观察；
- 缓慢偏航；
- 不要直接前冲。

理想动作：

- `yaw_left`
- `yaw_right`

不要混入这些情况：

- 其他房屋门特别显眼；
- 其实已经明显该前进的场景。

### 模板 C：目标入口绕行

用来采：

- `detour_left_to_target_entry`
- `detour_right_to_target_entry`

采集设置：

- 当前入口明确属于目标房屋；
- 前方有真实障碍；
- 障碍位置足够清楚，能形成明确的左右绕行偏好。

适合的遮挡物：

- 门廊柱子
- 灌木
- 栏杆
- 路灯
- 门前家具

最佳实践：

- 做成左右成对样本；
- 同一扇门；
- 同一距离；
- 只改变障碍位置或者 UAV 偏置方向。

这样模型最容易学清楚 left/right 差异。

### 模板 D：近距离稳住确认

用来采：

- `hold`

采集设置：

- 已经接近目标入口；
- 已确认目标房屋；
- 现在不适合继续大动作；
- 更合理的是先稳住，等待下一步确认。

理想动作：

- `hold`

这类样本后面也会对穿门前的短暂停顿很有帮助。

## 采集时必须遵守的规则

### 规则 1：一条样本只突出一个主决策

不要把下面这些混在同一条样本里：

- 重新找目标房屋；
- 目标立面搜索；
- 绕行；
- 直接前进靠近。

如果一条样本里主决策不清楚，标签就会变脏。

### 规则 2：不要存太多几乎一样的连续帧

不要为了凑数量保存很多几乎完全一样的图片。

更好的做法：

- 改一点点距离；
- 改一点点左右偏置；
- 改一点点角度；
- 改一点点障碍位置。

### 规则 3：先确认 target house 再采

每条 episode 开始前先确认：

- 当前 `target_house_id`
- 计划采的是哪一栋房子的入口
- 当前标签确实对应目标房屋

这一步现在尤其重要，因为 target house review 已经进入训练链了。

### 规则 4：左右绕行要平衡

采 detour 时，不要只采一侧。

尽量做到：

- 左绕一组
- 右绕一组

### 规则 5：严格区分“可见”和“可接近”

这是当前最容易混的边界之一。

如果现在更像：

- 目标房屋已经在画面里，
- 但还没到可以明确前进的时候，

那应标：

- `target_house_entry_visible`

如果现在已经明确：

- 门就是目标入口，
- 前方通道可行，
- 应该往前靠近，

那应标：

- `target_house_entry_approachable`

## 建议的下一轮补采目标

如果只做一轮高性价比补采，建议直接冲这组数：

- `target_house_entry_approachable / approach_target_entry / forward`：`+25`
- `target_house_entry_visible / keep_search_target_house`：`+15`
- `detour_left_to_target_entry / left`：`+15`
- `detour_right_to_target_entry / right`：`+8`
- `hold`：`+10`

这组量足够支撑一次 `v4`，而不用试图一次把全数据重新平衡完。

## 补采后怎么做

补采完成后，继续按现有流程走一遍：

1. 如有需要，先刷新 fusion 结果；
2. 如有需要，继续做 target house review；
3. 跑 teacher validator；
4. 跑 entry state builder；
5. 跑 dataset export；
6. 再和 `v3` 对比新的 target-conditioned 分布。

下一轮目标不是“单纯更多数据”，而是：

- 让 target-conditioned 的 approach / detour / hold 这些关键行为真正学起来。
