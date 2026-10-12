# 测试状态反馈闭环

`scripts/run_state_feedback.py` 先完成初始生成，再使用已冻结的服务器离线 harness 评测。默认修复模式为 `featurespace`：正式测试失败后采集状态，先更新 FeatureSpace，再生成下一轮代码，最多 4 轮修复。控制器可在本地或通过 `--same-host` 在服务器执行；评测容器使用已安装的依赖和 `network_mode=none`。初始召回、历史分析及 plan-edit 生成流程不变。

每次正式评测和诊断都用 manifest 中的不可变初始 image ID 新建独立容器，并检查 base commit 与初始工作区 diff 指纹。补丁只应用在本轮容器，结束后只删除本轮容器；初始镜像、保护容器和备份不删除。模型修复也使用临时源码树，导出的补丁始终相对于原 base，本地预备仓库不修改。

```sh
python scripts/run_state_feedback.py \
  --run-root /path/to/new/run \
  --inputs /path/to/sanitized/generation_inputs.json \
  --prepared-repos /path/to/verified/base/repos \
  --credentials-file /path/to/private/deepseek.json \
  --host 5022 --bundle /data2/ymz/feabench-offline-1007 \
  --workers 4 --repair-rounds 4
```

`--inputs` 只允许需求、声明组件、仓库、base 和 ID；不允许参考补丁或测试补丁。测试补丁由服务器冻结的评测脚本执行，反馈使用实际测试输出与当前候选轨迹。Conan adapter 自动识别对象/配置验证、包身份与模式状态关系；其它仓库使用通用对象身份、字段变化、异常、consumer guard 和调用轨迹。观察器有事件/字节上限，诊断失败或没有事件会记录局限，不能当成已观察到的状态。正式 harness 才决定 resolved。

每轮先选择真实测试输出和各类状态证据，诊断调用据此提出 FeatureSpace 增量更新。初始计划中的约束属于可撤回的假设。程序验证证据引用和代码范围后合并更新，修复调用使用更新后的 FeatureSpace、关联证据及当前源码生成 edits。诊断格式错误时仍尝试代码生成，保留已有记忆和本轮观测。代码经编译、重复方法/相同 AST/历史候选循环检查和 `git apply --check` 后才成为下一轮候选；预检失败保留上一个可应用候选。没有测试报告的 harness 错误单独报告，不虚构状态；测试文件修改被拒绝。

评测日志完整保存在归档中，控制器按测试阶段和失败项选择摘要。状态证据按类别和测试分配空间，记录未展示数量。正式测试结果决定 resolved；诊断轨迹仅解释失败。下一轮记录已修复测试、新回归、仍失败和未观测到的测试，不把“没有再执行”误判为“已通过”。

输出包括 `progress.json`、每例初始生成、逐轮预测/正式报告/状态轨迹、`featurespaces/<id>.json`、`repairs/<id>/<round>/` 中的诊断与修复 prompt/响应、FeatureSpace 前后快照和增量、usage、测试变化以及补丁。最终输出为 `predictions_final.jsonl`，全量运行另保留 `predictions_final200.jsonl`。`--resume` 复用已完成阶段和模型响应，源码哈希不符时拒绝恢复。远端评测进程仍可接管。

此流程是有测试反馈、固定最大修复预算的单候选 workflow。报告要同时披露规划/代码/结构修复/语义修复调用、token 与测试次数，不称为一次裸 LLM 成功。Conan-16871 是开发 case，全量 200 例和其它 199 例分别统计。参考补丁和参考执行轨迹均不进入模型输入。

冻结原初始候选、单例验证和消融模式见 [FeatureSpace self-evolution](featurespace_evolution.md)。

## 全部在服务器执行

服务器模式使用 `--same-host`，生成、状态提取、修复和调度都在服务器执行，无需 SSH 到自身，也不依赖本地电脑。使用 `--proxy http://127.0.0.1:10805` 显式设置模型调用代理，避免非交互后台进程丢失 shell 配置。正式评测和诊断进程移除代理与 API key 环境变量，保留离线网络隔离和初始镜像检查。

`scripts/prepare_server_generation.py` 从已有公开 Git 缓存创建每例独立 base worktree，并从已有服务器配置读取凭据写入任务专用 0600 私密文件。迁移时保留原初始生成结果、源码版本记录和当前评测 PID。`--resume --same-host` 会接管正在执行的本轮评测，复用已完成结果，继续后续自动反馈修复。
