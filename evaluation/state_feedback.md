# 测试状态反馈闭环

`scripts/run_state_feedback.py` 在本地生成全量 200 个任务，全部第一轮生成结束后，上传补丁到已冻结的服务器离线 harness。正式测试失败的任务自动采集状态并生成下一轮修复，最多 4 轮。模型调用仅在本地，服务器使用已安装的依赖和 `network_mode=none`，不下载镜像、不调用模型。

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

生成 prompt 增加 producer/configuration/consumer 的通用约束。每轮修复 prompt 自动包含失败断言、状态 packet、当前候选源码、已有修复的预检错误。模型返回结构化 edits，经编译和 `git apply --check` 后才成为下一轮候选；失败保留上一个可应用候选，并记录原因。未取得测试报告的 harness 错误单独报告，不当作代码失败虚构状态。测试文件修改被拒绝。

输出包括 `progress.json`、每例初始生成、逐轮预测/正式报告/状态轨迹、`repairs/<id>/<round>/` 的 prompt/响应/usage/增量和合并补丁、最终 `predictions_final200.jsonl` 和统计。`--resume` 复用已完成工作，冻结源码哈希不符时拒绝恢复。远端评测进程也挂后台，本地中断后可恢复下载结果。

此流程是有测试反馈、固定最大修复预算的单候选 workflow。报告要同时披露规划/代码/结构修复/语义修复调用、token 与测试次数，不称为一次裸 LLM 成功。Conan-16871 是开发 case，全量 200 例和其它 199 例分别统计。参考补丁和参考执行轨迹均不进入模型输入。
