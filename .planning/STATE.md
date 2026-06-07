# 项目状态

## 当前阶段
- [ ] 初始化完成
- [ ] Phase 1: Flow Matching 模型实现
- [ ] Phase 2: Stage 2 训练流程
- [ ] Phase 3: 推理替换 & PMLB 批量测试

## 关键决策记录
- 2026-06-07: 决定采用 ColaDLM 的 Flow Matching 范式替换 CMA-ES
- 保持 CVAE 预训练不变，新增 Flow Matching 作为 Stage 2
- Flow Matching 条件输入为 CVAE 的 prior_mu（数据嵌入）
