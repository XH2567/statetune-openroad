# StateTune 的 OpenROAD 复现说明

> 原论文的实验跑在 Cadence Genus + Innovus 上。我们没有 Cadence license，所以把 EDA 后端换成了开源的 **OpenROAD-flow-scripts（ORFS：Yosys + OpenROAD）**，并在论文的 6 个 benchmark（NanGate45 / ASAP7 × AES / JPEG / IBEX）上完整跑通了 StateTune。
>
> **结论**：整个闭环在 OpenROAD 上能稳定运行，包括 LLM 生成候选、RAG、知识记忆、EHVI 晋升和多保真度调度。6 个 case 跑出的 WNS、面积、功耗都优于 ORFS 默认参数。但是 **绝对数值不能和论文直接对比**，原因见[第 5 节](#5-与论文结果的差距及原因)，最主要的两点是：ASAP7 的工艺角不同（ORFS 默认 FF，论文用 TT），以及功耗估算时的翻转率假设不同。

---

## 1. 后端是怎么从 Cadence 换成 OpenROAD 的

StateTune 本身不做综合和布局布线。它是架在 EDA 流程外面的调参器：生成一组参数，让 EDA 流程跑一遍，读回 PPA，再更新记忆、生成下一组参数。所以换后端只需要替换 **"参数 → 跑流程 → 读指标"** 这一层，调参算法不用动。

| | 论文（Cadence） | 本复现（OpenROAD） |
|---|---|---|
| 综合 | Genus 21.17 | Yosys 0.68（ORFS 内置） |
| 布局布线 | Innovus 21.18 | OpenROAD（ORFS `26Q4-180-g5110a7c29`） |
| 运行方式 | SSH 远程 + LSF 提交 | 本机运行：解压 ORFS 官方 docker 镜像，用 bubblewrap 当容器，**不需要 root 和 docker 权限** |
| Runner | `runners/cadence_runner.py` | `runners/openroad_runner.py`（原仓库已有，补全并修了 bug） |
| 参数传递 | 生成 Tcl 覆盖 Innovus 选项 | 生成 ORFS 的 `config.mk`，执行 `make DESIGN_CONFIG=... FLOW_VARIANT=iter_N` |
| 指标读取 | 解析 Innovus 报告 | 解析 ORFS 每个阶段的 metrics JSON（`4_1_cts.json`、`6_report.json`） |
| quick / full 保真度 | 跑到 CTS / route | 跑到 `make cts` / `make finish` |
| 调参参数 | 19 个 Innovus 选项（`PLACE_CONG_EFFORT`、`ROUTE_SI_DRIVEN` 等） | 12 个 ORFS 变量（见下表） |
| LLM | 推理：Qwen3-235B-thinking；生成：DeepSeek-V3.2 | 推理和生成都用 **deepseek-flash**（DeepSeek 官方 API） |

Innovus 的参数在 ORFS 里大多没有对应项，所以参数空间是按 ORFS 的变量重新设计的。数值范围以 ORFS 自带的同名设计参考配置为中心：

| 阶段 | 参数（ORFS 变量） | 范围 |
|---|---|---|
| 综合 | `ABC_AREA` | {0, 1}（偏面积 / 偏速度） |
| 布图规划 | `CORE_UTILIZATION`（%）、`CORE_ASPECT_RATIO`、`CORE_MARGIN` | 利用率按设计设定，例如 NG45-AES 是 30–55，ASAP7 是 50–70 |
| 布局 | `PLACE_DENSITY_LB_ADDON`、`GPL_TIMING_DRIVEN`、`GPL_ROUTABILITY_DRIVEN`、`CELL_PAD_IN_SITES_GLOBAL_PLACEMENT`、`TNS_END_PERCENT` | |
| 时钟树 | `CTS_CLUSTER_SIZE`、`CTS_CLUSTER_DIAMETER` | 直径：NG45 是 50–150 µm，ASAP7 是 20–80 µm |
| 布线 | `ROUTING_LAYER_ADJUSTMENT` | 0.30–0.60 |

### 1.1 代码改动一览

所有改动都基于原仓库的 `main` 分支（`66ea606 Initial open-source release`）。

**OpenROAD 适配：**

| 文件 | 改动 |
|---|---|
| `src/EDATuner/runners/openroad_runner.py` | 新增 `orfs.exec_prefix`（通过 sandbox 执行）、`orfs.threads`、`orfs.extra_vars`；配置路径支持 `${ENV}` 和相对路径；Verilog 支持通配符；**make 中途失败时记为失败**，不再拿早期阶段的数据冒充结果 |
| `src/EDATuner/utils/genMetrics.py` | 支持新版 ORFS 带阶段前缀的指标名（如 `cts__timing__setup__ws`），并优先取最后一个阶段的值 |
| `src/EDATuner/agents/ka_agent.py` | KA 的 system prompt 原来写死了 "Cadence Genus+Innovus" 和 Innovus 参数名，现在按 `eda.flow` 切换成 ORFS 的领域知识 |
| `src/EDATuner/main.py` | quick 阶段参与指纹计算的参数集合可以在配置里设置（`fidelity.quick_effective_params`） |
| `configs/<pdk>_<design>/openroad_tuner.yaml` | 6 个 case 的配置，由 `tools/gen_openroad_configs.py` 从 `nangate_aes` 的模板生成 |
| `benchmark/jpeg/include/dct_cos_table.v` | 仓库里缺少这个文件，从 ORFS 自带的同一个 OpenCores JPEG 设计中拷贝。已核对：和 OpenCores 原版逻辑完全一致，只有注释和 pragma 的写法不同 |
| `tools/setup_orfs.sh`、`tools/orfs-sandbox` | 免 root 安装 ORFS 工具链，并在 sandbox 里执行 |
| `tools/run_openroad_matrix.sh` | 6 个 case 排队并行运行；启动每个 case 前检查磁盘剩余空间 |
| `tools/summarize_results.py` | 从结果文件生成第 4 节的表格 |

**原仓库代码的 bug 修复（与 OpenROAD 无关，Cadence 流程同样受影响）：**

| 文件 | 问题 | 影响 |
|---|---|---|
| `agents/mo_ehvi_agent.py` | `qLogEHVI` 返回的是 **log(EHVI)**，代码却当作 EHVI 原值来用：自适应阈值只统计正值，打分又用 EHVI / 预计运行时间。前沿稳定之后 log(EHVI) < 0，所有候选都被阈值挡掉，**之后再也不会 promote 到 full** | 修复前，NG45-AES 在 2.5 小时后就停止 promote，6 小时只得到 33 个 full 点；修复后同样 6 小时得到 99 个，promote 一直持续到结束 |
| `main.py`、`openroad_runner.py` | 用 `wns > -99` 判断"有没有数据"。这是按 ns 单位写的，ASAP7 用 ps，WNS = −133 ps 这样的真实结果会被当成缺失 | ASAP7-IBEX 的 7 次试跑中有 3 次被误判为无效 |
| `knowledge/design_prior.py` | 设计先验的 LLM 调用失败时不重试。设计先验整个实验只生成一次，偶尔一次空响应就会让它在整个实验期间都降级 | 加了 2 次重试 |

---

## 2. 环境搭建

只需要普通用户权限，前提是机器上有 `bwrap`（Ubuntu 22.04/24.04 一般自带）和 Python ≥ 3.10。

```bash
# 0) 获取代码（二选一）
tar -xzf statetune-openroad-2026-10-26.tar.gz && cd stateTune-openroad   # 压缩包
# git clone <仓库地址> stateTune && cd stateTune                         # git 仓库

# 1) Python 环境（GP 模型很小，CPU 版 torch 就够了）
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

# 2) RAG 语料和嵌入模型
git clone https://github.com/lesliepy99/RAG-EDA.git third_party/RAG-EDA
HF_ENDPOINT=https://hf-mirror.com .venv/bin/hf download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir third_party/RAG-EDA/MiniLM-L6-v2 --exclude "onnx/*" "openvino/*" "*.h5" "*.ot"

# 3) ORFS 工具链：拉取固定版本的镜像（下载约 1.4 GB，占用磁盘约 4 GB），不需要 root
ORFS_HOME=/大磁盘/orfs tools/setup_orfs.sh
#    国内默认走 docker.1ms.run 镜像源；境外可以加 ORFS_REGISTRY=docker.io

# 4) 环境变量和 API key
cp .env.example .env    # 填 ORFS_HOME 和 PROVIDER_API_KEY
```

## 3. 运行

```bash
source .env

# 冒烟测试：nangate45 AES，关闭 LLM，不需要 API key，约 30 分钟
python -m src.EDATuner.main --config configs/nangate_aes/openroad_smoke.yaml

# 单个 case 的正式实验（预算 6 小时，同时跑 4 个任务，每个 8 线程）
python -m src.EDATuner.main --config configs/asap7_jpeg/openroad_tuner.yaml

# 全部 6 个 case，最多 3 个并行（约 12.5 小时）。进度看 runs/matrix_status.txt
setsid nohup tools/run_openroad_matrix.sh > /dev/null 2>&1 &
```

结果在 `runs/openroad_<design>_<pdk>/` 下：`final_report.md`（LLM 写的分析报告）、`kb/pareto.json`、`kb/summary.csv`、`kb/runs.jsonl`（每次评估的完整记录）。

**资源说明：**
- 每个 case 平均实际占用 6～10 个核，峰值 32 线程。
- 每次评估的 ORFS 中间文件约 0.4～0.8 GB；配置里设了每 5 次评估清理一次非 Pareto 点的中间文件。
- **注意磁盘**：我们有一次跑到一半，共享磁盘被写满，6 个 case 全部崩溃。
- **多人或多个实验共用一个 `ORFS_HOME` 时要注意**：ORFS 的中间结果按 `flow/results/<平台>/<设计>/iter_N` 存放，每次运行前会先清空同名目录。所以**同一个设计的两个实验不能同时用同一个 `ORFS_HOME`**，否则会互相覆盖。不同设计可以并行（`run_openroad_matrix.sh` 就是这样跑的）。需要同时跑同一个设计时，给每个实验单独装一份 `ORFS_HOME`（`tools/setup_orfs.sh` 指定不同目录即可）。

修改参数空间或预算时，请改 `configs/nangate_aes/openroad_tuner.yaml`（模板）或 `tools/gen_openroad_configs.py`，然后执行 `python tools/gen_openroad_configs.py` 重新生成其余 5 份配置。

---

## 4. 实验结果

**实验设置**：2026-10-23 至 24，每个 case 的墙钟预算 6 小时（`total_seconds: 21600`），同时跑 4 个任务、每个 8 线程，quick 跑到 CTS、full 跑到 finish，LLM 用 deepseek-flash，其余超参数与论文配置相同。原始结果在 `results/openroad_2026-10-24/`，下表由 `python tools/summarize_results.py` 生成。

### 4.1 我们的结果（StateTune + OpenROAD）

只统计 full 有效点，也就是完整跑到 finish、三项指标都有效的评估。

| PDK | Design | 时钟周期 | 最佳 WNS | 最佳面积 (µm²) | 最佳功耗 (mW) | Pareto 点数 | full 点数 | 总评估数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NanGate45 | AES | 0.82 ns | +0.09837 ns | 17191.3 | 222.44 | 9 | 99 | 313 |
| NanGate45 | JPEG | 1.0 ns | −0.0867 ns | 84181.3 | 480.00 | 10 | 40 | 147 |
| NanGate45 | IBEX | 2.2 ns | −0.01075 ns | 28645.0 | 20.36 | 15 | 76 | 230 |
| ASAP7 | AES | 380 ps | +2.803 ps | 1606.8 | 139.02 | 17 | 73 | 226 |
| ASAP7 | JPEG | 680 ps | +65.95 ps | 6397.2 | 139.73 | 9 | 45 | 135 |
| ASAP7 | IBEX | 1000 ps | −24.78 ps | 2420.3 | 38.65 | 9 | 45 | 130 |

> 三个"最佳"值是分别对每项指标取最优，**可能来自不同的参数点**。完整的折中关系请看各 case 的 `pareto.json`。

### 4.2 与 ORFS 默认参数对比（调参带来的提升）

"默认"是用 ORFS 自带参考配置的参数，在同一套工具上跑一次完整流程得到的结果。

| Case | WNS：默认 → 调参后 | 面积 (µm²)：默认 → 调参后 | 功耗 (mW)：默认 → 调参后 |
|---|---|---|---|
| NG45 AES | −0.0131 → **+0.0984** ns | 19898 → **17191**（−13.6%） | 314.7 → **222.4**（−29.3%） |
| NG45 JPEG | −0.177 → **−0.0867** ns | 86007 → **84181**（−2.1%） | 490.9 → **480.0**（−2.2%） |
| NG45 IBEX | −0.0206 → **−0.0108** ns | 29015 → **28645**（−1.3%） | 20.9 → **20.4**（−2.5%） |
| ASAP7 AES | −1.36 → **+2.80** ps | 1667 → **1607**（−3.6%） | 141.4 → **139.0**（−1.7%） |
| ASAP7 JPEG | +58.7 → **+66.0** ps | 6442 → **6397**（−0.7%） | 147.6 → **139.7**（−5.3%） |
| ASAP7 IBEX | −48.5 → **−24.8** ps | 2469 → **2420**（−2.0%） | 60.7 → **38.7**（−36.3%） |

### 4.3 与论文 StateTune 结果对比

论文数据取自 Table 6 中的 StateTune 行。论文的 WNS 单位：NanGate45 为 ns，ASAP7 为 ps；面积单位 µm²；功耗单位 mW。

| Case | WNS 论文 | WNS 本复现 | 面积 论文 | 面积 本复现 | 面积比 | 功耗 论文 | 功耗 本复现 | 功耗比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NG45 AES | −0.0120 | +0.0984 | 22353 | 17191 | 0.77× | 21.49 | 222.4 | 10.3× |
| NG45 JPEG | +0.0150 | −0.0867 | 106163 | 84181 | 0.79× | 111.28 | 480.0 | 4.3× |
| NG45 IBEX | +0.0600 | −0.0108 | 28928 | 28645 | 0.99× | 11.56 | 20.4 | 1.8× |
| ASAP7 AES | −162.86 | +2.80 ¹ | 1669 | 1607 | 0.96× | 7.60 | 139.0 ¹ | 18.3× |
| ASAP7 JPEG | −6.75 | +65.95 ¹ | 5946 | 6397 | 1.08× | 29.45 | 139.7 ¹ | 4.7× |
| ASAP7 IBEX | −246.92 | −24.78 ¹ | 2050 | 2420 | 1.18× | 6.21 | 38.7 ¹ | 6.2× |

¹ ASAP7 的这几项是在 **FF（fast-fast）工艺角**下得到的，论文用的是 **TT（typical）角**，两者不可直接比较，见 5.1 节。

论文表中另外两列没有放进对比表：
- **GP Hits**：该方法的点落在"所有方法合并后的全局 Pareto 前沿"上的个数。它需要和其他方法一起统计，我们只跑了 StateTune，所以没法计算。
- **HV**：超体积。它的数值依赖参考点和三个指标的量纲，工具链不同，指标的量级就不同（比如功耗差了 10 倍），所以没有可比性。

---

## 5. 与论文结果的差距及原因

**一句话总结：工艺角、功耗估算口径、综合和布局布线工具、参数空间、LLM 和预算都不同，所以绝对数值不可比；能比较的是"同一套工具上，调参相对默认参数提升了多少"（4.2 节）。** 下面按影响大小排序。

### 5.1 ASAP7 的 WNS 差距：主要是工艺角不同（FF vs TT）

ORFS 的 ASAP7 平台默认使用 **BC 角，也就是 FF（fast-fast）库**（`platforms/asap7/config.mk`: `CORNER ?= BC`）。原仓库的 Cadence 流程用的是 **RVT_TT**（`lib_corner: "RVT_TT"`）。FF 库的单元明显更快，所以我们在 ASAP7 上的 WNS 好得多。

**验证实验**：取我们跑出的 ASAP7 最终网表（布线后），只把时序库从 FF 换成 TT，用 OpenROAD 重新做时序和功耗分析：

| 网表 | WNS @ FF | WNS @ TT | 论文 StateTune（TT） | 功耗 FF → TT |
|---|---:|---:|---:|---|
| ASAP7 AES（iter_51） | −6.3 ps | **−157.9 ps** | −162.9 ps | 152 → 115 mW（−24%） |
| ASAP7 JPEG（iter_112） | +53.0 ps | **−190.9 ps** | −6.8 ps | 150 → 113 mW（−25%） |
| ASAP7 IBEX（iter_65） | −38.2 ps | **−439.2 ps** | −246.9 ps | 41.6 → 31.1 mW（−25%） |

- 换到 TT 角之后，AES 的 WNS（−158 ps）和论文（−163 ps）已经非常接近。
- JPEG 和 IBEX 在 TT 下比论文差。原因是这些网表是按 FF 角优化的，换到 TT 只是"重新评估"，并没有按 TT 重新做时序修复，所以结果偏悲观。如果整个流程都在 TT 下跑，工具会针对 TT 修复时序，结果会好一些。
- 要和论文对齐，ASAP7 的 3 个 case 需要在配置的 `orfs.extra_vars` 里加 `CORNER: TC`（ORFS 里把 typical 角称为 TC）后重跑。**本文报告的结果没有加这个设置。**

### 5.2 功耗差 1.8～18 倍：主要是翻转率假设不同，不是设计更差

- 单位：OpenROAD 报告用 W，Innovus 用 mW，上表已经统一换算成 mW。
- 以 NG45 AES 为例，功耗里 **94.5% 是组合逻辑的动态功耗**，漏电只有约 0.4 mW。动态功耗正比于翻转率（switching activity）。这里没有仿真波形（VCD/SAIF），工具只能自己假设翻转率：
  - **OpenROAD/OpenSTA 默认**：从输入和寄存器出发，按概率沿逻辑往下传播。AES 的 S-box 和 MixColumns 里有大量 XOR，XOR 不会让翻转率衰减，所以内部翻转率很高。
  - **Innovus 默认**：没有波形时用默认翻转率。我们推测效果接近给所有线一个统一的较低值，但没有 Innovus，无法验证。
- **验证实验**：取同一个 NG45 AES 布局后的网表，只改翻转率假设，用 OpenROAD 重新算功耗：

  | 翻转率假设 | 总功耗 |
  |---|---:|
  | OpenSTA 默认（概率传播） | 218 mW |
  | 所有线统一 0.2（由 0.1 那行外推） | 约 25 mW |
  | 所有线统一 0.1 | 12.8 mW |
  | 所有线统一 0.05 | 7.9 mW |

  同一个网表，只改假设就能差 17 倍。论文的 21.5 mW 正好落在"统一 0.1～0.2"这个区间。
- **各设计的功耗比和这个解释一致**：XOR 密集的 AES 差得最多（NG45 10.3×，ASAP7 18.3×）；JPEG 居中（4.3×～4.7×）；IBEX 是以控制逻辑为主的 CPU，差得最少（1.8×）。
- ASAP7 还要叠加 FF 角的影响：FF 角比 TT 的功耗高约 25%，见 5.1 节。

### 5.3 面积和 NanGate45 WNS 的差异：综合和时序分析工具不同

- **综合器**：Genus 换成了 Yosys + ABC，网表本身就不一样。NG45 AES 和 JPEG 的面积小 20% 以上，IBEX 和 ASAP7 AES 基本持平，ASAP7 JPEG 和 IBEX 大 8%～18%。ORFS 的平台还自带 dont_use 单元列表、tie/buffer 策略等设置，和论文的 Genus 脚本不同。
- **时序分析和寄生参数**：Innovus 有自己的 RC 提取和延迟计算，OpenROAD 用 OpenSTA 加上自己的寄生估算或提取，同一个网表也会得出不同的 WNS。
- **优化强度不同**：Innovus 在 place/CTS/route 各阶段的优化，和 ORFS 的 `repair_timing` / `repair_design` 不同。NanGate45 上 WNS 的差异有正有负：AES 我们更好，JPEG 和 IBEX 论文更好。这说明并不是哪一方的工具整体占优，而是不同设计在两套工具上各有表现。

### 5.4 其他实验设置差异

| 差异 | 论文 | 本复现 | 影响 |
|---|---|---|---|
| ASAP7 工艺角 | RVT TT | RVT FF（ORFS 默认） | 见 5.1，影响最大 |
| 参数空间 | 19 个 Innovus 选项 | 12 个 ORFS 变量（重新设计） | 搜索空间不同，结果不可比 |
| full 阶段终点 | route | finish（多了 fill 和最终报告） | 指标来自最终报告，更完整 |
| LLM | Qwen3-235B-thinking + DeepSeek-V3.2 | 推理和生成都用 deepseek-flash | 推理能力和成本不同 |
| 预算 | 192 thread·h（4 线程 × 48 h） | 每个 case 墙钟 6 h，4 任务 × 8 线程 | 我们的评估次数少得多（130～313 次）。而且服务器是共享的，实际分到的算力会波动 |
| EHVI promote | 原代码 | 修复了 log-EHVI bug（见 1.1） | 论文的数字是否受这个 bug 影响无法确认 |
| 重复次数 | 3 个 seed | 1 次 | 没有方差估计 |

### 5.5 对结论的影响

- **能确认的**：StateTune 的完整闭环可以在开源工具链上稳定运行，6 个 case 全部正常结束，没有程序崩溃；KA 每个 case 生成了约 50～150 个候选，推理内容和领域知识相符；调参后的结果在所有 case 上都优于 ORFS 默认参数。
- **不能确认的**：论文报告的数值，以及 StateTune 相对各 baseline 的优势在 OpenROAD 上是否成立。要验证后者，需要在同一套工具链和同样的预算下运行 `baselines/run_baseline.py`。原仓库的 baseline 配置是 Cadence 版本，需要另外改写。

## 6. 已知限制和后续工作

- ASAP7 的 3 个 case 是在 FF 角下跑的；要和论文对齐，需要加 `CORNER: TC` 后重跑（见 5.1）。
- 只跑了 1 个 seed，结果没有方差。
- baseline（BO-qEHVI、Optuna-TPE、Random、RankTuner、CROP）还没在 OpenROAD 上跑。
- 如果需要让功耗量级接近论文，可以在 ORFS 报告功耗前统一设定翻转率（`set_power_activity -global`），或者用仿真得到的 SAIF 文件。但这只会让数字看起来接近，不代表更准确；而且修改之后，需要所有方法都重跑。
- README 里提到的 `--override` 命令行参数在代码里并不存在，改种子或预算需要直接修改 YAML。
