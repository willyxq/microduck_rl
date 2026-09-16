# UMR 输入是什么：不是视频，也不是「机器人 A 的动作文件」

小红书笔记（Microduck T-pose → Studio retarget）容易让人以为 UMR
是「两个异形机器人互拷动作，输入是源机器人上的一段动作」。
官方论文和仓库不是这样切的。

UMR = Unified Motion Retargeting
([arXiv:2609.02134](https://arxiv.org/abs/2609.02134)，
[代码](https://github.com/hanyang9/UMR))。
它做的是 **有外壳的源运动 → 目标机器人 MJCF 的 `qpos`**。
默认源是人体网格运动，目标才是 G1 / Microduck 这类机体。
点云对应是内部接口，不是用户先要去采的「mocap 表面」。

## 源和目标分别是谁

| 角色 | 是什么 | 不是什么 |
| --- | --- | --- |
| 源 | 能给出 **T-pose 网格 + 逐帧 posed 外壳** 的运动 | 一段 mp4，或任意机器人的 encoder 日志 |
| 目标 | 带 mesh 的机器人 MJCF + 一个对齐过的 T-pose `qpos` | Isaac USD / Lab 环境 |
| 输出 | 目标机器人广义坐标轨迹（本地仓库里是 `.npz`） | 可直接上 Isaac Lab 的策略 |

论文原话：任何源只要提供 canonical T-pose mesh 和随时间变化的 posed
surfaces，就能进同一条管线（SMPL 族、SOMA、蒙皮人形角色、扫描人体网格）。
对应关系在 **两边的 T-pose 外壳** 上只学一次，再绑到各自 mesh 上随运动走。

所以：

- Microduck 笔记里的鸭子是 **目标机体**，不是源。
- 源仍是 Studio 里选的那段人体 / 角色参考动作。
- 机器人对机器人在几何上说得通（源也是 MJCF + mesh + 动作），但发布的
  adapter 几乎全是 **人 → 人形**。角色管线是唯一接近「有骨架的源角色 → 机器人」
  的官方路径。

UMR 本体是 MuJoCo + Clarabel，不接 Isaac Sim / Lab。下游 BeyondMimic /
Holosoma 可以再吃它的参考轨迹。

## 用户真正喂进去的文件

UMR **不读视频**。各 adapter 先把源变成「能驱动外壳的 3D 运动」，再采样点云。

| 源 | 磁盘格式 | 变成外壳的方式 |
| --- | --- | --- |
| LAFAN1 / SMPL-X | `.npz`：`poses`/`pose_aa` `[T,165]` 或 `[T,55,3]`，`trans` `[T,3]`，`betas`，可选 `fps`/`gender` | 官方 SMPL-X 模型把轴角 pose 建成人体 mesh |
| GRAIL / OmniContact / OMOMO | SMPL-X 序列 + 物体位姿（pkl / 自有 layout） | 同上，外加物体点云做接触 |
| BONES-SEED / SOMA | 门控 BVH + SOMA-X / MHR 身材参数 | NVIDIA SOMA 网格，不是光骨架 BVH |
| MimicKit 角色 | `.pkl`：`frames` `[T, 6+N]`（root XYZ + root expmap + 源角色 DoF）+ `fps` | 源角色自己的蒙皮 MJCF |
| NR | FBX / BVH | 解析出 mesh 运动后再采样 |
| AdaPT | SMPL-X + 球拍 | 人体和球拍一起当外壳 |

LAFAN1 原 BVH **不能**直接进 UMR，要先 `lafan_to_smplx`。
Studio 浏览器结果因授权 **不能下载**；要数据得跑本地仓库。

## 「mocap 表面」是不是输入

不是采集端输入，是 **统一几何接口**。

1. 磁盘上先有能驱动 mesh 的运动（SMPL-X pose、角色 DoF、SOMA BVH+shape）。
2. 从 T-pose 外壳采有序点云，学源 ↔ 目标对应。
3. 运动时用重心坐标 / FK 把这些点带走，再优化目标 `q`。

传统光学 mocap 贵，是因为它直接给出第 1 步的 3D。
UMR 没有省掉这一步，只是不再手工配关节点。

## 视频好采，但还到不了 UMR

拍机器人或生成机器人视频，解决的是 **像素便宜**。
UMR 要的是 **逐帧 3D 外壳**。中间缺的是 video → posed mesh：

- 人体：4D-Humans / GVHMR / SMPL 拟合，已经能喂 LAFAN 那条 adapter。
- 异形机器人（Microduck）：没有 SMPL。要从像素反解 mesh 或 14 关节 + root。
  imitation 分支上的 2D 跟踪 + 启发式 / 弱透视 IK 已经试过，不够像
  （`docs/video_retarget_lessons.md`）。
- 视频生成模型给出的是新像素，不是 `qpos`，也不是对应好的点云。

可走的路，按离 UMR 的距离：

1. **人体视频 → SMPL-X → UMR → Microduck**  
   用现成人体重建喂官方 adapter。得到的是「人的动作糊到鸭子上」，
   不是「另一只鸭子的动作」。笔记里的示例 locomotion 就是这条。
2. **源机器人已有 3D 动作（仿真 rollout / 编码器）→ 当源角色**  
   才是真正的机器人对机器人。输入应是 MJCF + mesh + `[T, 6+N]` 一类
   轨迹，不是 mp4。
3. **机器人视频 → 3D 外壳 / 关节**  
   仍是缺的那一截。标定多机位、silhouette render-and-compare、真机编码器
   比再调 2D 点规则更接近 UMR 要的输入。

视频模型有用的位置是 **扩增看起来像的片段**，或当重建网络的训练数据；
不能绕过「像素 → 带时间的 3D 表面」。没有这条，UMR 无输入可吃。
