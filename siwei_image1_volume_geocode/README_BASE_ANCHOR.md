# 建筑底面 SAR 投影基准

建筑底面不是 SAR 图上的某一条亮线。右视、降轨 SAR 中，墙面与屋顶会发生叠掩，墙脚还可能被遮蔽。因此本目录统一采用以下可复现定义：

1. 平面位置：建筑矢量的地面轮廓；
2. 报告高程：项目 DSM 的吴淞高程 `4.000 m`；
3. 成像投影高程：暂用 GAMMA 自带 EGM96 高程面转换后的 WGS84 椭球高，校园内约 `15.182–15.241 m`；
4. 传感器模型：厂家 `source.rpb` RPC，并以 GAMMA 轨道模型交叉检查；
5. 模拟/实测 SAR 配准：只估计传感器模型残差，不再代替垂直基准转换。

## 已修正的问题

旧代码把 `4 m` 吴淞高程直接作为椭球高送入 RPC/GAMMA，这是错误的。厂家 `sceneAverageHeight/RPC heightOffset=15.9946346283 m` 是场景参考高和 RPC 归一化中心，不能直接当作吴淞至 WGS84 的转换结果。

当前底面首投使用 `coord_to_sarpix_list`，按每个轮廓顶点的位置从 GAMMA EGM96 格网取得高程异常，并暂按 `h_WGS84 = 4.000 m + N_EGM96` 转换。EGM96 不是经测量确认的吴淞高程转换面，因此该结果是明确标注的暂行基准；待本地吴淞—WGS84控制点到位后替换。

旧的“错误高程 + 大平移”可能在数值上部分互相抵消，但没有物理可解释性，不能作为底面准确性的证明。`work/gamma_simulated_sar_ellipsoid` 中曾使用厂家平均高构造的固定 `DEM_hgt_offset`，不再作为吴淞转换依据。

## 审计产物

- `PICALL/017_图件_859104265266.png`：错误基准与物理底面的全区对照；
- `work/building_base_projection_rpc_audit.csv`：1028 栋建筑的底面中心 SAR 像素；
- `work/building_base_projection_rpc_audit_summary.json`：垂直基准和配准改正摘要；
- `code/audit_building_base_projection.py`：可重复生成上述产物；
- `code/rpc_projection_core.py`：厂家 RPC 正投影实现。
- `PICALL/018_图件_239081998462.png`：当前 GAMMA 底面首投影；
- `work/building_base_gamma_projection_vertices.csv`：每个底面顶点的吴淞高程、EGM96高程异常、椭球高及 SAR 像素；
- `code/project_building_bases_with_gamma.py`：调用 `coord_to_sarpix_list` 的可重复批处理脚本。
- `PICALL/019_图件_715556404984.png`：应用最终 `DIFF_par` 后的底面；
- `work/building_base_gamma_projection_vertices_refined.csv`：精化后的7143个底面顶点；
- `work/gamma_base_refinement/base_refinement.diff_par`：最终常数距离向/方位向改正；
- `work/gamma_base_refinement/refinement_summary.json`：两轮窗口筛选、模型选择和收敛统计。

## GAMMA 残差迭代结果

在 EGM96 暂行椭球高 DSM 上重新运行 `gc_map2` 和 `geocode`，再以模拟 SAR 为参考、实测 SAR 为待配准影像运行 `offset_pwrm`。第一轮560个窗口中，保留相关系数不低于0.15且与多尺度全局峰距离不超过3 pixel的33个一致窗口；`offset_fitm` 常数模型得到距离向 `+3.82297 pixel`、方位向 `-19.07479 pixel`。三参数空间模型没有改善五折交叉验证中位残差，故拒绝。

应用 `gc_map_fine` 后重新匹配，第二轮一致窗口的残差均值为距离向 `+0.020 pixel`、方位向 `-0.020 pixel`，中位数为 `-0.031/+0.063 pixel`，低于0.1 pixel收敛阈值，因此不再追加改正。最终 `coord_to_sarpix_list` 通过 `DIFF_par` 将同一常数改正应用到全部底面顶点。

## 仍需满足的绝对精度条件

当前结果已经做到“传感器模型、垂直基准、裁剪偏移、残差配准”分项明确，但不能仅凭建筑亮边宣称绝对正确。若要求测量意义上的确定，还需要至少若干个分布于影像四周、具有已知三维坐标且能在 SAR 中唯一识别的地面控制目标（优先角反射器或实测固定散射点）。应用这些控制点后，应报告距离向/方位向 RMSE、最大残差和空间趋势；在此之前，图中的底面应称为“RPC/GAMMA 模型确定的物理底面位置”。

控制点录入格式见 `inputs/image1/ground_control_points_template.csv`。坐标必须是 WGS84 经度、纬度和椭球高，SAR 像素必须是当前裁剪影像的零起算列、行；示例行使用前必须删除。
