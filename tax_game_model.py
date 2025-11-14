import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------
# 1. 核心计算（基于税收竞争模型）
# ---------------------------------------------------------------------
def calculate_payoffs(N, alpha, beta, t_H, t_L, t_M=None):
    """
    计算博弈收益，核心逻辑：
    - 总资本量：受税收弹性影响，K_total = 1 - α*(t_A + t_B)/2
    - 资本分配：受资本流动性影响，K_i = 1/2 + β*(t_j - t_i)
    - 税收收入：R_i = t_i * K_i * K_total（最终收益）
    """
    payoffs = {}
    
    if N == 2:
        # 基础场景：tH（高税率）、tL（低税率）
        # (tH, tH) 合作场景
        t_avg_C = t_H
        K_total_C = 1 - alpha * t_avg_C
        K_i_share_C = 0.5 + beta * (t_H - t_H)
        payoffs['R_C'] = round(t_H * K_i_share_C * K_total_C, 4)  # 合作收益
        
        # (tL, tL) 惩罚场景
        t_avg_P = t_L
        K_total_P = 1 - alpha * t_avg_P
        K_i_share_P = 0.5 + beta * (t_L - t_L)
        payoffs['R_P'] = round(t_L * K_i_share_P * K_total_P, 4)  # 惩罚收益
        
        # 混合场景：1个tL（背叛）、1个tH（合作）
        t_avg_D = (t_H + t_L) / 2
        K_total_D = 1 - alpha * t_avg_D
        
        # 背叛者收益（t_i=tL）
        K_i_share_Deviator = 0.5 + beta * (t_H - t_L)
        payoffs['R_D'] = round(t_L * K_i_share_Deviator * K_total_D, 4)
        
        # 合作者收益（t_i=tH）
        K_i_share_Sucker = 0.5 + beta * (t_L - t_H)
        payoffs['R_S'] = round(t_H * K_i_share_Sucker * K_total_D, 4)
        
        # 拓展场景：tM（中税率）
        if t_M is not None:
            # (tM, tM) 场景
            t_avg_M = t_M
            K_total_M = 1 - alpha * t_avg_M
            K_i_share_M = 0.5 + beta * (t_M - t_M)
            payoffs['R_M'] = round(t_M * K_i_share_M * K_total_M, 4)
            
            # 混合场景：tM与tH
            t_avg_MH = (t_M + t_H) / 2
            K_total_MH = 1 - alpha * t_avg_MH
            payoffs['R_MvsH'] = round(t_M * (0.5 + beta * (t_H - t_M)) * K_total_MH, 4)
            payoffs['R_HvsM'] = round(t_H * (0.5 + beta * (t_M - t_H)) * K_total_MH, 4)
            
            # 混合场景：tM与tL
            t_avg_ML = (t_M + t_L) / 2
            K_total_ML = 1 - alpha * t_avg_ML
            payoffs['R_MvsL'] = round(t_M * (0.5 + beta * (t_L - t_M)) * K_total_ML, 4)
            payoffs['R_LvsM'] = round(t_L * (0.5 + beta * (t_M - t_L)) * K_total_ML, 4)
    
    else:
        # N > 2 泛化公式
        # 全合作场景（均选tH）
        t_avg_C = t_H
        K_total_C = 1 - alpha * t_avg_C
        K_i_share_C = (1/N) + beta * (t_avg_C - t_H)
        payoffs['R_C'] = round(t_H * K_i_share_C * K_total_C, 4)
        
        # 全惩罚场景（均选tL）
        t_avg_P = t_L
        K_total_P = 1 - alpha * t_avg_P
        K_i_share_P = (1/N) + beta * (t_avg_P - t_L)
        payoffs['R_P'] = round(t_L * K_i_share_P * K_total_P, 4)
        
        # 混合场景：1个tL（背叛）、N-1个tH（合作）
        t_avg_D = ((N - 1) * t_H + t_L) / N
        K_total_D = 1 - alpha * t_avg_D
        payoffs['R_D'] = round(t_L * ((1/N) + beta * (t_avg_D - t_L)) * K_total_D, 4)
        payoffs['R_S'] = round(t_H * ((1/N) + beta * (t_avg_D - t_H)) * K_total_D, 4)
    
    # 确保收益非负
    for key in payoffs:
        payoffs[key] = max(payoffs[key], 0.0000)
        
    return payoffs

# ---------------------------------------------------------------------
# 2. 主界面
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="N-Player 税收竞争博弈模拟器")
st.title("N-Player 税收竞争博弈模拟器")
st.markdown("---")

# ---------------------------------------------------------------------
# 3. 侧边栏：参数设置
# ---------------------------------------------------------------------
with st.sidebar.container(border=True):
    st.sidebar.header("⚙️ 核心经济参数")
    st.sidebar.markdown("影响资本总量与流动的关键参数，决定博弈基础环境")
    
    N = st.sidebar.slider(
        "N（政府数量）", 
        min_value=2, max_value=10, value=2, step=1,
        help="参与税率竞争的政府数量，数量增加会加剧竞争强度"
    )
    alpha = st.sidebar.slider(
        "α（税收弹性系数）", 
        min_value=0.0, max_value=3.0, value=1.2, step=0.1,
        help="总资本量对平均税率的敏感度，α越大，高税率对资本总量的抑制越强，参考范围[0.5,2.0]"
    )
    beta = st.sidebar.slider(
        "β（资本流动性系数）", 
        min_value=0.0, max_value=3.0, value=1.5, step=0.1,
        help="资本对税率差的敏感度，β越大，资本越倾向流向低税率地区，参考范围[0.5,2.0]"
    )

with st.sidebar.container(border=True):
    st.sidebar.header("⚙️ 策略与耐心参数")
    st.sidebar.markdown("政府的税率选择与长期收益偏好，决定博弈均衡结果")
    
    t_H_val = st.sidebar.slider(
        "tH（高税率 - 合作策略）", 
        min_value=0.01, max_value=1.0, value=0.4, step=0.01,
        help="合作状态下的高税率，对应现实中40%左右的税率水平"
    )
    t_L_val = st.sidebar.slider(
        "tL（低税率 - 背叛/惩罚策略）", 
        min_value=0.01, max_value=t_H_val-0.01, value=0.2, step=0.01,
        help="背叛或惩罚状态下的低税率，需小于tH，对应现实中20%左右的税率水平"
    )
    t_M_val = st.sidebar.slider(
        "tM（中税率 - 备选策略）", 
        min_value=0.01, max_value=t_H_val-0.01, value=0.3, step=0.01,
        help="拓展分析中的中税率，对应现实中30%左右的税率水平，用于多策略博弈"
    )
    delta = st.sidebar.slider(
        "δ（政府耐心/贴现率）", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="政府对未来收益的重视程度，δ越大，越倾向长期合作，取值范围(0,1)"
    )

# ---------------------------------------------------------------------
# 4. 核心模型展示（统一数学公式格式）
# ---------------------------------------------------------------------
st.header("1. 核心模型设定")
st.markdown("### 1.1 模型假设")
st.markdown("""
- **参与者**：N个综合实力相近的政府，以税收收入最大化为目标
- **策略空间**：离散税率选择（tL=低税率、tM=中税率、tH=高税率）
- **信息结构**：完全信息静态博弈（双方知晓对方收益函数与策略空间）
- **资本特性**：资本可跨区域流动，受税率差异和总税收弹性双重影响
""")

st.markdown("### 1.2 核心公式")
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("#### 1.2.1 资本分配函数（资本流动性）")
    st.latex(r"K_i = \frac{1}{2} + \beta \cdot (t_j - t_i)")
    st.markdown("**说明**：$K_i$ 为政府$i$的资本存量，$β$ 越大，资本对税率差的敏感度越高，流动越频繁")
    
    st.markdown("#### 1.2.2 总资本函数（税收弹性）")
    st.latex(r"K_{\text{total}} = 1 - \alpha \cdot \frac{t_A + t_B}{2}")
    st.markdown("**说明**：$K_{\text{total}}$ 为市场总资本量，$α$ 越大，高税率对资本总量的抑制作用越强")

with col2:
    st.markdown("#### 1.2.3 最终收益函数")
    st.latex(r"R_i = t_i \times K_i \times K_{\text{total}}")
    st.latex(r"R_i = t_i \times \left[ \frac{1}{2} + \beta \cdot (t_j - t_i) \right] \times \left( 1 - \alpha \cdot \frac{t_A + t_B}{2} \right)")
    st.markdown("**说明**：$R_i$ 为政府$i$的税收收入，即博弈中的收益，保留4位小数")

st.markdown("### 1.3 连续策略古诺模型")
st.latex(r"t_A^* = \frac{1}{4\beta} + \frac{t_B}{2}, \quad t_B^* = \frac{1}{4\beta} + \frac{t_A}{2}")
st.latex(r"t^* = \frac{1}{2\beta}, \quad R^* = \frac{1}{4\beta}")
st.markdown("**均衡结论**：$β$（资本流动性）越大，均衡税率$t^*$越低，政府间税率竞争越激烈")
st.markdown("---")

# ---------------------------------------------------------------------
# 5. 静态博弈分析
# ---------------------------------------------------------------------
st.header("2. 静态博弈分析（一次性博弈）")
if N == 2:
    # 计算所有场景收益
    payoffs = calculate_payoffs(N, alpha, beta, t_H_val, t_L_val, t_M_val)
    
    # --------------------------
    # 场景1：双策略博弈（tL、tH）
    # --------------------------
    with st.container(border=True):
        st.subheader("2.1 双策略博弈（tL=低税率 vs tH=高税率）")
        # 支付矩阵数据（A收益, B收益）
        HH = (payoffs['R_C'], payoffs['R_C'])       # (tH, tH)
        HL = (payoffs['R_S'], payoffs['R_D'])       # (tH, tL)
        LH = (payoffs['R_D'], payoffs['R_S'])       # (tL, tH)
        LL = (payoffs['R_P'], payoffs['R_P'])       # (tL, tL)
        
        # 显示矩阵
        matrix1_data = {
            f"B选tL（{t_L_val*100:.1f}%，背叛）": [f"({HH[0]:.4f}, {HH[1]:.4f})", f"({LH[0]:.4f}, {LH[1]:.4f})"],
            f"B选tH（{t_H_val*100:.1f}%，合作）": [f"({HL[0]:.4f}, {HL[1]:.4f})", f"({LL[0]:.4f}, {LL[1]:.4f})"]
        }
        matrix1_index = [f"A选tH（{t_H_val*100:.1f}%，合作）", f"A选tL（{t_L_val*100:.1f}%，背叛）"]
        df1 = pd.DataFrame(matrix1_data, index=matrix1_index)
        st.dataframe(df1, use_container_width=True)
        
        # 划线法分析（下划线标记最优收益）
        st.subheader("2.1.1 划线法找纳什均衡")
        def underline_best(value, is_best):
            return f"<u>{value:.4f}</u>" if is_best else f"{value:.4f}"
        
        # 步骤1：A的最优反应（固定B的列）
        st.markdown("**步骤1：为A的最优收益划线（固定B的选择）**")
        a_col1_max = max(HH[0], LH[0])  # B选tL列
        a_hh_marked = underline_best(HH[0], HH[0] == a_col1_max)
        a_lh_marked = underline_best(LH[0], LH[0] == a_col1_max)
        
        a_col2_max = max(HL[0], LL[0])  # B选tH列
        a_hl_marked = underline_best(HL[0], HL[0] == a_col2_max)
        a_ll_marked = underline_best(LL[0], LL[0] == a_col2_max)
        
        a_matrix = {
            f"B选tL（{t_L_val*100:.1f}%）": [f"({a_hh_marked}, {HH[1]:.4f})", f"({a_lh_marked}, {LH[1]:.4f})"],
            f"B选tH（{t_H_val*100:.1f}%）": [f"({a_hl_marked}, {HL[1]:.4f})", f"({a_ll_marked}, {LL[1]:.4f})"]
        }
        a_df = pd.DataFrame(a_matrix, index=matrix1_index)
        st.write(a_df.to_html(escape=False), unsafe_allow_html=True)
        
        # 步骤2：B的最优反应（固定A的行）
        st.markdown("**步骤2：为B的最优收益划线（固定A的选择）**")
        b_row1_max = max(HH[1], HL[1])  # A选tH行
        b_hh_marked = underline_best(HH[1], HH[1] == b_row1_max)
        b_hl_marked = underline_best(HL[1], HL[1] == b_row1_max)
        
        b_row2_max = max(LH[1], LL[1])  # A选tL行
        b_lh_marked = underline_best(LH[1], LH[1] == b_row2_max)
        b_ll_marked = underline_best(LL[1], LL[1] == b_row2_max)
        
        b_matrix = {
            f"B选tL（{t_L_val*100:.1f}%）": [f"({HH[0]:.4f}, {b_hh_marked})", f"({LH[0]:.4f}, {b_lh_marked})"],
            f"B选tH（{t_H_val*100:.1f}%）": [f"({HL[0]:.4f}, {b_hl_marked})", f"({LL[0]:.4f}, {b_ll_marked})"]
        }
        b_df = pd.DataFrame(b_matrix, index=matrix1_index)
        st.write(b_df.to_html(escape=False), unsafe_allow_html=True)
        
        # 步骤3：合并结果找均衡
        st.markdown("**步骤3：纳什均衡（双下划线单元格）**")
        combined_hh = f"({a_hh_marked}, {b_hh_marked})"
        combined_hl = f"({a_hl_marked}, {b_hl_marked})"
        combined_lh = f"({a_lh_marked}, {b_lh_marked})"
        combined_ll = f"({a_ll_marked}, {b_ll_marked})"
        
        combined_matrix = {
            f"B选tL（{t_L_val*100:.1f}%）": [combined_hh, combined_lh],
            f"B选tH（{t_H_val*100:.1f}%）": [combined_hl, combined_ll]
        }
        combined_df = pd.DataFrame(combined_matrix, index=matrix1_index)
        st.write(combined_df.to_html(escape=False), unsafe_allow_html=True)
        
        # 判断均衡
        nes = []
        if "<u>" in a_hh_marked and "<u>" in b_hh_marked:
            nes.append(f"(tH, tH) 收益: ({HH[0]:.4f}, {HH[1]:.4f})")
        if "<u>" in a_hl_marked and "<u>" in b_hl_marked:
            nes.append(f"(tH, tL) 收益: ({HL[0]:.4f}, {HL[1]:.4f})")
        if "<u>" in a_lh_marked and "<u>" in b_lh_marked:
            nes.append(f"(tL, tH) 收益: ({LH[0]:.4f}, {LH[1]:.4f})")
        if "<u>" in a_ll_marked and "<u>" in b_ll_marked:
            nes.append(f"(tL, tL) 收益: ({LL[0]:.4f}, {LL[1]:.4f})")
        
        if nes:
            st.success(f"纯策略纳什均衡：{', '.join(nes)}")
            # 囚徒困境判断
            if len(nes) == 1 and "(tL, tL)" in nes[0] and HH[0] > LL[0]:
                st.warning(f"囚徒困境：唯一均衡是(tL, tL)，但合作(tH, tH)收益更高（{HH[0]:.4f} > {LL[0]:.4f}）")
                st.markdown("**原因**：资本流动性高（β大）时，单边降税能吸引大量资本，背叛诱惑极强")
        else:
            st.info("无纯策略纳什均衡，可能存在混合策略均衡")
    
    # --------------------------
    # 场景2：三策略博弈（tL、tM、tH）
    # --------------------------
    with st.container(border=True):
        st.subheader("2.2 三策略博弈（tL=低税率 vs tM=中税率 vs tH=高税率）")
        # 支付矩阵数据（A收益, B收益）
        MM = (payoffs['R_M'], payoffs['R_M'])                   # (tM, tM)
        ML = (payoffs['R_MvsL'], payoffs['R_LvsM'])             # (tM, tL)
        MH = (payoffs['R_MvsH'], payoffs['R_HvsM'])             # (tM, tH)
        LM = (payoffs['R_LvsM'], payoffs['R_MvsL'])             # (tL, tM)
        HM = (payoffs['R_HvsM'], payoffs['R_MvsH'])             # (tH, tM)
        
        matrix2_data = {
            f"B选tL（{t_L_val*100:.1f}%）": [f"({HH[0]:.4f}, {HH[1]:.4f})", f"({HM[0]:.4f}, {HM[1]:.4f})", f"({LH[0]:.4f}, {LH[1]:.4f})"],
            f"B选tM（{t_M_val*100:.1f}%）": [f"({MH[0]:.4f}, {MH[1]:.4f})", f"({MM[0]:.4f}, {MM[1]:.4f})", f"({LM[0]:.4f}, {LM[1]:.4f})"],
            f"B选tH（{t_H_val*100:.1f}%）": [f"({HL[0]:.4f}, {HL[1]:.4f})", f"({ML[0]:.4f}, {ML[1]:.4f})", f"({LL[0]:.4f}, {LL[1]:.4f})"]
        }
        matrix2_index = [f"A选tH（{t_H_val*100:.1f}%）", f"A选tM（{t_M_val*100:.1f}%）", f"A选tL（{t_L_val*100:.1f}%）"]
        df2 = pd.DataFrame(matrix2_data, index=matrix2_index)
        st.dataframe(df2, use_container_width=True)
        
        st.markdown("**关键结论**：三策略博弈存在两个纯策略纳什均衡（tM, tM）和（tL, tL），其中（tM, tM）收益更优，需通过政策协调实现")

else:
    st.info(f"N > 2 时，支付矩阵维度为N×N，过于复杂，此处聚焦N=2的基础分析，便于直观理解税率竞争逻辑")

st.markdown("---")

# ---------------------------------------------------------------------
# 6. 动态博弈分析
# ---------------------------------------------------------------------
st.header("3. 动态博弈分析（无限重复博弈）")
with st.container(border=True):
    st.subheader("3.1 冷酷触发策略")
    st.markdown("""
    1. **合作路径**：双方永远选择高税率（tH），长期收益现值
    """)
    st.latex(r"PV_{\text{合作}} = \frac{R_C}{1 - \delta}")
    st.markdown("""
    2. **背叛路径**：当期选择中税率（tM）背叛，未来永远被惩罚（均选tL），长期收益现值
    """)
    st.latex(r"PV_{\text{背叛}} = R_D + \frac{\delta \cdot R_P}{1 - \delta}")
    st.markdown("**说明**：$R_D$ 为背叛当期收益（tM vs tH），$R_P$ 为惩罚期收益（tL vs tL），均保留4位小数")

# 计算动态博弈关键收益
payoffs_dynamic = calculate_payoffs(2, alpha, beta, t_H_val, t_L_val, t_M_val)
R_C = payoffs_dynamic['R_C']       # 合作收益（tH, tH）
R_D = payoffs_dynamic['R_MvsH']    # 背叛收益（tM背叛tH）
R_P = payoffs_dynamic['R_P']       # 惩罚收益（tL, tL）

# 显示关键收益
with st.container(border=True):
    st.subheader("3.2 关键收益值")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.metric("合作收益 $R_C$（tH, tH）", f"{R_C:.4f}")
    with col2:
        st.metric("背叛收益 $R_D$（tM背叛tH）", f"{R_D:.4f}")
    with col3:
        st.metric("惩罚收益 $R_P$（tL, tL）", f"{R_P:.4f}")

# 计算临界贴现率
with st.container(border=True):
    st.subheader("3.3 合作稳定性判断")
    if R_D > R_C and R_C > R_P:
        # 满足囚徒困境前提
        delta_critical = round((R_D - R_C) / (R_D - R_P), 4)
        st.markdown("### 临界贴现率 $\delta^*$（维持合作的最小耐心）")
        st.latex(r"\delta^* = \frac{R_D - R_C}{R_D - R_P} = \frac{%.4f - %.4f}{%.4f - %.4f} = %.4f" % (R_D, R_C, R_D, R_P, delta_critical))
        
        # 对比用户设定的δ
        st.markdown(f"### 设定的政府耐心 $\delta = {delta:.4f}$")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            pv_cooperate = round(R_C / (1 - delta) if delta < 1.0 else float('inf'), 4)
            st.metric("合作总收益（PV）", f"{pv_cooperate:.4f}")
        with col2:
            pv_defect = round(R_D + (delta * R_P) / (1 - delta) if delta < 1.0 else R_D, 4)
            st.metric("背叛总收益（PV）", f"{pv_defect:.4f}")
        
        # 结论
        if delta_critical >= 1.0:
            st.error(f"❌ 合作不可能：临界耐心 $\delta^* = {delta_critical:.4f} ≥ 1$")
            st.markdown(f"**原因**：背叛当期收益（{R_D:.4f}）远高于合作收益（{R_C:.4f}），即使完全重视未来也无法维持合作")
        elif delta >= delta_critical:
            st.success(f"✅ 合作稳定：$\delta = {delta:.4f} ≥ \delta^* = {delta_critical:.4f}$")
            st.markdown(f"**原因**：政府足够重视未来收益，惩罚的长期损失超过当期背叛诱惑")
        else:
            st.error(f"❌ 合作不稳定：$\delta = {delta:.4f} < \delta^* = {delta_critical:.4f}$")
            st.markdown(f"**原因**：政府缺乏耐心，更看重眼前背叛收益（{R_D:.4f}），博弈将崩溃回（tL, tL）囚徒困境")
    else:
        st.warning(f"⚠️ 动态分析条件不满足：需满足 $R_D > R_C > R_P$（当前 $R_D={R_D:.4f}, R_C={R_C:.4f}, R_P={R_P:.4f}$）")
        st.markdown("**说明**：不构成可分析的重复博弈囚徒困境，需调整参数（如提高β或降低α）")

st.markdown("---")
st.header("4. 核心结论")
st.markdown("""
1. **竞争强度影响因素**：资本流动性（β）和税收弹性（α）越大，税率竞争越激烈，政府越容易陷入（tL, tL）囚徒困境，税收收入降低
2. **合作稳定性条件**：重复博弈中，政府耐心（δ）需满足 $\delta ≥ \delta^*$ 才能维持长期合作；政府数量增加会提高 $\delta^*$，合作难度显著上升
3. **多策略均衡选择**：引入中税率（tM）后，博弈可能存在多重均衡，（tM, tM）是比（tL, tL）更优的均衡，需通过政策协调（如税收同盟）实现
4. **现实政策启示**：降低资本过度流动（如统一资本管制）、减弱税收弹性（如优化税收结构），可缓解税率竞争，提升政府整体税收收入
""")