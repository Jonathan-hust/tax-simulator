import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------
# 1. 核心计算引擎 (基于您提供的博弈论作业 .docx)
# ---------------------------------------------------------------------

def calculate_revenue(t_i, t_j, alpha, beta):
    """
    根据您文件中的收益函数计算政府i的税收收入 (R_i)
    R_i = t_i * K_i' = t_i * [ (1/2) + β(t_j - t_i) ] * [ 1 - α * (t_i + t_j) / 2 ]
    """
    try:
        # K_i (资本份额)
        K_i_share = 0.5 + beta * (t_j - t_i)
        
        # K_total (总资本)
        K_total = 1 - alpha * ((t_i + t_j) / 2)
        
        # K_i' (实际资本)
        K_i_actual = K_i_share * K_total
        
        # R_i (税收收入)
        R_i = t_i * K_i_actual
        
        # 确保收益不为负（在某些极端参数下可能发生）
        if R_i < 0:
            return 0.0
        return R_i
    
    except Exception as e:
        st.error(f"公式计算出错: {e}")
        return 0.0

# ---------------------------------------------------------------------
# 2. Streamlit 网页界面
# ---------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("税收竞争博弈模拟器 (V2 - 基于经济函数)")
st.markdown("---")

# ---------------------------------------------------------------------
# 3. 侧边栏：参数滑块
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ 1. 核心经济参数")
st.sidebar.markdown("调整经济学参数，观察均衡变化")

# α 和 β 是核心变量 
alpha = st.sidebar.slider("α (税收弹性系数)", 0.0, 3.0, 1.0, 0.1,
                          help="衡量总资本量对平均税率的敏感程度。α 越大，税率升高导致的总资本流失越多。[cite: 1599]")
beta = st.sidebar.slider("β (资本流动性系数)", 0.0, 3.0, 2.0, 0.1,
                         help="衡量资本在两地间对'税率差'的敏感程度。β 越大，资本越容易从高税率地区流向低税率地区。[cite: 1598]")

st.sidebar.header("⚙️ 2. 策略与耐心参数")
t_H = st.sidebar.slider("高税率 (tH - 合作策略)", 0.0, 1.0, 0.4, 0.01,
                        help="双方约定共同执行的高税率。")
t_L = st.sidebar.slider("低税率 (tL - 背叛/惩罚策略)", 0.0, 1.0, 0.2, 0.01,
                        help="单方面背叛或双方互害时选择的低税率。")
delta = st.sidebar.slider("δ (政府的耐心 / 贴现率)", 0.0, 1.0, 0.5, 0.01,
                          help="对未来收益的重视程度。越接近1，越有耐心。")

# ---------------------------------------------------------------------
# 4. 主页面：显示公式和计算结果
# ---------------------------------------------------------------------

st.header("1. 核心收益函数（模型设定）")
st.markdown("本模拟器根据您提供的收益函数 [cite: 1603] 实时计算收益。")
st.latex(r"K_i' = \left[ \frac{1}{2} + \beta (t_j - t_i) \right] \times \left[ 1 - \alpha \left( \frac{t_i + t_j}{2} \right) \right]")
st.latex(r"R_i (\text{政府i的收益}) = t_i \times K_i'")

st.markdown("---")

# ---------------------------------------------------------------------
# 5. 静态博弈分析
# ---------------------------------------------------------------------
st.header("2. 静态博弈分析（一次性博弈）")

# (Cooperate, Cooperate) -> (tH, tH)
R_C = calculate_revenue(t_H, t_H, alpha, beta)

# (Defect, Defect) -> (tL, tL)
R_P = calculate_revenue(t_L, t_L, alpha, beta)

# (Defect, Cooperate) -> (tL, tH)
# 我(A)背叛 (tL)，对手(B)合作 (tH)
# 我的收益 R_D (Temptation)
R_D = calculate_revenue(t_L, t_H, alpha, beta)
# 对手的收益 R_S (Sucker)
R_S = calculate_revenue(t_H, t_L, alpha, beta)

# 构建支付矩阵
st.subheader("支付矩阵 (政府A, 政府B)")
matrix_data = {
    f'tH (合作, {t_H*100}%)': [f"({R_C:.4f}, {R_C:.4f})", f"({R_S:.4f}, {R_D:.4f})"],
    f'tL (背叛, {t_L*100}%)': [f"({R_D:.4f}, {R_S:.4f})", f"({R_P:.4f}, {R_P:.4f})"]
}
matrix_index = [f'tH (合作, {t_H*100}%)', f'tL (背叛, {t_L*100}%)']
df_matrix = pd.DataFrame(matrix_data, index=matrix_index)
st.dataframe(df_matrix, use_container_width=True)

# 静态均衡分析
st.subheader("纳什均衡分析")
if (R_D > R_C) and (R_P > R_S):
    st.error(f"**占优策略均衡 (囚徒困境):**")
    st.markdown(f"双方的占优策略都是选择 **tL (背叛)**。")
    st.markdown(f"唯一的纳什均衡是 **(tL, tL)**，双方获得 `({R_P:.4f}, {R_P:.4f})`。")
    st.warning(f"**困境：** 此均衡收益 `({R_P:.4f})` 劣于合作时的收益 `({R_C:.4f})`。")
else:
    st.info("根据当前参数，此博弈**不是**一个标准的囚徒困境。请尝试调整参数（例如，增加 `β` 来提高背叛诱惑）。")

st.markdown("---")

# ---------------------------------------------------------------------
# 6. 动态博弈分析
# ---------------------------------------------------------------------
st.header("3. 动态博弈分析（无限重复）")
st.markdown("我们使用“冷酷触发策略”：双方开始时都选择 tH (合作)。一旦有人背叛 (选择 tL)，双方将从下期开始永远选择 tL (惩罚)。")

# 检查是否满足分析前提
if not (R_D > R_C > R_P > R_S):
    st.warning(f"**分析中止：** 当前参数 (R_C={R_C:.4f}, R_D={R_D:.4f}, R_P={R_P:.4f}) 不构成能用触发策略解决的囚徒困境。")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("A. 坚持合作的收益 (PV)")
        st.latex(r"PV_{\text{合作}} = R_C + \delta R_C + \delta^2 R_C + \dots = \frac{R_C}{1 - \delta}")
        pv_cooperate = R_C / (1 - delta) if delta < 1.0 else float('inf')
        st.metric("合作的总收益 (现值)", f"{pv_cooperate:.4f}")

    with col2:
        st.subheader("B. 当期背叛的收益 (PV)")
        st.latex(r"PV_{\text{背叛}} = R_D + \delta R_P + \delta^2 R_P + \dots = R_D + \frac{\delta R_P}{1 - \delta}")
        pv_defect = R_D + (delta * R_P) / (1 - delta) if delta < 1.0 else R_D
        st.metric("背叛的总收益 (现值)", f"{pv_defect:.4f}")
    
    st.markdown("---")
    
    # 5. 均衡结论
    st.subheader("分析结论：合作是否可能？")
    st.markdown(r"合作稳定的条件是 $PV_{\text{合作}} \ge PV_{\text{背叛}}$。这需要政府的耐心 $\delta$ 必须大于一个临界值 $\delta^*$。")

    # 计算临界折现因子
    try:
        delta_critical = (R_D - R_C) / (R_D - R_P)
        st.markdown("临界值 $\delta^*$ 的计算公式如下 [cite: 1648]：")
        st.latex(r"\delta^* = \frac{R_D - R_C}{R_D - R_P} = \frac{%.4f - %.4f}{%.4f - %.4f} = \mathbf{%.4f}" % (R_D, R_C, R_D, R_P, delta_critical))
    
        # 最终判定
        st.markdown(f"**您设定的政府耐心 $\delta = {delta:.2f}$**")
        if delta_critical > 1.0:
             st.error(f"**结论：合作不可能 ($\delta^* > 1$)**")
             st.markdown(f"在此参数下，背叛的诱惑 (`{R_D:.4f}`) 相比合作 (`{R_C:.4f}`) 实在太高，即使是惩罚 (`{R_P:.4f}`) 也无法阻止背叛。")
        elif delta >= delta_critical:
            st.success(f"**结论：合作是稳定的 (因为 $\delta \ge \delta^*$)**")
            st.markdown(f"您设定的政府**足够有耐心** ($\delta={delta:.2f}$)，对未来惩罚的恐惧超过了当期背叛的诱惑。")
        else:
            st.error(f"**结论：合作不稳定 (因为 $\delta < \delta^*$)**")
            st.markdown(f"您设定的政府**缺乏耐心** ($\delta={delta:.2f}$)，更看重眼前的利益（背叛）。触发策略无效，博弈将崩溃回 (tL, tL) 的囚徒困境。")

    except ZeroDivisionError:
        st.error("计算错误：背叛收益 (R_D) 和 惩罚收益 (R_P) 相等，无法计算临界值。")