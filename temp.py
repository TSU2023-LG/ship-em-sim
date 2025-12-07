import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# 🌟 模拟一个超简单的“神经网络模型”（实际项目替换成你的模型）
@st.cache_resource
def load_model():
    """模拟模型：根据频率和角度生成电流分布（实际项目替换为你的模型加载）"""
    def model_predict(input_data):
        # input_data = [[freq, angle, ship_model]]，但这里我们只用前两个参数
        freq, angle = input_data[0][0], input_data[0][1]
        
        # 模拟电流分布：用正弦波+随机噪声（实际项目用你的模型预测）
        x = np.linspace(0, 10, 100)  # 100个面元
        return np.sin(x * freq) * (angle / 50) + np.random.normal(0, 0.2, 100)
    
    return model_predict

model = load_model()

# 🌟 模拟画图函数（实际项目替换为你的绘图逻辑）
def plot_current_distribution(current_distribution):
    """生成电流分布图"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(current_distribution, 'b-', linewidth=2)
    ax.set_title('Triangle Mesh Current Distribution', fontsize=14)
    ax.set_xlabel('Face Element Index', fontsize=12)
    ax.set_ylabel('Current (A)', fontsize=12)
    ax.grid(alpha=0.3)
    return fig

# 🌟 Streamlit界面
st.title("船舰电磁仿真交互系统 (示范版)")
st.markdown("### ✨ 甲方爸爸的专属交互界面（无真实模型，纯模拟演示）")

# 用户输入
freq = st.select_slider("频率 (GHz)", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], value=5)
angle = st.slider("入射角 (°)", 0, 90, 30, help="入射角越大，电流波动越明显")
ship_model = st.selectbox("船舰模型", ["航母", "驱逐舰", "护卫舰"], index=0)

# 生成按钮
if st.button("✨ 生成电流分布图", type="primary"):
    # 准备输入（简化：忽略船舰模型，实际项目需编码处理）
    input_data = np.array([[freq, angle]])
    
    # 模拟预测
    current_dist = model(input_data)
    
    # 显示结果
    st.subheader("📊 电流分布结果")
    st.success(f"已生成 {freq} GHz 频率、{angle}° 入射角的电流分布")
    
    # 画图
    fig = plot_current_distribution(current_dist)
    st.pyplot(fig)
    
    # 额外彩蛋：显示数据摘要
    st.metric("最大电流", f"{np.max(current_dist):.2f} A", f"变化率: {np.std(current_dist)*100:.1f}%")

# 💡 小贴士：甲方最爱看这个！
st.markdown("""
---
> 📌 **技术小注**：  
> 实际项目中：  
> 1️⃣ 用 `ship_model` 的 one-hot 编码替换当前的字符串  
> 2️⃣ 用 `model = load_model('your_model.h5')` 替换模拟模型  
> 3️⃣ 画图函数用 `plotly` 可实现3D交互（甲方看了直呼“高级”！）
""")