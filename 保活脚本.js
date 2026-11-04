function keepAlive() {
    console.log("🟢 [Colab Keep-Alive] 正在触发活跃点击...");
    
    // 尝试寻找 Colab 的连接按钮并点击
    const connectButton = document.querySelector("colab-connect-button");
    if (connectButton && connectButton.shadowRoot) {
        const btn = connectButton.shadowRoot.querySelector("#connect");
        if (btn) {
            btn.click();
            return;
        }
    }
    
    // 备用方案：点击顶部工具栏的连接状态
    const toolbarBtn = document.querySelector("#top-toolbar > colab-connect-button");
    if (toolbarBtn) {
        toolbarBtn.click();
    }
}

// 每 60,000 毫秒（60秒）执行一次
const keepAliveInterval = setInterval(keepAlive, 60000);
console.log("✅ Colab 保活脚本已启动！如果要停止，请在控制台输入: clearInterval(" + keepAliveInterval + ")");