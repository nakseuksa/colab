function ClickConnect() {
    console.log("💓 [Colab Keep-Alive] 触发前端防断连心跳 - " + new Date().toLocaleTimeString());
    // 点击顶部连接状态按钮两次（展开再收起），向 Colab 前端派发真实 DOM 交互事件
    const connectBtn = document.querySelector("colab-connect-button") || document.querySelector("#top-toolbar colab-connect-button");
    if (connectBtn && connectBtn.shadowRoot) {
        const btn = connectBtn.shadowRoot.querySelector("#connect");
        if (btn) { btn.click(); setTimeout(() => btn.click(), 500); }
    }
    // 自动点击可能弹出的“您仍在吗？”确认框
    const okBtn = document.querySelector("mwc-button[slot='primaryAction']") || document.querySelector("paper-button#ok");