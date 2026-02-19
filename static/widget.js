(function () {
  const script = document.currentScript;

  // Default: /chat si tu ne passes pas data-chat-url
  const chatUrl = script.getAttribute("data-chat-url") || "/chat";
  const botId = script.getAttribute("data-bot-id") || "default";
  const title = script.getAttribute("data-title") || "Assistant";
  const position = script.getAttribute("data-position") || "right";
  const primary = script.getAttribute("data-primary") || "#111111";

  const right = position !== "left";
  const x = right ? "right:20px;" : "left:20px;";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.setAttribute("aria-label", "Open chat");
  btn.textContent = "💬";
  btn.style.cssText = `
    position:fixed; ${x} bottom:20px; z-index:2147483647;
    width:52px; height:52px; border-radius:999px;
    border:none; cursor:pointer;
    background:${primary}; color:#fff;
    box-shadow:0 10px 28px rgba(0,0,0,.25);
    font-size:20px;
  `;

  const panel = document.createElement("div");
  panel.style.cssText = `
    position:fixed; ${x} bottom:84px;
    width:min(380px, calc(100vw - 40px));
    height:min(560px, calc(100vh - 140px));
    z-index:2147483647;
    display:none;
    border-radius:16px;
    overflow:hidden;
    background:#fff;
    box-shadow:0 16px 60px rgba(0,0,0,.28);
  `;

  const header = document.createElement("div");
  header.style.cssText = `
    height:48px; display:flex; align-items:center; justify-content:space-between;
    padding:0 12px; background:${primary}; color:#fff; font:14px system-ui;
  `;

  const left = document.createElement("div");
  left.style.cssText = "display:flex;gap:8px;align-items:center;";
  left.innerHTML = `<div style="width:10px;height:10px;border-radius:999px;background:#35d07f;"></div><div>${title}</div>`;

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.setAttribute("aria-label", "Close");
  closeBtn.textContent = "✕";
  closeBtn.style.cssText = "background:transparent;border:none;color:#fff;cursor:pointer;font-size:18px;";

  header.appendChild(left);
  header.appendChild(closeBtn);

  const iframe = document.createElement("iframe");

  const url = new URL(chatUrl, window.location.href);
  url.searchParams.set("botId", botId);
  url.searchParams.set("title", title);

  iframe.src = url.toString();
  iframe.style.cssText = "width:100%; height:calc(100% - 48px); border:0;";

  panel.appendChild(header);
  panel.appendChild(iframe);

  document.body.appendChild(btn);
  document.body.appendChild(panel);

  function toggle(open) {
    const isOpen = panel.style.display === "block";
    const next = (typeof open === "boolean") ? open : !isOpen;
    panel.style.display = next ? "block" : "none";
  }

  btn.addEventListener("click", () => toggle());
  closeBtn.addEventListener("click", () => toggle(false));
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") toggle(false); });
})();
