let __lastInputEl = null;

// textarea/input/contenteditable 포커스가 들어오면 기억
document.addEventListener("focusin", (e) => {
  const el = e.target;
  if (!el) return;

  const isTextInput =
    el.tagName === "TEXTAREA" ||
    (el.tagName === "INPUT" && (el.type === "text" || el.type === "search" || el.type === "email")) ||
    el.isContentEditable;

  if (isTextInput) __lastInputEl = el;
}, true);

function getTextFromEl(el) {
  if (!el) return "";
  if (el.tagName === "TEXTAREA") return el.value || "";
  if (el.tagName === "INPUT") return el.value || "";
  if (el.isContentEditable) return el.innerText || "";
  return "";
}

// 기존 getActiveText() 대신 이걸 사용
function getLastText() {
  return getTextFromEl(__lastInputEl);
}


function ensurePanel() {
  let el = document.getElementById("__guard_panel");
  if (el) return el;

  el = document.createElement("div");
  el.id = "__guard_panel";
  el.style.cssText = `
    position: fixed; right: 16px; bottom: 16px; z-index: 2147483647;
    width: 320px; padding: 12px; border-radius: 12px;
    background: #111; color: #fff; font-size: 13px; box-shadow: 0 10px 30px rgba(0,0,0,.35);
  `;
  el.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <div style="font-weight:700">Inline Similarity Guard</div>
      <button id="__guard_close" style="background:transparent;color:#fff;border:0;cursor:pointer;">✕</button>
    </div>
    <div id="__guard_body" style="margin-top:10px;line-height:1.4;">
      활성 입력창에서 텍스트를 가져와 검사합니다.
    </div>
    <div style="margin-top:10px;display:flex;gap:8px;">
      <button id="__guard_check" style="flex:1;padding:8px 10px;border-radius:10px;border:0;cursor:pointer;">검사</button>
      <button id="__guard_update" style="padding:8px 10px;border-radius:10px;border:0;cursor:pointer;">모델갱신</button>
    </div>
  `;
  document.body.appendChild(el);

  el.querySelector("#__guard_close").onclick = () => el.remove();
  el.querySelector("#__guard_update").onclick = () => {
    chrome.runtime.sendMessage({ type: "UPDATE_MODEL_NOW" }, (res) => {
      const body = el.querySelector("#__guard_body");
      if (!res || !res.ok) body.textContent = "모델 갱신 실패";
      else body.textContent = res.updated ? `모델 갱신 완료: ${res.version}` : `이미 최신: ${res.version}`;
    });
  };

    el.querySelector("#__guard_check").onclick = () => {
    const text = getLastText();   // ✅ activeElement 대신 마지막 입력창
    const body = el.querySelector("#__guard_body");
    if (!text) {
        body.textContent = "최근 포커스된 입력창이 없습니다. (textarea/contenteditable을 먼저 클릭하세요)";
        return;
    }
    chrome.runtime.sendMessage({ type: "CHECK_TEXT", text }, (res) => {
        if (!res || !res.ok) {
        body.textContent = "분석 실패: 모델 준비 안 됨 또는 오류";
        return;
        }
        const lvl = res.risk_level;
        const score = (res.risk_score || 0).toFixed(3);
        const top = (res.top_matches || []).map(x => `- ${x.class} (${x.sim.toFixed(3)}) : ${x.title}`).join("\n");
        body.textContent = `Risk: ${lvl} / score=${score}\n\nTop matches:\n${top}`;
    });
    };

  return el;
}

// 단축키: Alt+Shift+G 패널 토글
window.addEventListener("keydown", (e) => {
  if (e.altKey && e.shiftKey && e.code === "KeyG") {
    const p = document.getElementById("__guard_panel");
    if (p) p.remove();
    else ensurePanel();
  }
});

// Alt+Shift+U = 모델 업데이트 강제 실행
// Alt+Shift+C = 최근 포커스 입력창 텍스트 검사
window.addEventListener("keydown", (e) => {
  if (e.altKey && e.shiftKey && e.code === "KeyU") {
    chrome.runtime.sendMessage({ type: "UPDATE_MODEL_NOW" }, (res) => {
      console.log("[Guard] UPDATE_MODEL_NOW result:", res);
      alert(JSON.stringify(res, null, 2));
    });
  }
  if (e.altKey && e.shiftKey && e.code === "KeyC") {
    // 여기서는 activeElement 대신 lastInput을 쓰는 버전이 베스트지만,
    // 최소 확인용으로 일단 activeElement로도 OK
    const el = document.activeElement;
    let text = "";
    if (el && el.tagName === "TEXTAREA") text = el.value || "";
    else if (el && el.tagName === "INPUT") text = el.value || "";
    else if (el && el.isContentEditable) text = el.innerText || "";

    chrome.runtime.sendMessage({ type: "CHECK_TEXT", text }, (res) => {
      console.log("[Guard] CHECK_TEXT result:", res);
      alert(JSON.stringify(res, null, 2));
    });
  }
});
