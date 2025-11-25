// Image preview + loading spinner
(function () {
  const fileInput = document.getElementById("file-input");
  const preview = document.getElementById("preview");
  const previewImg = document.getElementById("preview-img");
  const form = document.getElementById("analyze-form");
  const btn = document.getElementById("analyze-btn");
  const spinner = document.getElementById("spinner");
  const msg = document.getElementById("form-msg");

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      const f = fileInput.files?.[0];
      if (!f) return (preview.classList.add("hidden"));
      const url = URL.createObjectURL(f);
      previewImg.src = url;
      preview.classList.remove("hidden");
      msg.textContent = `${f.name} • ${(f.size/1024/1024).toFixed(2)} MB`;
    });
  }

  if (form) {
    form.addEventListener("submit", () => {
      spinner?.classList.remove("hidden");
      btn?.setAttribute("disabled", "true");
      btn?.classList.add("opacity-80");
    });
  }
})
// =====================
// Save & Track + Compare
// =====================
(function () {
  const root = document.querySelector("[data-aeye-result]");
  if (!root) return;

  const key = "aeye-history"; // localStorage key
  const imgUrl = root.dataset.img;
  const probs = {
    dark: parseFloat(root.dataset.dark || "0"),
    acne: parseFloat(root.dataset.acne || "0"),
  };

  const saveToggle = document.getElementById("save-track");
  const compare = document.getElementById("compare-block");
  const prevImg = document.getElementById("compare-prev-img");
  const prevMeta = document.getElementById("compare-prev-meta");

  function loadHistory() {
    try { return JSON.parse(localStorage.getItem(key) || "[]"); }
    catch { return []; }
  }
  function saveHistory(list) {
    localStorage.setItem(key, JSON.stringify(list));
  }
  function tsToText(ts) {
    const d = new Date(ts);
    return d.toLocaleString();
  }

  // If we already have a previous save, show compare
  const history = loadHistory();
  const prev = history.length ? history[history.length - 1] : null;
  if (prev && compare && prevImg && prevMeta) {
    prevImg.src = prev.img;
    prevMeta.textContent = `Saved on ${tsToText(prev.ts)} — dark: ${(prev.probs.dark*100).toFixed(0)}%, acne: ${(prev.probs.acne*100).toFixed(0)}%`;
    compare.classList.remove("hidden");
  }

  // Handle toggle: when ON, save current result to history
  if (saveToggle) {
    saveToggle.addEventListener("change", (e) => {
      if (!e.target.checked) return;
      const list = loadHistory();
      list.push({ img: imgUrl, probs, ts: Date.now() });
      saveHistory(list);
      // Show compare immediately if not already showing
      if (compare && compare.classList.contains("hidden") && prevImg && prevMeta) {
        prevImg.src = imgUrl;
        prevMeta.textContent = `Saved on ${tsToText(list[list.length - 1].ts)} — dark: ${(probs.dark*100).toFixed(0)}%, acne: ${(probs.acne*100).toFixed(0)}%`;
        compare.classList.remove("hidden");
      }
    });
  }
})();


