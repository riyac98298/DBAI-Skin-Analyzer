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
})();
