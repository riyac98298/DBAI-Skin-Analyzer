document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("analysis-form");
  const formContainer = document.getElementById("form-container");
  const resultsContainer = document.getElementById("results-container");
  const analysisOutput = document.getElementById("analysis-output");
  const recommendationOutput = document.getElementById("recommendation-output");
  const restartBtn = document.getElementById("restart-btn");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    // Construct form data
    const formData = new FormData(form);
    try {
      // Send request to the backend
      const response = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const err = await response.json();
        alert(err.error || "An error occurred");
        return;
      }
      const data = await response.json();
      displayResults(data);
    } catch (err) {
      console.error(err);
      alert("Failed to analyze the image. Please try again.");
    }
  });

  restartBtn.addEventListener("click", () => {
    // Reset form and toggle visibility
    form.reset();
    analysisOutput.innerHTML = "";
    recommendationOutput.innerHTML = "";
    resultsContainer.classList.add("hidden");
    formContainer.classList.remove("hidden");
  });

  function displayResults(data) {
    // Hide the form and show results
    formContainer.classList.add("hidden");
    resultsContainer.classList.remove("hidden");

    // Display annotated image
    const img = document.createElement("img");
    img.src = `data:image/jpeg;base64,${data.annotatedImage}`;
    img.alt = "Annotated analysis";
    img.classList.add("w-full", "mb-4", "rounded");
    analysisOutput.appendChild(img);

    // Display probabilities and detected issues
    const probsContainer = document.createElement("div");
    probsContainer.classList.add("mb-4");
    const probsHeading = document.createElement("h3");
    probsHeading.classList.add("text-lg", "font-semibold", "mb-2");
    probsHeading.textContent = "Detected issues";
    probsContainer.appendChild(probsHeading);

    // List detected issues with probabilities
    if (data.detected.length === 0) {
      const p = document.createElement("p");
      p.textContent = "No issues detected above the threshold.";
      probsContainer.appendChild(p);
    } else {
      const ul = document.createElement("ul");
      ul.classList.add("list-disc", "pl-5", "space-y-1");
      data.detected.forEach((issue) => {
        const li = document.createElement("li");
        const prob = data.probabilities[issue];
        li.textContent = `${issue.replace(/_/g, " ")}: ${(prob * 100).toFixed(1)}%`;
        ul.appendChild(li);
      });
      probsContainer.appendChild(ul);
    }
    analysisOutput.appendChild(probsContainer);

    // Display product recommendations
    recommendationOutput.innerHTML = "";
    if (data.recommendations && data.recommendations.length > 0) {
      const recHeading = document.createElement("h3");
      recHeading.classList.add("text-lg", "font-semibold", "mb-2");
      recHeading.textContent = "Recommended products";
      recommendationOutput.appendChild(recHeading);
      data.recommendations.forEach((rec) => {
        const card = document.createElement("div");
        card.classList.add(
          "border",
          "border-gray-200",
          "rounded",
          "p-3",
          "mb-3",
          "shadow-sm",
          "bg-gray-50"
        );
        const title = document.createElement("h4");
        title.classList.add("font-medium");
        title.textContent = rec.product_name;
        const desc = document.createElement("p");
        desc.classList.add("text-sm", "mt-1");
        desc.textContent = rec.description;
        card.appendChild(title);
        card.appendChild(desc);
        recommendationOutput.appendChild(card);
      });
    }
  }
});