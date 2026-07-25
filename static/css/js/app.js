document.addEventListener("DOMContentLoaded", () => {
    const themeToggle = document.getElementById("theme-toggle");
    const html = document.documentElement;

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            if (html.getAttribute("data-theme") === "light") {
                html.setAttribute("data-theme", "dark");
                themeToggle.textContent = "🌙";
            } else {
                html.setAttribute("data-theme", "light");
                themeToggle.textContent = "☀️";
            }
        });
    }

    const imageInput = document.getElementById("image");
    const previewPanel = document.getElementById("preview-panel");

    if (imageInput && previewPanel) {
        imageInput.addEventListener("change", () => {
            previewPanel.innerHTML = "";

            if (imageInput.files.length > 0) {
                const file = imageInput.files[0];

                const img = document.createElement("img");
                img.src = URL.createObjectURL(file);
                img.style.maxWidth = "250px";
                img.style.borderRadius = "12px";

                previewPanel.hidden = false;
                previewPanel.appendChild(img);
            }
        });
    }
});