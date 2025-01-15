document.addEventListener("DOMContentLoaded", () => {
    console.log("Page loaded successfully!");

    // Handle file upload and analysis
    const uploadBtn = document.querySelector("#upload-btn");
    if (uploadBtn) {
        uploadBtn.addEventListener("click", handleFileUpload);
    }
});

// Async function to handle file upload
async function handleFileUpload() {
    const fileInput = document.getElementById("file-input");
    const uploadStatus = document.getElementById("upload-status");
    const resultsContainer = document.getElementById("results-container");
    const visualizationSection = document.querySelector(".visualization-section");

    if (fileInput.files.length === 0) {
        uploadStatus.textContent = "Please select a file.";
        return;
    }

    const file = fileInput.files[0];
    console.log('fileInput: ',fileInput, 'file', file)
    const formData = new FormData();
    formData.append("file", file);
    console.log(file)
    console.log(formData)

    uploadStatus.textContent = "Uploading and analyzing...";

    try {
        const response = await fetch("/predict", {  // Fixed the endpoint here
            method: "POST",
            body: formData,
        });

        const result = await response.json();
        console.log(result)
        if (!response.ok) {
            uploadStatus.textContent = result.message || "Upload failed.";
            uploadStatus.style.color = "red";
            return;
        }

        uploadStatus.textContent = "Upload and analysis successful.";
        uploadStatus.style.color = "green";

        // Display results
        resultsContainer.innerHTML = `Classification: <strong>${result.class_name}</strong> Confidence: <strong>${result.confidence}</strong> Recyclable: <strong>${result.recyclable}</strong>`;  // Show classification label

        // Clear and populate visualizations
        visualizationSection.innerHTML = "";

        // Grad-CAM Heatmap
        const heatmapImg = document.createElement("img");
        heatmapImg.src = result.overlayed_img_url;  // Assuming the Flask server returns the path to the heatmap
        heatmapImg.alt = "Grad-CAM Heatmap";
        heatmapImg.className = "img-fluid mb-3";
        visualizationSection.appendChild(heatmapImg);

    } catch (error) {
        uploadStatus.textContent = "Error analyzing the file. Please try again.";
        uploadStatus.style.color = "red";
        console.error("Error:", error);
    }
}


