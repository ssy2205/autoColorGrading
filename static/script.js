document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("upload-form");
  const originalImage = document.getElementById("original-image");
  const targetImage = document.getElementById("target-image");
  const resultImage = document.getElementById("result-image");
  const downloadButton = document.getElementById("download-button");

  form.addEventListener("submit", function (e) {
    e.preventDefault();

    const formData = new FormData(form);

    // Display original and target images immediately
    const inputFileInput = form.querySelector('input[name="input"]');
    if (inputFileInput && inputFileInput.files.length > 0) {
      originalImage.src = URL.createObjectURL(inputFileInput.files[0]);
    }

    const targetFileInput = form.querySelector('input[name="target"]');
    if (targetFileInput && targetFileInput.files.length > 0) {
      targetImage.src = URL.createObjectURL(targetFileInput.files[0]);
    }

    fetch("/api/process", {
      method: "POST",
      body: formData
    })
    .then((response) => {
      if (!response.ok) {
        return response.json().then(err => {
          throw new Error(err.error || "알 수 없는 오류가 발생했습니다.");
        });
      }
      return response.blob();
    })
    .then((blob) => {
      const resultUrl = URL.createObjectURL(blob);
      resultImage.src = resultUrl;
      downloadButton.style.display = "block";
      downloadButton.onclick = () => {
        const a = document.createElement("a");
        a.href = resultUrl;
        a.download = "corrected_image.jpg";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      };
    })
    .catch((error) => {
      alert("에러가 발생했습니다: " + error.message);
    });
  });
});