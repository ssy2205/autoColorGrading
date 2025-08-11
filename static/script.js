document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("upload-form");
  const originalImage = document.getElementById("original-image");
  const targetImage = document.getElementById("target-image");
  const resultImage = document.getElementById("result-image");
  const downloadButton = document.getElementById("download-button");
  const loadingMessage = document.getElementById("loading-message");

  // Function to resize an image
  function resizeImage(file, maxSize) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
          let width = img.width;
          let height = img.height;

          if (width > height) {
            if (width > maxSize) {
              height *= maxSize / width;
              width = maxSize;
            }
          } else {
            if (height > maxSize) {
              width *= maxSize / height;
              height = maxSize;
            }
          }

          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0, width, height);

          canvas.toBlob(resolve, file.type);
        };
        img.src = e.target.result;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    const inputFileInput = form.querySelector('input[name="input"]');
    const targetFileInput = form.querySelector('input[name="target"]');

    if (!inputFileInput.files[0] || !targetFileInput.files[0]) {
        alert("Please select both input and target images.");
        return;
    }

    // Show loading message
    loadingMessage.style.display = "block";

    try {
        // Create FormData from the form
        const formData = new FormData(form);

        // Resize images
        const resizedInput = await resizeImage(inputFileInput.files[0], 1024);
        const resizedTarget = await resizeImage(targetFileInput.files[0], 1024);

        // Set the resized images in the FormData
        formData.set("input", resizedInput, inputFileInput.files[0].name);
        formData.set("target", resizedTarget, targetFileInput.files[0].name);

        // Display original and target images immediately
        originalImage.src = URL.createObjectURL(resizedInput);
        targetImage.src = URL.createObjectURL(resizedTarget);

        const response = await fetch("https://autocolorgrading.onrender.com/api/process", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "알 수 없는 오류가 발생했습니다.");
        }

        const blob = await response.blob();
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
        alert("이미지 처리가 완료되었습니다.");

    } catch (error) {
        alert("에러가 발생했습니다: " + error.message);
    } finally {
        // Hide loading message
        loadingMessage.style.display = "none";
    }
  });
});