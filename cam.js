(async () => {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    try {
        // Start the webcam
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        console.log("Camera started successfully!");

        // Capture frames and send to Flask every 200ms
        setInterval(async () => {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg"));
            
            // --- NEW: Convert blob to base64 using FileReader for better performance ---
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = async () => {
                const base64data = reader.result;

                try {
                    // Send frame to Flask and wait for the response
                    const response = await fetch("http://127.0.0.1:5000/frame", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ image: base64data })
                    });

                    const result = await response.json();

                    // --- NEW: Check the response and send it to the background script ---
                    if (result.success && result.command !== "no_hand") {
                        // Forward the prediction to background.js
                        chrome.runtime.sendMessage({
                            action: "gesturePrediction",
                            gesture: result.command
                        });
                    }
                } catch (error) {
                    console.error("Error sending frame or receiving prediction:", error);
                }
            };
        }, 200); // You can adjust this interval for performance

    } catch (err) {
        console.error("Camera error:", err);
    }
})();
