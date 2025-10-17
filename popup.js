document.addEventListener("DOMContentLoaded", async () => {
    const enableCameraBox = document.getElementById("enableCamera");
    const enableGesturesBtn = document.getElementById("gesturesBtn");

    // --- Load initial state from Flask ---
    await refreshStatus();

    // --- Camera checkbox ---
    enableCameraBox.addEventListener("change", async () => {
        const enabled = enableCameraBox.checked;
        try {
            await fetch("http://127.0.0.1:5000/toggle", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ enabled })
            });

            // Send message to background to open/close camera tab
            chrome.runtime.sendMessage({
                action: enabled ? "startCamera" : "stopCamera"
            });

            await refreshStatus();
        } catch (err) {
            console.error("Error updating camera toggle:", err);
        }
    });

    // --- Gestures toggle button ---
    enableGesturesBtn.addEventListener("click", async () => {
        try {
            const res = await fetch("http://127.0.0.1:5000/status");
            const data = await res.json();
            const newState = !data.gestures_enabled;

            await fetch("http://127.0.0.1:5000/toggle_gestures", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ enabled: newState })
            });

            updateStatusUI({ gestures_enabled: newState });
        } catch (err) {
            console.error("Error toggling gestures:", err);
        }
    });

    // --- Gesture buttons ---
    document.querySelectorAll("button[data-cmd]").forEach(btn => {
        btn.addEventListener("click", () => {
            const cmd = btn.dataset.cmd;
            chrome.runtime.sendMessage({ action: "simulateCommand", command: cmd });
        });
    });

    // --- Poll every 2s to update background color ---
    setInterval(refreshStatus, 2000);
});

// --- Helpers ---
async function refreshStatus() {
    try {
        const res = await fetch("http://127.0.0.1:5000/status");
        const data = await res.json();

        updateStatusUI(data);

        // update camera checkbox
        const enableCameraBox = document.getElementById("enableCamera");
        enableCameraBox.checked = data.camera_enabled;
    } catch (err) {
        console.error("Error fetching status:", err);
        document.body.style.backgroundColor = "gray";
    }
}

function updateStatusUI(data) {
    document.body.style.backgroundColor = data.gestures_enabled ? "green" : "red";
}
