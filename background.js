let cameraTabId = null;

// --- State for Smooth Scrolling ---
let scrollIntervalId = null; 
let scrollStopTimer = null;

// --- Gesture to Browser Command Mapping ---
const gestureMap = {
    "open_palm": "pausePlay",
    "thumbs_up": "scrollUp",
    "pointing_right": "nextTab",
    "pointing_left": "prevTab",
    "fist": "voiceSearch",
    "l_shape": "openTab",
    "ok_sign": "closeTab",
    "thumbs_down": "scrollDown",
    "peace_sign": "toggleGestures"
};

// Cooldowns for discrete (non-scrolling) actions
const commandCooldowns = {
    pausePlay: 500,
    nextTab: 500,
    prevTab: 500,
    voiceSearch: 2000,
    openTab: 1000,
    closeTab: 1000,
    toggleGestures: 1000
};
let lastExecutionTimes = {};

// --- Listen for messages ---
chrome.runtime.onMessage.addListener((msg) => {
    if (!msg.action) return;

    switch (msg.action) {
        case "startCamera":
            if (cameraTabId) return;
            chrome.tabs.create({ url: chrome.runtime.getURL("cam.html") }, (tab) => { cameraTabId = tab.id; });
            break;
        case "stopCamera":
            if (cameraTabId) {
                chrome.tabs.remove(cameraTabId);
                cameraTabId = null;
            }
            break;
        case "gesturePrediction":
            handleGesturePrediction(msg.gesture);
            break;
    }
});

// --- Main Logic for Handling Gestures ---
function handleGesturePrediction(gesture) {
    if (scrollStopTimer) clearTimeout(scrollStopTimer);

    const command = gestureMap[gesture];
    
    if (command === 'scrollUp') {
        // --- CONTROL POINT 1: SCROLL SPEED ---
        // Change the `-20` to a larger negative number (e.g., -30) to scroll faster,
        // or a smaller negative number (e.g., -10) to scroll slower.
        startScrolling(-30); 
    } else if (command === 'scrollDown') {
        // Change the `20` to a larger positive number (e.g., 30) to scroll faster,
        // or a smaller positive number (e.g., 10) to scroll slower.
        startScrolling(30);
    } else {
        stopScrolling();
        if (command) {
            handleDiscreteCommand(command);
        }
    }
    
    // --- CONTROL POINT 2: AUTO-STOP DELAY ---
    // This is the grace period in milliseconds. If gesture detection flickers,
    // increasing this value (e.g., to 400) will make scrolling less likely to stop.
    scrollStopTimer = setTimeout(stopScrolling, 300);
}

// --- Helper functions for continuous scrolling ---
function startScrolling(scrollAmount) {
    if (scrollIntervalId) return; 

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (!tabs[0]) return;
        const tabId = tabs[0].id;
        
        scrollIntervalId = setInterval(() => {
            executeInTab(tabId, (amount) => {
                window.scrollBy(0, amount);
            }, [scrollAmount]);
        }, 30); 
    });
}

function stopScrolling() {
    if (scrollIntervalId) {
        clearInterval(scrollIntervalId);
        scrollIntervalId = null;
    }
}

// --- Handle only one-shot commands ---
function handleDiscreteCommand(cmd) {
    const now = Date.now();
    const cooldown = commandCooldowns[cmd] || 200;
    const lastExecution = lastExecutionTimes[cmd] || 0;

    if (now - lastExecution < cooldown) return;
    
    lastExecutionTimes[cmd] = now;
    console.log(`[Background] Executing discrete command: ${cmd}`);

    fetch("http://127.0.0.1:5000/status")
        .then(res => res.json())
        .then(data => {
            if (!data.gestures_enabled && cmd !== "toggleGestures") return;

            chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                if (!tabs[0]) return;
                const tabId = tabs[0].id;
                switch (cmd) {
                    case "openTab":
                        chrome.tabs.create({ url: "https://www.google.com" });
                        break;
                    case "closeTab":
                        chrome.tabs.remove(tabId);
                        break;
                    case "nextTab":
                        switchTab(tabId, 1);
                        break;
                    case "prevTab":
                        switchTab(tabId, -1);
                        break;
                    case "pausePlay":
                        executeInTab(tabId, () => {
                            const video = document.querySelector("video");
                            if (video) video.paused ? video.play() : video.pause();
                        });
                        break;
                    case "voiceSearch":
                        executeInTab(tabId, () => {
                            if (!window.location.hostname.includes("google.")) return;
                            const micBtn = document.querySelector('div[aria-label="Search by voice"]');
                            if (micBtn) micBtn.click();
                        });
                        break;
                    case "toggleGestures":
                         fetch("http://127.0.0.1:5000/toggle_gestures", { 
                             method: "POST",
                             headers: { 'Content-Type': 'application/json' },
                             body: JSON.stringify({ prediction: 'peace_sign' })
                         });
                        break;
                }
            });
        })
        .catch(err => console.error("[Background] Error fetching status:", err));
}

// --- Helper Functions ---
function switchTab(currentTabId, offset) {
    chrome.tabs.query({ currentWindow: true }, (tabs) => {
        const idx = tabs.findIndex(t => t.id === currentTabId);
        if (idx ===-1) return;
        const newIdx = (idx + offset + tabs.length) % tabs.length;
        chrome.tabs.update(tabs[newIdx].id, { active: true });
    });
}

function executeInTab(tabId, func, args = []) {
    chrome.scripting.executeScript({
        target: { tabId },
        func: func,
        args: args
    }, () => {
        if (chrome.runtime.lastError) {
            console.error("[Background] Script injection error:", chrome.runtime.lastError.message);
        }
    });
}

