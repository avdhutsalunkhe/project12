const BASE_URL = 'https://shl-assessment-api-u2mr.onrender.com';
const CHAT_ENDPOINT = `${BASE_URL}/api/v1/chat`;
const HEALTH_ENDPOINT = `${BASE_URL}/api/v1/health`;

// State to maintain entire conversation
let chatHistory = [];

// DOM Elements
const chatWrapper = document.getElementById('chat-wrapper');
const chatContainer = document.getElementById('chat-container');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const typingIndicator = document.getElementById('typing-indicator');
const wakeupMsg = document.getElementById('wakeup-msg');
const statusDot = document.querySelector('.status-dot');
const statusText = document.querySelector('.status-text');

// Initialize API status check on load
document.addEventListener('DOMContentLoaded', checkServerStatus);

// Form submit handler
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;

    // 1. Add User Message to UI and History
    appendMessage('user', text);
    chatHistory.push({ role: 'user', content: text });
    
    // 2. Clear input and disable UI
    userInput.value = '';
    setLoadingState(true);

    // 3. Setup timeout for cold start warning (Render free tier wakes up slowly)
    const coldStartTimer = setTimeout(() => {
        wakeupMsg.classList.remove('hidden');
    }, 5000); // show after 5s

    try {
        // 4. Call Backend API
        const response = await fetch(CHAT_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: chatHistory })
        });

        clearTimeout(coldStartTimer);

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        // Render's Free Tier intercepts requests during cold start and returns an HTML loading page!
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("text/html")) {
            throw new Error("RENDER_COLD_START");
        }

        const data = await response.json();
        
        // 5. Add Assistant Message to UI and History
        appendMessage('assistant', data.reply, data.recommendations);
        chatHistory.push({ role: 'assistant', content: data.reply });

    } catch (error) {
        clearTimeout(coldStartTimer);
        console.error("Chat Error:", error);
        
        if (error.message === "RENDER_COLD_START") {
            appendMessage('assistant', "I am currently waking up from sleep mode. Please wait about 60 seconds and try your message again!");
        } else {
            appendMessage('assistant', "Sorry, I'm having trouble connecting to the server. Please check your connection and try again.");
        }
        
        // Pop the user message so they don't lose it from history, and can retry
        chatHistory.pop();
    } finally {
        setLoadingState(false);
    }
});

function appendMessage(role, text, recommendations = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    let htmlContent = `<div class="bubble">${escapeHTML(text)}</div>`;

    if (recommendations && recommendations.length > 0) {
        htmlContent += `<div class="recommendations">`;
        recommendations.forEach(rec => {
            htmlContent += `
                <div class="rec-card">
                    <div class="rec-name">${escapeHTML(rec.name)}</div>
                    <div class="rec-type">${escapeHTML(rec.test_type || 'Assessment')}</div>
                    <a href="${escapeHTML(rec.url)}" target="_blank" rel="noopener noreferrer" class="rec-link">View Test &rarr;</a>
                </div>
            `;
        });
        htmlContent += `</div>`;
    }

    msgDiv.innerHTML = htmlContent;
    chatWrapper.appendChild(msgDiv);
    scrollToBottom();
}

function setLoadingState(isLoading) {
    userInput.disabled = isLoading;
    sendBtn.disabled = isLoading;
    
    if (isLoading) {
        typingIndicator.classList.remove('hidden');
        wakeupMsg.classList.add('hidden'); // hidden initially
    } else {
        typingIndicator.classList.add('hidden');
    }
    scrollToBottom();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Utility to prevent XSS
function escapeHTML(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.innerText = str;
    return div.innerHTML;
}

async function checkServerStatus() {
    try {
        const res = await fetch(HEALTH_ENDPOINT);
        if (res.ok) {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'API Online';
        } else {
            throw new Error('Not ok');
        }
    } catch (err) {
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'API Offline';
    }
}
