/**
 * AI Classification Interface
 *
 * Handles real-time AI-powered exercise classification from the backend.
 */

export class AIInterface {
    constructor(socket) {
        this.socket = socket;
        this.enabled = false;
        this.currentPrediction = null;
        this.bufferStatus = null;
        this.available = false;

        this.setupSocketListeners();
        this.createUI();
    }

    setupSocketListeners() {
        // AI classification results
        this.socket.on('ai_classification', (data) => {
            this.handleClassification(data);
        });

        // AI buffer status updates
        this.socket.on('ai_buffer_status', (status) => {
            this.updateBufferStatus(status);
        });

        // AI enabled/disabled status
        this.socket.on('ai_status', (status) => {
            this.updateAIStatus(status);
        });

        // Request initial AI status
        this.socket.emit('get_ai_status');
    }

    createUI() {
        // Find dashboard + main visualization panel
        const dashboard = document.querySelector('.dashboard');
        if (!dashboard) return;
        const mainPanel = dashboard.querySelector('.main-panel');
        if (!mainPanel) return;

        // Create AI control panel
        const aiPanel = document.createElement('div');
        aiPanel.className = 'ai-panel';
        aiPanel.innerHTML = `
            <div class="ai-header">
                <h3>🤖 AI Exercise Classification</h3>
                <button id="ai-toggle-btn" class="btn-secondary">Enable AI</button>
            </div>

            <div id="ai-status" class="ai-status">
                <div class="status-indicator">
                    <span class="dot"></span>
                    <span id="ai-status-text">AI Disabled</span>
                </div>
            </div>

            <div id="ai-buffer-status" class="buffer-status" style="display: none;">
                <div class="buffer-label">Data Buffer</div>
                <div class="progress-bar">
                    <div id="buffer-progress" class="progress-fill"></div>
                </div>
                <div id="buffer-text" class="buffer-text">0 / 150 samples</div>
            </div>

            <div id="ai-prediction" class="ai-prediction" style="display: none;">
                <div class="prediction-header">
                    <h4 id="predicted-exercise">-</h4>
                    <span id="prediction-confidence" class="confidence-badge">-</span>
                </div>

                <div class="probabilities">
                    <div class="prob-label">All Probabilities:</div>
                    <div id="prob-bars" class="prob-bars"></div>
                </div>

                <div id="ai-warning" class="ai-warning" style="display: none;">
                    ⚠️ Using demo model - predictions are random
                </div>
            </div>
        `;

        // Place AI panel within shared AI/Coach row
        const row = this.getOrCreateAIRow(mainPanel);
        row.appendChild(aiPanel);

        // Setup toggle button
        const toggleBtn = document.getElementById('ai-toggle-btn');
        toggleBtn.addEventListener('click', () => this.toggleAI());

        // Add CSS styles
        this.injectStyles();
    }

    toggleAI() {
        if (this.enabled) {
            this.socket.emit('disable_ai');
        } else {
            if (!this.available) {
                this.showWarning('AI classifier not available. Train the model to enable AI.');
                return;
            }
            this.socket.emit('enable_ai');
        }
    }

    updateAIStatus(status) {
        this.enabled = status.enabled;
        this.available = Boolean(status.available);

        const statusDot = document.querySelector('.status-indicator .dot');
        const statusText = document.getElementById('ai-status-text');
        const toggleBtn = document.getElementById('ai-toggle-btn');
        const bufferStatus = document.getElementById('ai-buffer-status');
        const predictionPanel = document.getElementById('ai-prediction');

        if (!statusDot || !statusText || !toggleBtn || !bufferStatus || !predictionPanel) {
            return;
        }

        if (status.enabled) {
            statusDot.className = 'dot active';
            statusText.textContent = 'AI Active';
            toggleBtn.textContent = 'Disable AI';
            toggleBtn.className = 'btn-danger';
            bufferStatus.style.display = 'block';
        } else {
            statusDot.className = 'dot';
            statusText.textContent = 'AI Disabled';
            toggleBtn.textContent = 'Enable AI';
            toggleBtn.className = 'btn-secondary';
            bufferStatus.style.display = 'none';
            predictionPanel.style.display = 'none';
        }

        // Handle availability
        if (!this.available) {
            toggleBtn.disabled = true;
            toggleBtn.classList.add('btn-disabled');
            statusText.textContent = 'AI Unavailable';
            this.showWarning('AI classifier not available. Train the model first.');
        } else {
            toggleBtn.disabled = false;
            toggleBtn.classList.remove('btn-disabled');
            if (!status.enabled) {
                this.hideWarning();
            }
        }

        // Show warning if model not loaded while enabled
        if (status.enabled && !status.model_loaded && this.available) {
            this.showWarning('AI model not loaded. Using demo mode.');
        }
    }

    updateBufferStatus(status) {
        this.bufferStatus = status;

        const progressBar = document.getElementById('buffer-progress');
        const bufferText = document.getElementById('buffer-text');

        if (progressBar && bufferText) {
            progressBar.style.width = `${status.percentage}%`;
            bufferText.textContent = `${status.current_size} / ${status.required_size} samples`;

            // Change color based on readiness
            if (status.ready) {
                progressBar.classList.add('ready');
            } else {
                progressBar.classList.remove('ready');
            }
        }
    }

    handleClassification(prediction) {
        this.currentPrediction = prediction;

        // Show prediction panel
        const predictionPanel = document.getElementById('ai-prediction');
        predictionPanel.style.display = 'block';

        // Update exercise and confidence
        const exerciseEl = document.getElementById('predicted-exercise');
        const confidenceEl = document.getElementById('prediction-confidence');

        exerciseEl.textContent = prediction.exercise.toUpperCase();
        confidenceEl.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

        // Color code confidence
        if (prediction.confidence > 0.7) {
            confidenceEl.className = 'confidence-badge high';
        } else if (prediction.confidence > 0.4) {
            confidenceEl.className = 'confidence-badge medium';
        } else {
            confidenceEl.className = 'confidence-badge low';
        }

        // Update probability bars
        this.updateProbabilityBars(prediction.probabilities);

        // Show warning for dummy classifier
        const warningEl = document.getElementById('ai-warning');
        if (prediction.is_dummy) {
            warningEl.style.display = 'block';
        } else {
            warningEl.style.display = 'none';
        }

        // Emit custom event for other components
        window.dispatchEvent(new CustomEvent('ai-classification', {
            detail: prediction
        }));
    }

    updateProbabilityBars(probabilities) {
        const container = document.getElementById('prob-bars');
        if (!container) return;

        // Sort probabilities
        const sorted = Object.entries(probabilities)
            .sort((a, b) => b[1] - a[1]);

        // Create bars
        container.innerHTML = sorted.map(([exercise, prob]) => `
            <div class="prob-item">
                <div class="prob-label-small">${exercise}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: ${prob * 100}%"></div>
                </div>
                <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
            </div>
        `).join('');
    }

    showWarning(message) {
        const warningEl = document.getElementById('ai-warning');
        if (warningEl) {
            warningEl.textContent = message;
            warningEl.style.display = 'block';
        }
    }

    hideWarning() {
        const warningEl = document.getElementById('ai-warning');
        if (warningEl) {
            warningEl.style.display = 'none';
        }
    }

    getCurrentPrediction() {
        return this.currentPrediction;
    }

    injectStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .ai-coach-row {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 20px;
            }

            .ai-coach-row > * {
                flex: 1;
                min-width: 280px;
            }

            .ai-panel {
                background: linear-gradient(135deg, #0f172a 0%, #1f2937 100%);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 0;
                color: white;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                flex: 1;
            }

            .ai-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }

            .ai-header h3 {
                margin: 0;
                font-size: 1.3em;
            }

            .ai-status {
                margin-bottom: 15px;
            }

            .status-indicator {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .status-indicator .dot {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #ccc;
                animation: pulse 2s infinite;
            }

            .status-indicator .dot.active {
                background: #4ade80;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }

            .buffer-status {
                background: rgba(255, 255, 255, 0.1);
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 15px;
            }

            .buffer-label {
                font-size: 0.9em;
                margin-bottom: 8px;
                opacity: 0.9;
            }

            .progress-bar {
                height: 8px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                overflow: hidden;
                margin-bottom: 5px;
            }

            .progress-fill {
                height: 100%;
                background: #38bdf8;
                transition: width 0.3s ease;
            }

            .progress-fill.ready {
                background: #22c55e;
            }

            .buffer-text {
                font-size: 0.85em;
                opacity: 0.8;
                text-align: center;
            }

            .ai-prediction {
                background: rgba(255, 255, 255, 0.15);
                padding: 15px;
                border-radius: 8px;
                backdrop-filter: blur(10px);
            }

            .prediction-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }

            .prediction-header h4 {
                margin: 0;
                font-size: 1.8em;
                font-weight: bold;
            }

            .confidence-badge {
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 0.9em;
            }

            .confidence-badge.high {
                background: #4ade80;
                color: #064e3b;
            }

            .confidence-badge.medium {
                background: #fbbf24;
                color: #78350f;
            }

            .confidence-badge.low {
                background: #f87171;
                color: #7f1d1d;
            }

            .probabilities {
                margin-top: 15px;
            }

            .prob-label {
                font-size: 0.9em;
                margin-bottom: 10px;
                opacity: 0.9;
            }

            .prob-item {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 8px;
            }

            .prob-label-small {
                width: 80px;
                font-size: 0.85em;
                text-transform: capitalize;
            }

            .prob-bar-container {
                flex: 1;
                height: 6px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
                overflow: hidden;
            }

            .prob-bar {
                height: 100%;
                background: linear-gradient(90deg, #38bdf8, #6366f1);
                transition: width 0.5s ease;
            }

            .prob-value {
                width: 50px;
                text-align: right;
                font-size: 0.85em;
                font-weight: bold;
            }

            .ai-warning {
                margin-top: 12px;
                padding: 8px 12px;
                background: rgba(251, 191, 36, 0.15);
                border-left: 3px solid #fbbf24;
                border-radius: 4px;
                font-size: 0.85em;
            }

            .btn-secondary {
                background: #1f2937;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                transition: background 0.2s;
            }

            .btn-secondary:hover {
                background: #374151;
            }

            .btn-danger {
                background: #b91c1c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                transition: background 0.2s;
            }

            .btn-danger:hover {
                background: #991b1b;
            }

            .btn-disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
        `;
        document.head.appendChild(style);
    }

    getOrCreateAIRow(mainPanel) {
        let row = document.getElementById('ai-coach-row');
        if (row) return row;

        row = document.createElement('div');
        row.id = 'ai-coach-row';
        row.className = 'ai-coach-row';

        const vizCard = mainPanel.querySelector('.card');
        if (vizCard) {
            vizCard.insertAdjacentElement('afterend', row);
        } else {
            mainPanel.appendChild(row);
        }

        return row;
    }
}

// Export for use in main.js
window.AIInterface = AIInterface;
