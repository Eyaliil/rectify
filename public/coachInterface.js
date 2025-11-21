/**
 * AI Coach Interface
 * Real-time form analysis and coaching feedback
 */

class CoachInterface {
    constructor(socket) {
        this.socket = socket;
        this.enabled = false;
        this.currentAnalysis = null;

        this.setupSocketListeners();
        this.createUI();
    }

    setupSocketListeners() {
        // Form analysis results
        this.socket.on('form_analysis', (analysis) => {
            this.currentAnalysis = analysis;
            this.updateUI(analysis);
        });

        // Form buffer status
        this.socket.on('form_buffer_status', (status) => {
            this.updateBufferStatus(status);
        });

        // Form status
        this.socket.on('form_status', (status) => {
            this.enabled = status.enabled;
            this.updateToggleButton();
        });

        // Session summary
        this.socket.on('session_summary', (summary) => {
            this.showSessionSummary(summary);
        });
    }

    createUI() {
        // Main coach panel
        const panel = document.createElement('div');
        panel.id = 'coach-panel';
        panel.innerHTML = `
            <div class="coach-header">
                <h3>AI Coach</h3>
                <div class="coach-controls">
                    <button id="coach-toggle" class="coach-btn">Enable Coach</button>
                    <button id="session-reset" class="coach-btn secondary">Reset Session</button>
                </div>
            </div>

            <div class="coach-content">
                <!-- Form Quality Indicator -->
                <div class="form-indicator" id="form-indicator">
                    <div class="form-quality-circle" id="form-circle">
                        <span id="form-emoji">🎯</span>
                    </div>
                    <div class="form-label" id="form-label">Analyzing...</div>
                </div>

                <!-- Coaching Message -->
                <div class="coaching-message" id="coaching-message">
                    <div class="message-icon">💬</div>
                    <div class="message-text" id="coach-text">Enable AI Coach to get real-time feedback</div>
                </div>

                <!-- Tips Panel -->
                <div class="tips-panel" id="tips-panel">
                    <h4>Tips</h4>
                    <ul id="tips-list"></ul>
                </div>

                <!-- Metrics Panel -->
                <div class="metrics-panel" id="metrics-panel">
                    <h4>Form Metrics</h4>
                    <div class="metrics-grid" id="metrics-grid"></div>
                </div>

                <!-- Session Stats -->
                <div class="session-stats" id="session-stats">
                    <div class="stat-item">
                        <span class="stat-value" id="total-reps">0</span>
                        <span class="stat-label">Reps</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value" id="good-reps">0</span>
                        <span class="stat-label">Good</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value" id="form-score">-</span>
                        <span class="stat-label">Score</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value" id="form-grade">-</span>
                        <span class="stat-label">Grade</span>
                    </div>
                </div>

                <!-- Buffer Progress -->
                <div class="buffer-progress" id="coach-buffer">
                    <div class="buffer-bar" id="coach-buffer-bar"></div>
                    <span class="buffer-text" id="coach-buffer-text">0%</span>
                </div>
            </div>
        `;

        // Add styles
        this.addStyles();

        // Append to body
        document.body.appendChild(panel);

        // Setup event listeners
        document.getElementById('coach-toggle').addEventListener('click', () => this.toggleCoach());
        document.getElementById('session-reset').addEventListener('click', () => this.resetSession());
    }

    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            #coach-panel {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 320px;
                background: linear-gradient(135deg, #1a5f2a 0%, #0d3d18 100%);
                border-radius: 16px;
                padding: 20px;
                color: white;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                z-index: 1000;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }

            .coach-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            }

            .coach-header h3 {
                margin: 0;
                font-size: 18px;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .coach-header h3::before {
                content: '🏋️';
            }

            .coach-controls {
                display: flex;
                gap: 8px;
            }

            .coach-btn {
                padding: 6px 12px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
                font-weight: 600;
                transition: all 0.2s;
            }

            .coach-btn:not(.secondary) {
                background: #4ade80;
                color: #0a2e14;
            }

            .coach-btn:not(.secondary):hover {
                background: #22c55e;
            }

            .coach-btn.secondary {
                background: rgba(255,255,255,0.2);
                color: white;
            }

            .coach-btn.active {
                background: #ef4444;
                color: white;
            }

            .form-indicator {
                text-align: center;
                padding: 20px;
            }

            .form-quality-circle {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 10px;
                font-size: 36px;
                background: rgba(255,255,255,0.1);
                transition: all 0.3s;
            }

            .form-quality-circle.good {
                background: linear-gradient(135deg, #22c55e, #16a34a);
                box-shadow: 0 0 20px rgba(34, 197, 94, 0.5);
            }

            .form-quality-circle.bad {
                background: linear-gradient(135deg, #f59e0b, #d97706);
                box-shadow: 0 0 20px rgba(245, 158, 11, 0.5);
            }

            .form-label {
                font-size: 16px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .coaching-message {
                background: rgba(0,0,0,0.2);
                border-radius: 12px;
                padding: 15px;
                margin: 15px 0;
                display: flex;
                align-items: flex-start;
                gap: 12px;
            }

            .message-icon {
                font-size: 24px;
            }

            .message-text {
                flex: 1;
                font-size: 14px;
                line-height: 1.4;
            }

            .tips-panel {
                background: rgba(0,0,0,0.2);
                border-radius: 12px;
                padding: 12px;
                margin: 10px 0;
            }

            .tips-panel h4 {
                margin: 0 0 8px 0;
                font-size: 12px;
                text-transform: uppercase;
                opacity: 0.8;
            }

            #tips-list {
                margin: 0;
                padding-left: 20px;
                font-size: 13px;
            }

            #tips-list li {
                margin: 5px 0;
            }

            .metrics-panel {
                background: rgba(0,0,0,0.2);
                border-radius: 12px;
                padding: 12px;
                margin: 10px 0;
            }

            .metrics-panel h4 {
                margin: 0 0 8px 0;
                font-size: 12px;
                text-transform: uppercase;
                opacity: 0.8;
            }

            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 8px;
            }

            .metric-item {
                text-align: center;
                padding: 8px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
            }

            .metric-item.good {
                background: rgba(34, 197, 94, 0.3);
            }

            .metric-item.warning {
                background: rgba(245, 158, 11, 0.3);
            }

            .metric-label {
                font-size: 10px;
                text-transform: uppercase;
                opacity: 0.8;
            }

            .metric-value {
                font-size: 14px;
                font-weight: 600;
            }

            .session-stats {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 10px;
                margin: 15px 0;
            }

            .stat-item {
                text-align: center;
                padding: 10px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
            }

            .stat-value {
                display: block;
                font-size: 20px;
                font-weight: 700;
            }

            .stat-label {
                font-size: 10px;
                text-transform: uppercase;
                opacity: 0.7;
            }

            .buffer-progress {
                height: 6px;
                background: rgba(255,255,255,0.2);
                border-radius: 3px;
                overflow: hidden;
                position: relative;
            }

            #coach-buffer-bar {
                height: 100%;
                background: linear-gradient(90deg, #4ade80, #22c55e);
                width: 0%;
                transition: width 0.3s;
            }

            #coach-buffer-text {
                position: absolute;
                right: 5px;
                top: -18px;
                font-size: 10px;
                opacity: 0.7;
            }
        `;
        document.head.appendChild(style);
    }

    toggleCoach() {
        if (this.enabled) {
            this.socket.emit('disable_form_analysis');
        } else {
            this.socket.emit('enable_form_analysis');
        }
    }

    updateToggleButton() {
        const btn = document.getElementById('coach-toggle');
        if (this.enabled) {
            btn.textContent = 'Disable Coach';
            btn.classList.add('active');
        } else {
            btn.textContent = 'Enable Coach';
            btn.classList.remove('active');
        }
    }

    resetSession() {
        this.socket.emit('reset_session');
        // Reset UI
        document.getElementById('total-reps').textContent = '0';
        document.getElementById('good-reps').textContent = '0';
        document.getElementById('form-score').textContent = '-';
        document.getElementById('form-grade').textContent = '-';
    }

    updateBufferStatus(status) {
        const bar = document.getElementById('coach-buffer-bar');
        const text = document.getElementById('coach-buffer-text');

        bar.style.width = `${status.percentage}%`;
        text.textContent = `${Math.round(status.percentage)}%`;
    }

    updateUI(analysis) {
        // Update form quality indicator
        const circle = document.getElementById('form-circle');
        const emoji = document.getElementById('form-emoji');
        const label = document.getElementById('form-label');

        circle.className = 'form-quality-circle ' + analysis.form_quality;
        emoji.textContent = analysis.form_quality === 'good' ? '✅' : '⚠️';
        label.textContent = analysis.form_quality.toUpperCase() + ' FORM';

        // Update coaching message
        const coaching = analysis.coaching;
        document.getElementById('coach-text').textContent = coaching.message;

        // Update tips
        const tipsList = document.getElementById('tips-list');
        tipsList.innerHTML = coaching.tips.map(tip => `<li>${tip}</li>`).join('');

        // Update metrics
        this.updateMetrics(coaching.metrics);

        // Update session stats
        const stats = analysis.session_stats;
        if (stats) {
            document.getElementById('total-reps').textContent = stats.total_reps;
            document.getElementById('good-reps').textContent = stats.good_reps;
            document.getElementById('form-score').textContent =
                stats.form_score > 0 ? `${stats.form_score.toFixed(0)}%` : '-';
            document.getElementById('form-grade').textContent = stats.grade || '-';
        }
    }

    updateMetrics(metrics) {
        const grid = document.getElementById('metrics-grid');
        grid.innerHTML = '';

        if (!metrics || Object.keys(metrics).length === 0) {
            grid.innerHTML = '<div class="metric-item"><span class="metric-value">-</span></div>';
            return;
        }

        for (const [key, metric] of Object.entries(metrics)) {
            const item = document.createElement('div');
            item.className = `metric-item ${metric.status}`;
            item.innerHTML = `
                <div class="metric-label">${key}</div>
                <div class="metric-value">${metric.value}</div>
            `;
            item.title = metric.tip;
            grid.appendChild(item);
        }
    }

    showSessionSummary(summary) {
        // Could show a modal or update a summary panel
        console.log('Session Summary:', summary);
    }
}

// ES Module export for browser
export { CoachInterface };
