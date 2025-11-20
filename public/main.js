// Main Application Controller

import { ExerciseAnalyzer } from './exerciseAnalyzer.js';
import { FlexTail3DViewer } from './FlexTail3DViewer.js';

const DEFAULT_BACKEND_PORT = 5000;

class FlexTailSocketBridge {
    constructor(appInstance) {
        this.app = appInstance;
        this.socket = null;
        this.connected = false;
        this.backendUrl = this.resolveBackendUrl();
        this.connectionStatusListeners = [];
        this.streamingStatusListeners = [];
        this.errorListeners = [];
    }

    resolveBackendUrl() {
        if (window.FLEXTAIL_BACKEND_URL) {
            return window.FLEXTAIL_BACKEND_URL;
        }

        const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
        const hostname = window.location.hostname || 'localhost';
        const port = window.FLEXTAIL_BACKEND_PORT || DEFAULT_BACKEND_PORT;
        return `${protocol}//${hostname}:${port}`;
    }

    initialize() {
        if (typeof window.io !== 'function') {
            console.warn('[FlexTailBridge] Socket.IO client is not available. Skipping live sensor bridge.');
            return;
        }

        this.socket = window.io(this.backendUrl, {
            transports: ['websocket', 'polling'],
            path: '/socket.io'
        });

        this.registerEventHandlers();
    }

    registerEventHandlers() {
        if (!this.socket) return;

        this.socket.on('connect', () => {
            this.connected = true;
            console.info(`[FlexTailBridge] Connected to backend at ${this.backendUrl}`);
            this.emitConnectionStatus({ status: 'ready' });
        });

        this.socket.on('disconnect', () => {
            this.connected = false;
            console.warn('[FlexTailBridge] Disconnected from backend.');
            this.emitConnectionStatus({ status: 'disconnected' });
        });

        this.socket.on('connect_error', (error) => {
            console.error('[FlexTailBridge] Connection error:', error?.message || error);
            this.emitError(error);
        });

        this.socket.on('measurement_data', (payload) => {
            this.handleMeasurement(payload);
        });

        this.socket.on('connection_status', (payload) => {
            this.emitConnectionStatus(payload);
        });

        this.socket.on('streaming_status', (payload) => {
            this.emitStreamingStatus(payload);
        });

        this.socket.on('error', (payload) => {
            this.emitError(payload);
        });
    }

    handleMeasurement(payload) {
        if (!payload || !this.app) return;

        const radians = (value) =>
            typeof value === 'number' ? (value * Math.PI) / 180 : 0;

        const sensorPacket = {
            lumbarAngle: radians(payload.bend),
            sagittal: radians(payload.pitch),
            lateral: radians(payload.roll),
            twist: radians(payload.yaw || 0),
            acceleration: payload.acceleration ?? payload.accel ?? 0
        };

        // Forward to the existing ingestion pipeline.
        this.app.ingestSensorData(sensorPacket);
    }

    connectSensor({ macAddress, hardwareVersion }) {
        if (!this.socket) return;
        this.socket.emit('connect_sensor', {
            mac_address: macAddress || undefined,
            hw_version: hardwareVersion || '6.0'
        });
    }

    disconnectSensor() {
        if (!this.socket) return;
        this.socket.emit('disconnect_sensor');
    }

    startMeasurement(frequency = 25) {
        if (!this.socket) return;
        this.socket.emit('start_measurement', { frequency });
    }

    stopMeasurement() {
        if (!this.socket) return;
        this.socket.emit('stop_measurement');
    }

    onConnectionStatus(callback) {
        this.connectionStatusListeners.push(callback);
    }

    onStreamingStatus(callback) {
        this.streamingStatusListeners.push(callback);
    }

    onError(callback) {
        this.errorListeners.push(callback);
    }

    emitConnectionStatus(payload) {
        this.connectionStatusListeners.forEach((cb) => cb(payload));
    }

    emitStreamingStatus(payload) {
        this.streamingStatusListeners.forEach((cb) => cb(payload));
    }

    emitError(payload) {
        this.errorListeners.forEach((cb) => cb(payload));
    }
}

class App {
    constructor() {
        this.analyzer = new ExerciseAnalyzer();
        this.flextailViewer = null;
        
        this.isAnalyzing = false;
        this.selectedExercise = 'auto';
        this.videoElement = null;
        this.videoPlaceholder = null;
        this.defaultVideoMessage = 'Select an exercise to preview the movement.';
        this.missingVideoMessage = 'Video coming soon for this movement.';
        this.exerciseVideos = {
            squat: 'videos/squat.mp4',
            deadlift: 'videos/deadlift.mp4',
            row: 'videos/row.mp4',
            pushup: 'videos/pushup.mp4',
            plank: 'videos/plank.mp4',
            burpee: 'videos/burpee.mp4'
        };
        
        // Wait for analysis view to be shown before initializing
        this.initialized = false;
    }

    // Initialize when analysis view is shown
    initialize() {
        if (this.initialized) return;
        
        this.init();
        this.initialized = true;
    }

    // Set exercise from navigation
    setExercise(exerciseId) {
        this.selectedExercise = exerciseId;
        this.updateExerciseVideo(exerciseId);
        // Map exercise IDs to analyzer format
        const exerciseMap = {
            'squat': 'squat',
            'deadlift': 'deadlift',
            'row': 'deadlift', // Row uses similar analysis to deadlift
            'pushup': 'pushup',
            'plank': 'plank',
            'burpee': 'pushup', // Burpee uses pushup analysis
            'warrior': 'squat', // Warrior pose uses squat-like analysis
            'downward': 'plank', // Downward dog uses plank analysis
            'tree': 'plank' // Tree pose uses plank analysis
        };
        
        const mappedExercise = exerciseMap[exerciseId] || 'auto';
        const exerciseSelect = document.getElementById('exercise-select');
        if (exerciseSelect) {
            exerciseSelect.value = mappedExercise;
        }
        
        this.selectedExercise = mappedExercise;
    }

    init() {
        this.videoElement = document.getElementById('exercise-video');
        this.videoPlaceholder = document.getElementById('exercise-video-placeholder');
        if (this.videoPlaceholder) {
            this.defaultVideoMessage = this.videoPlaceholder.textContent.trim() || this.defaultVideoMessage;
        }
        this.updateExerciseVideo(this.selectedExercise);

        // Initialize FlexTail 3D viewer (only if container exists)
        const flextailContainer = document.getElementById('flextail-container');
        if (flextailContainer) {
            this.flextailViewer = new FlexTail3DViewer(flextailContainer, {
                flexTailColor: 0xff6600,
                backgroundColor: 0x1a1a1a,
                groundPlaneColor: 0x404040,
                showGroundPlane: true,
                useAccelerometer: true,
            });
        }

        // Setup UI event listeners
        this.setupEventListeners();

        // Initial UI update (no data yet)
        this.updateUI(null, null, null, null, null);
    }

    setupEventListeners() {
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        const exerciseSelect = document.getElementById('exercise-select');

        startBtn.addEventListener('click', () => this.startAnalysis());
        stopBtn.addEventListener('click', () => this.stopAnalysis());
        exerciseSelect.addEventListener('change', (e) => {
            this.selectedExercise = e.target.value;
            this.updateExerciseVideo(this.selectedExercise);
        });
    }

    updateExerciseVideo(exerciseId) {
        if (!this.videoElement) return;

        const videoSource = this.getVideoSource(exerciseId);

        if (videoSource) {
            if (this.videoElement.getAttribute('src') !== videoSource) {
                this.videoElement.setAttribute('src', videoSource);
                this.videoElement.load();
            }
            this.videoElement.style.display = 'block';
            if (this.videoPlaceholder) {
                this.videoPlaceholder.style.display = 'none';
                this.videoPlaceholder.textContent = this.defaultVideoMessage;
            }
        } else {
            const hadSource = this.videoElement.hasAttribute('src');
            this.videoElement.removeAttribute('src');
            if (hadSource) {
                this.videoElement.load();
            }
            this.videoElement.style.display = 'none';
            if (this.videoPlaceholder) {
                this.videoPlaceholder.style.display = 'block';
                if (exerciseId && exerciseId !== 'auto') {
                    this.videoPlaceholder.textContent = this.missingVideoMessage;
                } else {
                    this.videoPlaceholder.textContent = this.defaultVideoMessage;
                }
            }
        }
    }

    getVideoSource(exerciseId) {
        if (!exerciseId || exerciseId === 'auto') return null;
        return this.exerciseVideos[exerciseId] || null;
    }

    startAnalysis() {
        if (this.isAnalyzing) return;

        this.isAnalyzing = true;
        this.analyzer.reset();

        document.getElementById('start-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;
    }

    stopAnalysis() {
        if (!this.isAnalyzing) return;

        this.isAnalyzing = false;
        this.analyzer.reset();

        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
        // Reset UI
        this.updateUI(null, null, null, null, null);
    }

    /**
     * Converts sensor data to FlexTail measurement format
     */
    convertToFlexTailMeasurement(rawData, adaptedData) {
        if (!rawData && !adaptedData) return null;

        // Use raw data if available, otherwise use adapted
        const data = rawData || adaptedData;
        
        // Generate reconstruction from sensor angles
        const numPoints = 18;
        const length = 360;
        const width = 20;
        
        // Convert angles to radians if needed
        const radiansToDegrees = (rad) => rad * (180 / Math.PI);
        const degreesToRadians = (deg) => deg * (Math.PI / 180);
        
        // Get angles in radians
        let lumbarAngle = data.lumbarAngle !== undefined 
            ? data.lumbarAngle 
            : degreesToRadians(adaptedData?.lumbarLordosis || 0);
        
        let sagittal = data.sagittal !== undefined 
            ? data.sagittal 
            : degreesToRadians(adaptedData?.sagittalTilt || 0);
        
        let lateral = data.lateral !== undefined 
            ? data.lateral 
            : degreesToRadians(adaptedData?.lateralTilt || 0);
        
        let twist = data.twist !== undefined 
            ? data.twist 
            : degreesToRadians(adaptedData?.rotation || 0);

        // Generate reconstruction based on sensor data
        const left = { x: [], y: [], z: [] };
        const right = { x: [], y: [], z: [] };
        const center = { x: [], y: [], z: [] };
        
        for (let i = 0; i < numPoints; i++) {
            const t = i / (numPoints - 1);
            const z = length * t;
            
            // Use lumbar angle to create bend (curvature)
            const bendCurve = Math.sin(t * Math.PI) * radiansToDegrees(lumbarAngle) * 0.5;
            const twistAngle = t * radiansToDegrees(twist);
            const cosT = Math.cos(twistAngle * Math.PI / 180);
            const sinT = Math.sin(twistAngle * Math.PI / 180);
            
            // Apply sagittal and lateral tilts
            const sagittalOffset = radiansToDegrees(sagittal) * t * 0.3;
            const lateralOffset = radiansToDegrees(lateral) * 0.5;
            
            // Left side
            const leftX = -width / 2 + lateralOffset;
            const leftY = bendCurve + sagittalOffset;
            left.x.push(leftX * cosT - leftY * sinT);
            left.y.push(leftX * sinT + leftY * cosT);
            left.z.push(z);
            
            // Right side
            const rightX = width / 2 + lateralOffset;
            const rightY = bendCurve + sagittalOffset;
            right.x.push(rightX * cosT - rightY * sinT);
            right.y.push(rightX * sinT + rightY * cosT);
            right.z.push(z);
            
            // Center
            center.x.push(bendCurve * sinT + sagittalOffset);
            center.y.push(bendCurve * cosT);
            center.z.push(z);
        }
        
        return {
            reconstruction: { left, right, center },
            orientation: {
                pitch: sagittal,
                roll: lateral,
                yaw: twist
            }
        };
    }

    /**
     * Converts raw sensor data format to internal format.
     * Handles both actual sensor format (lumbarAngle, sagittal, lateral, twist, acceleration)
     * and legacy format (lumbarLordosis, sagittalTilt, lateralTilt, rotation, acceleration object).
     */
    adaptSensorData(rawData) {
        // If data is already in expected format, return as-is
        if (rawData.lumbarLordosis !== undefined) {
            return rawData;
        }

        // Convert from actual sensor format
        // Angles are in radians, convert to degrees
        const radiansToDegrees = (rad) => rad * (180 / Math.PI);
        
        return {
            // Map lumbarAngle (radians) to lumbarLordosis (degrees)
            lumbarLordosis: rawData.lumbarAngle !== undefined 
                ? radiansToDegrees(rawData.lumbarAngle) 
                : rawData.lumbarLordosis || 0,
            
            // Map sagittal (radians) to sagittalTilt (degrees)
            sagittalTilt: rawData.sagittal !== undefined 
                ? radiansToDegrees(rawData.sagittal) 
                : rawData.sagittalTilt || 0,
            
            // Map lateral (radians) to lateralTilt (degrees)
            lateralTilt: rawData.lateral !== undefined 
                ? radiansToDegrees(rawData.lateral) 
                : rawData.lateralTilt || 0,
            
            // Map twist (radians) to rotation (degrees)
            rotation: rawData.twist !== undefined 
                ? radiansToDegrees(rawData.twist) 
                : rawData.rotation || 0,
            
            // Handle acceleration (can be single value or object)
            acceleration: typeof rawData.acceleration === 'number'
                ? { x: 0, y: rawData.acceleration, z: 0 } // Use as y-component
                : rawData.acceleration || { x: 0, y: 0, z: 0 },
            
        };
    }

    /**
     * Public method to feed sensor data into the system.
     * Call this with real sensor packets once available.
     * 
     * Expected data format (from FlexTail sensors):
     * {
     *   lumbarAngle: number (radians),
     *   sagittal: number (radians),
     *   lateral: number (radians),
     *   twist: number (radians),
     *   acceleration: number (magnitude),
     *   gyro: number (optional),
     *   thoracicAngle: number (optional, radians)
     * }
     */
    ingestSensorData(rawSensorData) {
        if (!this.isAnalyzing || !rawSensorData) return;

        // Adapt data to internal format
        const sensorData = this.adaptSensorData(rawSensorData);

        // Update FlexTail spine visualization
        if (this.flextailViewer) {
            const flextailMeasurement = this.convertToFlexTailMeasurement(rawSensorData, sensorData);
            this.flextailViewer.updateMeasurement(flextailMeasurement);
        }

        // Recognize exercise
        const recognition = this.analyzer.recognizeExercise(sensorData);
        
        // Use selected exercise if not auto-detect
        const exercise = this.selectedExercise !== 'auto' ? this.selectedExercise : recognition.exercise;

        // Assess quality
        const metrics = this.analyzer.assessQuality(sensorData, exercise);

        // Generate feedback
        const feedback = this.analyzer.generateFeedback(exercise, metrics);

        // Update UI (pass both raw and adapted data)
        this.updateUI(rawSensorData, sensorData, recognition, metrics, feedback);
    }

    updateUI(rawSensorData, adaptedSensorData, recognition, metrics, feedback) {
        // Update exercise display
        const exerciseDisplay = document.getElementById('exercise-display');
        const confidenceDisplay = document.getElementById('confidence-display');
        
        if (recognition && recognition.exercise) {
            exerciseDisplay.textContent = recognition.exercise.charAt(0).toUpperCase() + 
                                         recognition.exercise.slice(1);
            confidenceDisplay.textContent = recognition.confidence
                ? `Confidence: ${(recognition.confidence * 100).toFixed(0)}%`
                : '';
        } else {
            exerciseDisplay.textContent = 'Detecting...';
            confidenceDisplay.textContent = '';
        }

        // Update quality metrics with progress bars
        const safeMetrics = metrics || {};
        const overallScore = safeMetrics.overall > 0 ? safeMetrics.overall : 0;
        const depthScore = safeMetrics.depth > 0 ? safeMetrics.depth : 0;
        const stabilityScore = safeMetrics.stability > 0 ? safeMetrics.stability : 0;
        const postureScore = safeMetrics.posture > 0 ? safeMetrics.posture : 0;
        const symmetryScore = safeMetrics.symmetry > 0 ? safeMetrics.symmetry : 0;
        
        // Update overall score
        document.getElementById('overall-score').textContent = 
            overallScore > 0 ? `${overallScore.toFixed(0)}%` : '--';
        document.getElementById('overall-progress').style.width = `${overallScore}%`;
        
        // Update individual metrics with progress bars
        document.getElementById('depth-score').textContent = 
            depthScore > 0 ? `${depthScore.toFixed(0)}%` : '--';
        document.getElementById('depth-progress').style.width = `${depthScore}%`;
        
        document.getElementById('stability-score').textContent = 
            stabilityScore > 0 ? `${stabilityScore.toFixed(0)}%` : '--';
        document.getElementById('stability-progress').style.width = `${stabilityScore}%`;
        
        document.getElementById('posture-score').textContent = 
            postureScore > 0 ? `${postureScore.toFixed(0)}%` : '--';
        document.getElementById('posture-progress').style.width = `${postureScore}%`;
        
        document.getElementById('symmetry-score').textContent = 
            symmetryScore > 0 ? `${symmetryScore.toFixed(0)}%` : '--';
        document.getElementById('symmetry-progress').style.width = `${symmetryScore}%`;
        
        // Update risk indicator
        this.updateRiskIndicator(overallScore);

        // Update feedback
        const feedbackContainer = document.getElementById('feedback-container');
        feedbackContainer.innerHTML = '';
        
        if (feedback && feedback.length > 0) {
            feedback.forEach(item => {
                const feedbackEl = document.createElement('div');
                feedbackEl.className = `feedback-item feedback-${item.type}`;
                feedbackEl.textContent = item.message;
                feedbackContainer.appendChild(feedbackEl);
            });
        } else {
            feedbackContainer.innerHTML = '<p class="feedback-placeholder">No feedback yet</p>';
        }

        // Update sensor data display (use adapted data for display)
        if (adaptedSensorData) {
            document.getElementById('lordosis').textContent = 
                (adaptedSensorData.lumbarLordosis || 0).toFixed(1) + '°';
            document.getElementById('sagittal').textContent = 
                (adaptedSensorData.sagittalTilt || 0).toFixed(1) + '°';
            document.getElementById('lateral').textContent = 
                (adaptedSensorData.lateralTilt || 0).toFixed(1) + '°';
            document.getElementById('rotation').textContent = 
                (adaptedSensorData.rotation || 0).toFixed(1) + '°';
            
            // Calculate acceleration magnitude
            const accel = adaptedSensorData.acceleration;
            const accelMag = typeof accel === 'number' 
                ? accel 
                : (accel ? Math.sqrt(accel.x**2 + accel.y**2 + accel.z**2) : 0);
            document.getElementById('acceleration').textContent = accelMag.toFixed(2) + ' m/s²';
        } else {
            // Reset to default values
            document.getElementById('lordosis').textContent = '--';
            document.getElementById('sagittal').textContent = '--';
            document.getElementById('lateral').textContent = '--';
            document.getElementById('rotation').textContent = '--';
            document.getElementById('acceleration').textContent = '--';
        }
    }

    updateRiskIndicator(score) {
        const riskIndicator = document.getElementById('risk-indicator');
        const riskLabel = document.getElementById('risk-label');
        const riskBadge = document.getElementById('risk-badge');
        
        if (!riskIndicator || score === 0) {
            if (riskIndicator) riskIndicator.style.display = 'none';
            return;
        }
        
        riskIndicator.style.display = 'block';
        
        let riskLevel, riskClass, iconClass, badgeClass, badgeText, icon;
        
        if (score >= 80) {
            riskLevel = 'Safe';
            riskClass = 'risk-safe';
            iconClass = 'risk-icon-safe';
            badgeClass = 'badge-success';
            badgeText = 'SAFE';
            icon = '✓';
        } else if (score >= 60) {
            riskLevel = 'Caution';
            riskClass = 'risk-caution';
            iconClass = 'risk-icon-caution';
            badgeClass = 'badge-warning';
            badgeText = 'CAUTION';
            icon = '⚠';
        } else {
            riskLevel = 'Warning';
            riskClass = 'risk-danger';
            iconClass = 'risk-icon-danger';
            badgeClass = 'badge-danger';
            badgeText = 'WARNING';
            icon = '⚠';
        }
        
        // Update classes
        riskIndicator.className = `risk-indicator ${riskClass}`;
        const riskIcon = riskIndicator.querySelector('.risk-icon');
        if (riskIcon) {
            riskIcon.className = `risk-icon ${iconClass}`;
            riskIcon.innerHTML = `<span style="color: white;">${icon}</span>`;
        }
        
        if (riskLabel) riskLabel.textContent = riskLevel;
        if (riskBadge) {
            riskBadge.className = `badge ${badgeClass}`;
            riskBadge.textContent = badgeText;
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
    window.flexTailBridge = new FlexTailSocketBridge(window.app);
    window.flexTailBridge.initialize();
    setupSensorControls(window.flexTailBridge);
    // Don't initialize immediately - wait for analysis view to be shown
});

function setupSensorControls(bridge) {
    const macInput = document.getElementById('sensor-mac');
    const hwSelect = document.getElementById('sensor-hw');
    const freqInput = document.getElementById('sensor-frequency');
    const connectBtn = document.getElementById('sensor-connect-btn');
    const disconnectBtn = document.getElementById('sensor-disconnect-btn');
    const startBtn = document.getElementById('sensor-start-btn');
    const stopBtn = document.getElementById('sensor-stop-btn');
    const statusDot = document.querySelector('#sensor-status .status-dot');
    const statusText = document.getElementById('sensor-status-text');

    if (!macInput || !connectBtn || !statusDot) return;

    const setStatus = (label, state) => {
        statusText.textContent = label;
        statusDot.classList.remove('status-connected', 'status-connecting', 'status-disconnected');
        statusDot.classList.add(`status-${state}`);
    };

    connectBtn.addEventListener('click', () => {
        const macAddress = macInput.value.trim();
        const hardwareVersion = hwSelect.value;
        setStatus('Connecting…', 'connecting');
        bridge.connectSensor({ macAddress, hardwareVersion });
    });

    disconnectBtn.addEventListener('click', () => {
        bridge.disconnectSensor();
        setStatus('Disconnecting…', 'connecting');
    });

    startBtn.addEventListener('click', () => {
        const frequency = parseInt(freqInput.value, 10) || 25;
        bridge.startMeasurement(frequency);
    });

    stopBtn.addEventListener('click', () => {
        bridge.stopMeasurement();
    });

    bridge.onConnectionStatus((payload = {}) => {
        const status = payload.status || 'unknown';
        switch (status) {
            case 'connected':
                setStatus('Connected', 'connected');
                break;
            case 'connecting':
            case 'retrying':
            case 'reconnecting':
                setStatus('Connecting…', 'connecting');
                break;
            case 'failed':
                setStatus('Failed to connect', 'disconnected');
                break;
            case 'disconnected':
                setStatus('Disconnected', 'disconnected');
                break;
            default:
                setStatus(status, 'connecting');
        }
    });

    bridge.onStreamingStatus((payload = {}) => {
        if (payload.streaming) {
            setStatus('Streaming', 'connected');
        }
    });

    bridge.onError((payload = {}) => {
        if (payload.message) {
            console.error('[FlexTailBridge] Error:', payload.message);
        }
    });
}

