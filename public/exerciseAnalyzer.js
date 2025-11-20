// Exercise Recognition and Quality Assessment System

export class ExerciseAnalyzer {
    constructor() {
        this.exercises = ['squat', 'deadlift', 'plank', 'pushup'];
        this.currentExercise = null;
        this.confidence = 0;
        this.qualityMetrics = {
            overall: 0,
            depth: 0,
            stability: 0,
            posture: 0,
            symmetry: 0
        };
        this.feedback = [];
        this.dataBuffer = [];
        this.bufferSize = 30; // Keep last 30 frames for analysis
    }

    // Recognize exercise from sensor data
    recognizeExercise(sensorData) {
        const { lumbarLordosis, sagittalTilt, lateralTilt, rotation, acceleration } = sensorData;
        
        // Feature extraction
        const features = this.extractFeatures(sensorData);
        
        // Simple rule-based classifier (can be replaced with trained ML model)
        const scores = {
            squat: this.scoreSquat(features),
            deadlift: this.scoreDeadlift(features),
            plank: this.scorePlank(features),
            pushup: this.scorePushup(features)
        };
        
        // Find exercise with highest score
        let maxScore = 0;
        let detectedExercise = 'unknown';
        
        for (const [exercise, score] of Object.entries(scores)) {
            if (score > maxScore) {
                maxScore = score;
                detectedExercise = exercise;
            }
        }
        
        this.currentExercise = maxScore > 0.3 ? detectedExercise : null;
        this.confidence = maxScore;
        
        return {
            exercise: this.currentExercise,
            confidence: this.confidence,
            scores: scores
        };
    }

    // Extract features from sensor data
    extractFeatures(sensorData) {
        return {
            lordosis: sensorData.lumbarLordosis,
            sagittal: sensorData.sagittalTilt,
            lateral: sensorData.lateralTilt,
            rotation: sensorData.rotation,
            accel: sensorData.acceleration,
            accelMagnitude: Math.sqrt(
                sensorData.acceleration.x ** 2 +
                sensorData.acceleration.y ** 2 +
                sensorData.acceleration.z ** 2
            )
        };
    }

    // Exercise-specific scoring functions
    scoreSquat(features) {
        let score = 0;
        // Squat characteristics: high lordosis change, vertical movement
        if (features.lordosis > 20 && features.lordosis < 60) score += 0.3;
        if (Math.abs(features.sagittal) < 15) score += 0.2;
        if (features.accelMagnitude > 2 && features.accelMagnitude < 8) score += 0.3;
        if (Math.abs(features.lateral) < 10) score += 0.2;
        return Math.min(score, 1.0);
    }

    scoreDeadlift(features) {
        let score = 0;
        // Deadlift: forward lean, high lordosis, vertical pull
        if (features.sagittal > 20 && features.sagittal < 45) score += 0.3;
        if (features.lordosis > 30 && features.lordosis < 70) score += 0.3;
        if (features.accelMagnitude > 1.5 && features.accelMagnitude < 6) score += 0.2;
        if (Math.abs(features.lateral) < 8) score += 0.2;
        return Math.min(score, 1.0);
    }

    scorePlank(features) {
        let score = 0;
        // Plank: low movement, neutral spine, stable
        if (features.accelMagnitude < 1.5) score += 0.4;
        if (Math.abs(features.sagittal) < 10) score += 0.3;
        if (Math.abs(features.lateral) < 5) score += 0.2;
        if (features.lordosis > 10 && features.lordosis < 40) score += 0.1;
        return Math.min(score, 1.0);
    }

    scorePushup(features) {
        let score = 0;
        // Push-up: horizontal movement, moderate lordosis
        if (features.accelMagnitude > 2 && features.accelMagnitude < 7) score += 0.3;
        if (Math.abs(features.sagittal) < 20) score += 0.3;
        if (features.lordosis > 15 && features.lordosis < 50) score += 0.2;
        if (Math.abs(features.lateral) < 10) score += 0.2;
        return Math.min(score, 1.0);
    }

    // Assess quality of exercise performance
    assessQuality(sensorData, exercise) {
        if (!exercise) return this.qualityMetrics;

        // Add to buffer
        this.dataBuffer.push(sensorData);
        if (this.dataBuffer.length > this.bufferSize) {
            this.dataBuffer.shift();
        }

        // Calculate metrics based on exercise type
        switch (exercise) {
            case 'squat':
                return this.assessSquatQuality();
            case 'deadlift':
                return this.assessDeadliftQuality();
            case 'plank':
                return this.assessPlankQuality();
            case 'pushup':
                return this.assessPushupQuality();
            default:
                return this.qualityMetrics;
        }
    }

    assessSquatQuality() {
        if (this.dataBuffer.length < 5) return this.qualityMetrics;

        const lordosisValues = this.dataBuffer.map(d => d.lumbarLordosis);
        const sagittalValues = this.dataBuffer.map(d => d.sagittalTilt);
        const lateralValues = this.dataBuffer.map(d => Math.abs(d.lateralTilt));
        const rotationValues = this.dataBuffer.map(d => Math.abs(d.rotation));

        // Depth: range of lordosis change
        const lordosisRange = Math.max(...lordosisValues) - Math.min(...lordosisValues);
        this.qualityMetrics.depth = Math.min((lordosisRange / 40) * 100, 100);

        // Stability: variance in lateral tilt and rotation
        const lateralVariance = this.calculateVariance(lateralValues);
        const rotationVariance = this.calculateVariance(rotationValues);
        this.qualityMetrics.stability = Math.max(0, 100 - (lateralVariance + rotationVariance) * 10);

        // Posture: sagittal alignment
        const avgSagittal = this.calculateMean(sagittalValues);
        this.qualityMetrics.posture = Math.max(0, 100 - Math.abs(avgSagittal) * 2);

        // Symmetry: lateral balance
        this.qualityMetrics.symmetry = Math.max(0, 100 - this.calculateMean(lateralValues) * 5);

        // Overall score
        this.qualityMetrics.overall = (
            this.qualityMetrics.depth * 0.3 +
            this.qualityMetrics.stability * 0.3 +
            this.qualityMetrics.posture * 0.2 +
            this.qualityMetrics.symmetry * 0.2
        );

        return this.qualityMetrics;
    }

    assessDeadliftQuality() {
        if (this.dataBuffer.length < 5) return this.qualityMetrics;

        const sagittalValues = this.dataBuffer.map(d => d.sagittalTilt);
        const lordosisValues = this.dataBuffer.map(d => d.lumbarLordosis);
        const lateralValues = this.dataBuffer.map(d => Math.abs(d.lateralTilt));

        // Depth: forward lean range
        const sagittalRange = Math.max(...sagittalValues) - Math.min(...sagittalValues);
        this.qualityMetrics.depth = Math.min((sagittalRange / 30) * 100, 100);

        // Stability: lateral variance
        const lateralVariance = this.calculateVariance(lateralValues);
        this.qualityMetrics.stability = Math.max(0, 100 - lateralVariance * 15);

        // Posture: lordosis maintenance (should stay relatively constant)
        const lordosisVariance = this.calculateVariance(lordosisValues);
        this.qualityMetrics.posture = Math.max(0, 100 - lordosisVariance * 2);

        // Symmetry
        this.qualityMetrics.symmetry = Math.max(0, 100 - this.calculateMean(lateralValues) * 5);

        this.qualityMetrics.overall = (
            this.qualityMetrics.depth * 0.25 +
            this.qualityMetrics.stability * 0.35 +
            this.qualityMetrics.posture * 0.25 +
            this.qualityMetrics.symmetry * 0.15
        );

        return this.qualityMetrics;
    }

    assessPlankQuality() {
        if (this.dataBuffer.length < 5) return this.qualityMetrics;

        const sagittalValues = this.dataBuffer.map(d => d.sagittalTilt);
        const lateralValues = this.dataBuffer.map(d => Math.abs(d.lateralTilt));
        const accelValues = this.dataBuffer.map(d => 
            Math.sqrt(d.acceleration.x**2 + d.acceleration.y**2 + d.acceleration.z**2)
        );

        // Depth: N/A for plank, set to 100 if stable
        this.qualityMetrics.depth = 100;

        // Stability: low movement
        const accelVariance = this.calculateVariance(accelValues);
        this.qualityMetrics.stability = Math.max(0, 100 - accelVariance * 50);

        // Posture: neutral alignment
        const avgSagittal = Math.abs(this.calculateMean(sagittalValues));
        this.qualityMetrics.posture = Math.max(0, 100 - avgSagittal * 5);

        // Symmetry
        this.qualityMetrics.symmetry = Math.max(0, 100 - this.calculateMean(lateralValues) * 10);

        this.qualityMetrics.overall = (
            this.qualityMetrics.stability * 0.4 +
            this.qualityMetrics.posture * 0.3 +
            this.qualityMetrics.symmetry * 0.3
        );

        return this.qualityMetrics;
    }

    assessPushupQuality() {
        if (this.dataBuffer.length < 5) return this.qualityMetrics;

        const lordosisValues = this.dataBuffer.map(d => d.lumbarLordosis);
        const lateralValues = this.dataBuffer.map(d => Math.abs(d.lateralTilt));
        const accelValues = this.dataBuffer.map(d => 
            Math.sqrt(d.acceleration.x**2 + d.acceleration.y**2 + d.acceleration.z**2)
        );

        // Depth: range of movement
        const accelRange = Math.max(...accelValues) - Math.min(...accelValues);
        this.qualityMetrics.depth = Math.min((accelRange / 5) * 100, 100);

        // Stability
        const lateralVariance = this.calculateVariance(lateralValues);
        this.qualityMetrics.stability = Math.max(0, 100 - lateralVariance * 15);

        // Posture: lordosis control
        const lordosisVariance = this.calculateVariance(lordosisValues);
        this.qualityMetrics.posture = Math.max(0, 100 - lordosisVariance * 2);

        // Symmetry
        this.qualityMetrics.symmetry = Math.max(0, 100 - this.calculateMean(lateralValues) * 5);

        this.qualityMetrics.overall = (
            this.qualityMetrics.depth * 0.3 +
            this.qualityMetrics.stability * 0.3 +
            this.qualityMetrics.posture * 0.2 +
            this.qualityMetrics.symmetry * 0.2
        );

        return this.qualityMetrics;
    }

    // Generate feedback based on quality metrics
    generateFeedback(exercise, metrics) {
        this.feedback = [];

        if (!exercise) {
            this.feedback.push({
                type: 'info',
                message: 'Waiting for exercise detection...'
            });
            return this.feedback;
        }

        // Overall feedback
        if (metrics.overall < 60) {
            this.feedback.push({
                type: 'error',
                message: `Overall performance needs improvement (${metrics.overall.toFixed(0)}%)`
            });
        } else if (metrics.overall < 80) {
            this.feedback.push({
                type: 'warning',
                message: `Good form, but room for improvement (${metrics.overall.toFixed(0)}%)`
            });
        } else {
            this.feedback.push({
                type: 'success',
                message: `Excellent form! (${metrics.overall.toFixed(0)}%)`
            });
        }

        // Exercise-specific feedback
        if (exercise === 'squat') {
            if (metrics.depth < 70) {
                this.feedback.push({
                    type: 'warning',
                    message: 'Warning: Squat depth insufficient. Go deeper!'
                });
            }
            if (metrics.posture < 70) {
                this.feedback.push({
                    type: 'error',
                    message: 'Warning: Back too rounded. Keep your chest up!'
                });
            }
            if (metrics.stability < 70) {
                this.feedback.push({
                    type: 'warning',
                    message: 'Warning: Unstable movement. Focus on control.'
                });
            }
        } else if (exercise === 'deadlift') {
            if (metrics.posture < 70) {
                this.feedback.push({
                    type: 'error',
                    message: 'Warning: Maintain neutral spine throughout the lift!'
                });
            }
            if (metrics.symmetry < 70) {
                this.feedback.push({
                    type: 'warning',
                    message: 'Warning: Asymmetric loading detected. Check balance.'
                });
            }
        } else if (exercise === 'plank') {
            if (metrics.posture < 70) {
                this.feedback.push({
                    type: 'warning',
                    message: 'Warning: Keep your body in a straight line!'
                });
            }
            if (metrics.stability < 70) {
                this.feedback.push({
                    type: 'warning',
                    message: 'Warning: Too much movement. Hold steady!'
                });
            }
        } else if (exercise === 'pushup') {
            if (metrics.depth < 70) {
                this.feedback.push({
                    type: 'warning',
                    message: 'Warning: Go lower for full range of motion!'
                });
            }
            if (metrics.posture < 70) {
                this.feedback.push({
                    type: 'error',
                    message: 'Warning: Back sagging. Engage your core!'
                });
            }
        }

        return this.feedback;
    }

    // Utility functions
    calculateMean(values) {
        return values.reduce((a, b) => a + b, 0) / values.length;
    }

    calculateVariance(values) {
        const mean = this.calculateMean(values);
        const squaredDiffs = values.map(v => (v - mean) ** 2);
        return this.calculateMean(squaredDiffs);
    }

    reset() {
        this.dataBuffer = [];
        this.currentExercise = null;
        this.confidence = 0;
        this.qualityMetrics = {
            overall: 0,
            depth: 0,
            stability: 0,
            posture: 0,
            symmetry: 0
        };
        this.feedback = [];
    }
}

