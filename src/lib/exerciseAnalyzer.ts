// Exercise Recognition and Quality Assessment System

export interface SensorData {
  lumbarLordosis?: number;
  sagittalTilt?: number;
  lateralTilt?: number;
  rotation?: number;
  acceleration?: { x: number; y: number; z: number } | number;
  // Raw sensor format
  lumbarAngle?: number;
  sagittal?: number;
  lateral?: number;
  twist?: number;
}

export interface QualityMetrics {
  overall: number;
  depth: number;
  stability: number;
  posture: number;
  symmetry: number;
}

export interface FeedbackItem {
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
}

export interface ExerciseRecognition {
  exercise: string | null;
  confidence: number;
  scores: Record<string, number>;
}

export class ExerciseAnalyzer {
  private exercises = ['squat', 'deadlift', 'plank', 'pushup'];
  private currentExercise: string | null = null;
  private confidence = 0;
  private qualityMetrics: QualityMetrics = {
    overall: 0,
    depth: 0,
    stability: 0,
    posture: 0,
    symmetry: 0
  };
  private feedback: FeedbackItem[] = [];
  private dataBuffer: SensorData[] = [];
  private bufferSize = 30;

  // Recognize exercise from sensor data
  recognizeExercise(sensorData: SensorData): ExerciseRecognition {
    const features = this.extractFeatures(sensorData);
    
    const scores = {
      squat: this.scoreSquat(features),
      deadlift: this.scoreDeadlift(features),
      plank: this.scorePlank(features),
      pushup: this.scorePushup(features)
    };
    
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

  private extractFeatures(sensorData: SensorData) {
    const adapted = this.adaptSensorData(sensorData);
    const accel = adapted.acceleration;
    const accelMagnitude = typeof accel === 'number' 
      ? accel 
      : (accel ? Math.sqrt(accel.x ** 2 + accel.y ** 2 + accel.z ** 2) : 0);

    return {
      lordosis: adapted.lumbarLordosis || 0,
      sagittal: adapted.sagittalTilt || 0,
      lateral: adapted.lateralTilt || 0,
      rotation: adapted.rotation || 0,
      accel: accel,
      accelMagnitude
    };
  }

  private adaptSensorData(rawData: SensorData): SensorData {
    if (rawData.lumbarLordosis !== undefined) {
      return rawData;
    }

    const radiansToDegrees = (rad: number) => rad * (180 / Math.PI);
    
    return {
      lumbarLordosis: rawData.lumbarAngle !== undefined 
        ? radiansToDegrees(rawData.lumbarAngle) 
        : rawData.lumbarLordosis || 0,
      sagittalTilt: rawData.sagittal !== undefined 
        ? radiansToDegrees(rawData.sagittal) 
        : rawData.sagittalTilt || 0,
      lateralTilt: rawData.lateral !== undefined 
        ? radiansToDegrees(rawData.lateral) 
        : rawData.lateralTilt || 0,
      rotation: rawData.twist !== undefined 
        ? radiansToDegrees(rawData.twist) 
        : rawData.rotation || 0,
      acceleration: typeof rawData.acceleration === 'number'
        ? { x: 0, y: rawData.acceleration, z: 0 }
        : rawData.acceleration || { x: 0, y: 0, z: 0 },
    };
  }

  private scoreSquat(features: ReturnType<typeof this.extractFeatures>) {
    let score = 0;
    if (features.lordosis > 20 && features.lordosis < 60) score += 0.3;
    if (Math.abs(features.sagittal) < 15) score += 0.2;
    if (features.accelMagnitude > 2 && features.accelMagnitude < 8) score += 0.3;
    if (Math.abs(features.lateral) < 10) score += 0.2;
    return Math.min(score, 1.0);
  }

  private scoreDeadlift(features: ReturnType<typeof this.extractFeatures>) {
    let score = 0;
    if (features.sagittal > 20 && features.sagittal < 45) score += 0.3;
    if (features.lordosis > 30 && features.lordosis < 70) score += 0.3;
    if (features.accelMagnitude > 1.5 && features.accelMagnitude < 6) score += 0.2;
    if (Math.abs(features.lateral) < 8) score += 0.2;
    return Math.min(score, 1.0);
  }

  private scorePlank(features: ReturnType<typeof this.extractFeatures>) {
    let score = 0;
    if (features.accelMagnitude < 1.5) score += 0.4;
    if (Math.abs(features.sagittal) < 10) score += 0.3;
    if (Math.abs(features.lateral) < 5) score += 0.2;
    if (features.lordosis > 10 && features.lordosis < 40) score += 0.1;
    return Math.min(score, 1.0);
  }

  private scorePushup(features: ReturnType<typeof this.extractFeatures>) {
    let score = 0;
    if (features.accelMagnitude > 2 && features.accelMagnitude < 7) score += 0.3;
    if (Math.abs(features.sagittal) < 20) score += 0.3;
    if (features.lordosis > 15 && features.lordosis < 50) score += 0.2;
    if (Math.abs(features.lateral) < 10) score += 0.2;
    return Math.min(score, 1.0);
  }

  assessQuality(sensorData: SensorData, exercise: string | null): QualityMetrics {
    if (!exercise) return this.qualityMetrics;

    this.dataBuffer.push(sensorData);
    if (this.dataBuffer.length > this.bufferSize) {
      this.dataBuffer.shift();
    }

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

  private assessSquatQuality(): QualityMetrics {
    if (this.dataBuffer.length < 5) return this.qualityMetrics;

    const adapted = this.dataBuffer.map(d => this.adaptSensorData(d));
    const lordosisValues = adapted.map(d => d.lumbarLordosis || 0);
    const sagittalValues = adapted.map(d => d.sagittalTilt || 0);
    const lateralValues = adapted.map(d => Math.abs(d.lateralTilt || 0));
    const rotationValues = adapted.map(d => Math.abs(d.rotation || 0));

    const lordosisRange = Math.max(...lordosisValues) - Math.min(...lordosisValues);
    this.qualityMetrics.depth = Math.min((lordosisRange / 40) * 100, 100);

    const lateralVariance = this.calculateVariance(lateralValues);
    const rotationVariance = this.calculateVariance(rotationValues);
    this.qualityMetrics.stability = Math.max(0, 100 - (lateralVariance + rotationVariance) * 10);

    const avgSagittal = this.calculateMean(sagittalValues);
    this.qualityMetrics.posture = Math.max(0, 100 - Math.abs(avgSagittal) * 2);

    this.qualityMetrics.symmetry = Math.max(0, 100 - this.calculateMean(lateralValues) * 5);

    this.qualityMetrics.overall = (
      this.qualityMetrics.depth * 0.3 +
      this.qualityMetrics.stability * 0.3 +
      this.qualityMetrics.posture * 0.2 +
      this.qualityMetrics.symmetry * 0.2
    );

    return this.qualityMetrics;
  }

  private assessDeadliftQuality(): QualityMetrics {
    if (this.dataBuffer.length < 5) return this.qualityMetrics;

    const adapted = this.dataBuffer.map(d => this.adaptSensorData(d));
    const sagittalValues = adapted.map(d => d.sagittalTilt || 0);
    const lordosisValues = adapted.map(d => d.lumbarLordosis || 0);
    const lateralValues = adapted.map(d => Math.abs(d.lateralTilt || 0));

    const sagittalRange = Math.max(...sagittalValues) - Math.min(...sagittalValues);
    this.qualityMetrics.depth = Math.min((sagittalRange / 30) * 100, 100);

    const lateralVariance = this.calculateVariance(lateralValues);
    this.qualityMetrics.stability = Math.max(0, 100 - lateralVariance * 15);

    const lordosisVariance = this.calculateVariance(lordosisValues);
    this.qualityMetrics.posture = Math.max(0, 100 - lordosisVariance * 2);

    this.qualityMetrics.symmetry = Math.max(0, 100 - this.calculateMean(lateralValues) * 5);

    this.qualityMetrics.overall = (
      this.qualityMetrics.depth * 0.25 +
      this.qualityMetrics.stability * 0.35 +
      this.qualityMetrics.posture * 0.25 +
      this.qualityMetrics.symmetry * 0.15
    );

    return this.qualityMetrics;
  }

  private assessPlankQuality(): QualityMetrics {
    if (this.dataBuffer.length < 5) return this.qualityMetrics;

    const adapted = this.dataBuffer.map(d => this.adaptSensorData(d));
    const sagittalValues = adapted.map(d => d.sagittalTilt || 0);
    const lateralValues = adapted.map(d => Math.abs(d.lateralTilt || 0));
    const accelValues = adapted.map(d => {
      const accel = d.acceleration;
      return typeof accel === 'number' 
        ? accel 
        : (accel ? Math.sqrt(accel.x**2 + accel.y**2 + accel.z**2) : 0);
    });

    this.qualityMetrics.depth = 100;

    const accelVariance = this.calculateVariance(accelValues);
    this.qualityMetrics.stability = Math.max(0, 100 - accelVariance * 50);

    const avgSagittal = Math.abs(this.calculateMean(sagittalValues));
    this.qualityMetrics.posture = Math.max(0, 100 - avgSagittal * 5);

    this.qualityMetrics.symmetry = Math.max(0, 100 - this.calculateMean(lateralValues) * 10);

    this.qualityMetrics.overall = (
      this.qualityMetrics.stability * 0.4 +
      this.qualityMetrics.posture * 0.3 +
      this.qualityMetrics.symmetry * 0.3
    );

    return this.qualityMetrics;
  }

  private assessPushupQuality(): QualityMetrics {
    if (this.dataBuffer.length < 5) return this.qualityMetrics;

    const adapted = this.dataBuffer.map(d => this.adaptSensorData(d));
    const lordosisValues = adapted.map(d => d.lumbarLordosis || 0);
    const lateralValues = adapted.map(d => Math.abs(d.lateralTilt || 0));
    const accelValues = adapted.map(d => {
      const accel = d.acceleration;
      return typeof accel === 'number' 
        ? accel 
        : (accel ? Math.sqrt(accel.x**2 + accel.y**2 + accel.z**2) : 0);
    });

    const accelRange = Math.max(...accelValues) - Math.min(...accelValues);
    this.qualityMetrics.depth = Math.min((accelRange / 5) * 100, 100);

    const lateralVariance = this.calculateVariance(lateralValues);
    this.qualityMetrics.stability = Math.max(0, 100 - lateralVariance * 15);

    const lordosisVariance = this.calculateVariance(lordosisValues);
    this.qualityMetrics.posture = Math.max(0, 100 - lordosisVariance * 2);

    this.qualityMetrics.symmetry = Math.max(0, 100 - this.calculateMean(lateralValues) * 5);

    this.qualityMetrics.overall = (
      this.qualityMetrics.depth * 0.3 +
      this.qualityMetrics.stability * 0.3 +
      this.qualityMetrics.posture * 0.2 +
      this.qualityMetrics.symmetry * 0.2
    );

    return this.qualityMetrics;
  }

  generateFeedback(exercise: string | null, metrics: QualityMetrics): FeedbackItem[] {
    this.feedback = [];

    if (!exercise) {
      this.feedback.push({
        type: 'info',
        message: 'Waiting for exercise detection...'
      });
      return this.feedback;
    }

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

  private calculateMean(values: number[]): number {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  private calculateVariance(values: number[]): number {
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

