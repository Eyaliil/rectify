import { useState, useEffect, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { FlexTail3DViewer } from '@/components/FlexTail3DViewer';
import { ExerciseAnalyzer, SensorData, QualityMetrics, FeedbackItem, ExerciseRecognition } from '@/lib/exerciseAnalyzer';
import { Play, Square, ArrowLeft, AlertCircle, CheckCircle, AlertTriangle, Info } from 'lucide-react';

interface Exercise {
  id: string;
  name: string;
  description: string;
  difficulty: string;
  category: string;
  points_reward: number;
}

const exerciseVideos: Record<string, string> = {
  'squat': '/videos/squat.mp4',
  'deadlift': '/videos/deadlift.mp4',
  'row': '/videos/row.mp4',
  'pushup': '/videos/pushup.mp4',
  'plank': '/videos/plank.mp4',
  'burpee': '/videos/burpee.mp4',
  'warrior': '/videos/warrior-pose.gif',
  'downward': '/videos/downward-dog-pose.gif',
  'tree': '/videos/tree-pose.gif'
};

const exerciseAnalysisMap: Record<string, string> = {
  'squat': 'squat',
  'deadlift': 'deadlift',
  'row': 'deadlift',
  'pushup': 'pushup',
  'plank': 'plank',
  'burpee': 'pushup',
  'warrior': 'squat',
  'downward': 'plank',
  'tree': 'plank'
};

const exercisesByCategory: Record<string, Exercise[]> = {
  'weight lift': [
    {
      id: 'squat',
      name: 'Squat',
      description: 'Lower body strength exercise for legs and glutes',
      difficulty: 'beginner',
      category: 'weight lift',
      points_reward: 15
    },
    {
      id: 'deadlift',
      name: 'Deadlift',
      description: 'Full body compound movement for strength and power',
      difficulty: 'advanced',
      category: 'weight lift',
      points_reward: 25
    },
    {
      id: 'row',
      name: 'Row',
      description: 'Upper body pulling exercise targeting back, shoulders, and arms',
      difficulty: 'intermediate',
      category: 'weight lift',
      points_reward: 20
    }
  ],
  'cardio': [
    {
      id: 'pushup',
      name: 'Push-up',
      description: 'Upper body strength exercise targeting chest, shoulders, and triceps',
      difficulty: 'intermediate',
      category: 'cardio',
      points_reward: 18
    },
    {
      id: 'plank',
      name: 'Plank',
      description: 'Core strengthening exercise for stability and endurance',
      difficulty: 'beginner',
      category: 'cardio',
      points_reward: 15
    },
    {
      id: 'burpee',
      name: 'Burpee',
      description: 'Full body high-intensity exercise combining squat, plank, and jump',
      difficulty: 'advanced',
      category: 'cardio',
      points_reward: 22
    }
  ],
  'yoga': [
    {
      id: 'warrior',
      name: 'Warrior Pose',
      description: 'Dynamic pose focusing on strength, balance, and flexibility',
      difficulty: 'intermediate',
      category: 'yoga',
      points_reward: 16
    },
    {
      id: 'downward',
      name: 'Downward Dog Pose',
      description: 'Inverted pose for flexibility, strength, and mindfulness',
      difficulty: 'beginner',
      category: 'yoga',
      points_reward: 12
    },
    {
      id: 'tree',
      name: 'Tree Pose',
      description: 'Balancing pose for focus, stability, and flexibility',
      difficulty: 'beginner',
      category: 'yoga',
      points_reward: 14
    }
  ]
};

const Exercises = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const category = searchParams.get('category') || '';
  const { toast } = useToast();

  const exercises = exercisesByCategory[category] || [];
  const [selectedExercise, setSelectedExercise] = useState<Exercise | null>(exercises[0] || null);
  const [isActive, setIsActive] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  
  // Analysis state
  const [analyzer] = useState(() => new ExerciseAnalyzer());
  const [recognition, setRecognition] = useState<ExerciseRecognition | null>(null);
  const [metrics, setMetrics] = useState<QualityMetrics>({
    overall: 0,
    depth: 0,
    stability: 0,
    posture: 0,
    symmetry: 0
  });
  const [feedback, setFeedback] = useState<FeedbackItem[]>([]);
  const [sensorData, setSensorData] = useState<SensorData | null>(null);
  const [measurement, setMeasurement] = useState<any>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const gifRef = useRef<HTMLImageElement>(null);
  const intervalRef = useRef<number | null>(null);

  // Update video when exercise changes
  useEffect(() => {
    if (selectedExercise && exerciseVideos[selectedExercise.id]) {
      const videoSource = exerciseVideos[selectedExercise.id];
      if (!videoRef.current || !gifRef.current) return;

      const isGif = videoSource.toLowerCase().endsWith('.gif');
      if (isGif) {
        gifRef.current.src = videoSource;
        gifRef.current.style.display = 'block';
        if (videoRef.current.src) {
          videoRef.current.src = '';
          videoRef.current.load();
        }
        videoRef.current.style.display = 'none';
      } else {
        if (videoRef.current.src !== videoSource) {
          videoRef.current.src = videoSource;
          videoRef.current.load();
        }
        videoRef.current.style.display = 'block';
        gifRef.current.src = '';
        gifRef.current.style.display = 'none';
      }
    }
  }, [selectedExercise]);

  // Handle analysis when exercise is active
  useEffect(() => {
    if (isActive && showAnalysis) {
      intervalRef.current = window.setInterval(() => {
        generateMockSensorData();
      }, 100);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (!isActive) {
        analyzer.reset();
        setRecognition(null);
        setMetrics({
          overall: 0,
          depth: 0,
          stability: 0,
          posture: 0,
          symmetry: 0
        });
        setFeedback([]);
        setSensorData(null);
        setMeasurement(null);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isActive, showAnalysis, selectedExercise]);

  const generateMockSensorData = () => {
    if (!selectedExercise) return;
    const analysisType = exerciseAnalysisMap[selectedExercise.id] || 'squat';
    
    const baseData: SensorData = {
      lumbarAngle: 0.5 + Math.sin(Date.now() / 1000) * 0.3,
      sagittal: Math.sin(Date.now() / 800) * 0.2,
      lateral: Math.sin(Date.now() / 1200) * 0.1,
      twist: Math.sin(Date.now() / 1500) * 0.05,
      acceleration: 2 + Math.abs(Math.sin(Date.now() / 500)) * 3
    };

    ingestSensorData(baseData, analysisType);
  };

  const convertToFlexTailMeasurement = (rawData: SensorData): any => {
    if (!rawData) return null;

    const radiansToDegrees = (rad: number) => rad * (180 / Math.PI);
    const degreesToRadians = (deg: number) => deg * (Math.PI / 180);
    
    const lumbarAngle = rawData.lumbarAngle !== undefined 
      ? rawData.lumbarAngle 
      : degreesToRadians(rawData.lumbarLordosis || 0);
    
    const sagittal = rawData.sagittal !== undefined 
      ? rawData.sagittal 
      : degreesToRadians(rawData.sagittalTilt || 0);
    
    const lateral = rawData.lateral !== undefined 
      ? rawData.lateral 
      : degreesToRadians(rawData.lateralTilt || 0);
    
    const twist = rawData.twist !== undefined 
      ? rawData.twist 
      : degreesToRadians(rawData.rotation || 0);

    const numPoints = 18;
    const length = 360;
    const width = 20;
    
    const left = { x: [] as number[], y: [] as number[], z: [] as number[] };
    const right = { x: [] as number[], y: [] as number[], z: [] as number[] };
    const center = { x: [] as number[], y: [] as number[], z: [] as number[] };
    
    for (let i = 0; i < numPoints; i++) {
      const t = i / (numPoints - 1);
      const z = length * t;
      
      const bendCurve = Math.sin(t * Math.PI) * radiansToDegrees(lumbarAngle) * 0.5;
      const twistAngle = t * radiansToDegrees(twist);
      const cosT = Math.cos(twistAngle * Math.PI / 180);
      const sinT = Math.sin(twistAngle * Math.PI / 180);
      
      const sagittalOffset = radiansToDegrees(sagittal) * t * 0.3;
      const lateralOffset = radiansToDegrees(lateral) * 0.5;
      
      left.x.push((-width / 2 + lateralOffset) * cosT - (bendCurve + sagittalOffset) * sinT);
      left.y.push((-width / 2 + lateralOffset) * sinT + (bendCurve + sagittalOffset) * cosT);
      left.z.push(z);
      
      right.x.push((width / 2 + lateralOffset) * cosT - (bendCurve + sagittalOffset) * sinT);
      right.y.push((width / 2 + lateralOffset) * sinT + (bendCurve + sagittalOffset) * cosT);
      right.z.push(z);
      
      center.x.push((bendCurve + sagittalOffset) * sinT);
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
  };

  const adaptSensorData = (rawData: SensorData): SensorData => {
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
  };

  const ingestSensorData = (rawSensorData: SensorData, exerciseType: string) => {
    if (!isActive || !rawSensorData) return;

    const adaptedData = adaptSensorData(rawSensorData);
    setSensorData(adaptedData);

    // Update FlexTail visualization
    const flextailMeasurement = convertToFlexTailMeasurement(rawSensorData);
    setMeasurement(flextailMeasurement);

    // Recognize exercise
    const recognitionResult = analyzer.recognizeExercise(adaptedData);
    setRecognition(recognitionResult);
    
    // Use mapped exercise type
    const exercise = exerciseType;

    // Assess quality
    const qualityMetrics = analyzer.assessQuality(adaptedData, exercise);
    setMetrics(qualityMetrics);

    // Generate feedback
    const feedbackItems = analyzer.generateFeedback(exercise, qualityMetrics);
    setFeedback(feedbackItems);
  };

  const startExercise = () => {
    if (!selectedExercise) return;
    setIsActive(true);
    setShowAnalysis(true);
    analyzer.reset();
    toast({ title: 'Exercise started!', description: `Let's go! ${selectedExercise.name}` });
  };

  const stopExercise = () => {
    if (!selectedExercise) return;
    setIsActive(false);
    setShowAnalysis(false);
    toast({ 
      title: 'Great work!', 
      description: `+${selectedExercise.points_reward} points earned!` 
    });
  };

  const getFeedbackIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Info className="h-4 w-4 text-blue-500" />;
    }
  };

  const getRiskLevel = (score: number) => {
    if (score >= 80) return { level: 'Safe', badge: 'bg-green-500', text: 'SAFE' };
    if (score >= 60) return { level: 'Caution', badge: 'bg-yellow-500', text: 'CAUTION' };
    return { level: 'Warning', badge: 'bg-red-500', text: 'WARNING' };
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-success';
      case 'intermediate': return 'bg-warning';
      case 'advanced': return 'bg-danger';
      default: return 'bg-muted';
    }
  };

  const getCategoryTitle = (cat: string) => {
    switch (cat) {
      case 'weight lift': return 'Weight Lifting';
      case 'cardio': return 'Cardio';
      case 'yoga': return 'Yoga';
      default: return cat;
    }
  };

  if (!category || !exercises.length) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle>Category Not Found</CardTitle>
            <CardDescription>Please select a valid category</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => navigate('/dashboard')} className="w-full">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Dashboard
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-4">
            <Button variant="ghost" onClick={() => navigate('/dashboard')}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                {getCategoryTitle(category)}
              </h1>
              <p className="text-sm text-muted-foreground">Choose an exercise to start</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Card className="border-border/50 shadow-card">
          <CardHeader>
            <CardTitle>Select Exercise</CardTitle>
            <CardDescription>Choose an exercise from the {getCategoryTitle(category)} category</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {exercises.map((exercise) => (
                <Card
                  key={exercise.id}
                  className={`cursor-pointer transition-all hover:shadow-lg ${
                    selectedExercise?.id === exercise.id
                      ? 'border-primary ring-2 ring-primary'
                      : 'border-border'
                  }`}
                  onClick={() => !isActive && setSelectedExercise(exercise)}
                >
                  <CardContent className="p-4">
                    <div className="space-y-3">
                      <div className="flex items-start justify-between">
                        <h3 className="font-semibold text-lg">{exercise.name}</h3>
                        <Badge className={getDifficultyColor(exercise.difficulty)}>
                          {exercise.difficulty}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{exercise.description}</p>
                      <div className="flex items-center justify-between pt-2 border-t border-border">
                        <span className="text-sm text-muted-foreground">Reward:</span>
                        <span className="text-sm font-semibold text-secondary">
                          +{exercise.points_reward} pts
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {selectedExercise && (
              <div className="mt-6 space-y-6">
                {/* Exercise Info */}
                <div className="p-4 rounded-lg bg-card border border-border space-y-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-semibold text-xl">{selectedExercise.name}</h3>
                      <p className="text-sm text-muted-foreground mt-1">{selectedExercise.description}</p>
                    </div>
                    <Badge className={getDifficultyColor(selectedExercise.difficulty)}>
                      {selectedExercise.difficulty}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-muted-foreground">
                      Category: <span className="text-foreground">{getCategoryTitle(selectedExercise.category)}</span>
                    </span>
                    <span className="text-muted-foreground">
                      Reward: <span className="text-secondary font-semibold">+{selectedExercise.points_reward} pts</span>
                    </span>
                  </div>
                </div>

                {/* Form Analysis (shown when exercise is active) */}
                {isActive && showAnalysis && (
                  <div className="grid grid-cols-1 lg:grid-cols-1 gap-6">
                    {/* Form Analysis */}
                    <Card>
                      <CardHeader>
                        <CardTitle>Form Analysis</CardTitle>
                        <CardDescription>Real-time quality metrics</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {/* Overall Score */}
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-sm">
                            <span>Form Correctness</span>
                            <span className="font-semibold">
                              {metrics.overall > 0 ? `${metrics.overall.toFixed(0)}%` : '--'}
                            </span>
                          </div>
                          <Progress value={metrics.overall} />
                        </div>

                        {/* Risk Indicator */}
                        {metrics.overall > 0 && (
                          <div className={`p-3 rounded-lg ${getRiskLevel(metrics.overall).badge} bg-opacity-10 border ${getRiskLevel(metrics.overall).badge} border-opacity-20`}>
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="text-xs text-muted-foreground mb-1">Risk Level</p>
                                <p className="font-semibold">{getRiskLevel(metrics.overall).level}</p>
                              </div>
                              <Badge className={getRiskLevel(metrics.overall).badge}>
                                {getRiskLevel(metrics.overall).text}
                              </Badge>
                            </div>
                          </div>
                        )}

                        {/* Quality Metrics */}
                        <div className="space-y-3 pt-2">
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                              <span>Depth</span>
                              <span className="font-semibold">
                                {metrics.depth > 0 ? `${metrics.depth.toFixed(0)}%` : '--'}
                              </span>
                            </div>
                            <Progress value={metrics.depth} />
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                              <span>Stability</span>
                              <span className="font-semibold">
                                {metrics.stability > 0 ? `${metrics.stability.toFixed(0)}%` : '--'}
                              </span>
                            </div>
                            <Progress value={metrics.stability} />
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                              <span>Posture</span>
                              <span className="font-semibold">
                                {metrics.posture > 0 ? `${metrics.posture.toFixed(0)}%` : '--'}
                              </span>
                            </div>
                            <Progress value={metrics.posture} />
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                              <span>Symmetry</span>
                              <span className="font-semibold">
                                {metrics.symmetry > 0 ? `${metrics.symmetry.toFixed(0)}%` : '--'}
                              </span>
                            </div>
                            <Progress value={metrics.symmetry} />
                          </div>
                        </div>

                        {/* AI Feedback */}
                        {feedback.length > 0 && (
                          <div className="space-y-2 pt-2 border-t">
                            <p className="text-sm font-semibold">AI Coach Feedback</p>
                            {feedback.map((item, index) => (
                              <div
                                key={index}
                                className="flex items-start gap-2 p-2 rounded-lg bg-muted/50"
                              >
                                {getFeedbackIcon(item.type)}
                                <p className="text-sm flex-1">{item.message}</p>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Sensor Data */}
                        {sensorData && (
                          <div className="space-y-2 pt-2 border-t text-sm">
                            <p className="font-semibold mb-2">Sensor Measurements</p>
                            <div className="grid grid-cols-2 gap-2">
                              <div className="flex justify-between">
                                <span className="text-muted-foreground">Lumbar</span>
                                <span className="font-semibold">
                                  {sensorData.lumbarLordosis?.toFixed(1)}°
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-muted-foreground">Sagittal</span>
                                <span className="font-semibold">
                                  {sensorData.sagittalTilt?.toFixed(1)}°
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-muted-foreground">Lateral</span>
                                <span className="font-semibold">
                                  {sensorData.lateralTilt?.toFixed(1)}°
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-muted-foreground">Rotation</span>
                                <span className="font-semibold">
                                  {sensorData.rotation?.toFixed(1)}°
                                </span>
                              </div>
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </div>
                )}
              </div>
            )}

            {/* Video and Spine Visualization Side by Side */}
            {selectedExercise && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Exercise Video Preview - Left Side */}
                {exerciseVideos[selectedExercise.id] && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Exercise Demo</CardTitle>
                      <CardDescription className="text-xs">Watch the correct form</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
                        <video
                          ref={videoRef}
                          className="w-full h-full object-cover"
                          playsInline
                          muted
                          loop
                          preload="metadata"
                          controls
                        />
                        <img
                          ref={gifRef}
                          className="w-full h-full object-cover hidden"
                          alt="Exercise demo preview"
                        />
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Spine Visualization - Right Side */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Spine Visualization</CardTitle>
                    <CardDescription className="text-xs">Real-time 3D spine analysis</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="aspect-video bg-muted rounded-lg overflow-hidden">
                      <FlexTail3DViewer measurement={measurement} />
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Exercise Controls */}
            {selectedExercise && (
              <div className="flex gap-2">
                <Button
                  className="flex-1"
                  size="lg"
                  onClick={isActive ? stopExercise : startExercise}
                  disabled={!selectedExercise}
                >
                  {isActive ? (
                    <>
                      <Square className="h-5 w-5 mr-2" />
                      Stop Exercise
                    </>
                  ) : (
                    <>
                      <Play className="h-5 w-5 mr-2" />
                      Start Exercise
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => {
                    const exerciseId = selectedExercise?.id || 'auto';
                    navigate(`/exercise-analysis?exercise=${exerciseId}&category=${category}`);
                  }}
                  disabled={!selectedExercise}
                >
                  Analyze Form
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default Exercises;

