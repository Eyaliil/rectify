import { useState, useEffect, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { FlexTail3DViewer } from '@/components/FlexTail3DViewer';
import { ExerciseAnalyzer, SensorData, QualityMetrics, FeedbackItem, ExerciseRecognition } from '@/lib/exerciseAnalyzer';
import { ArrowLeft, Play, Square, AlertCircle, CheckCircle, AlertTriangle, Info } from 'lucide-react';

const exerciseVideos: Record<string, string> = {
  squat: '/videos/squat.mp4',
  deadlift: '/videos/deadlift.mp4',
  row: '/videos/row.mp4',
  pushup: '/videos/pushup.mp4',
  plank: '/videos/plank.mp4',
  burpee: '/videos/burpee.mp4',
  warrior: '/videos/warrior-pose.gif',
  downward: '/videos/downward-dog-pose.gif',
  tree: '/videos/tree-pose.gif'
};

const exerciseMap: Record<string, string> = {
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

const ExerciseAnalysis = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const exerciseId = searchParams.get('exercise') || 'auto';
  const category = searchParams.get('category') || '';
  
  const [selectedExercise, setSelectedExercise] = useState<string>(exerciseId);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
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

  useEffect(() => {
    updateExerciseVideo(selectedExercise);
  }, [selectedExercise]);

  useEffect(() => {
    if (isAnalyzing) {
      // Simulate sensor data for demo purposes
      // In production, this would come from actual FlexTail sensors
      intervalRef.current = window.setInterval(() => {
        generateMockSensorData();
      }, 100); // Update every 100ms
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
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

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isAnalyzing, selectedExercise]);

  const generateMockSensorData = () => {
    // Generate mock sensor data based on exercise type
    const exercise = selectedExercise !== 'auto' ? selectedExercise : 'squat';
    const baseData: SensorData = {
      lumbarAngle: 0.5 + Math.sin(Date.now() / 1000) * 0.3,
      sagittal: Math.sin(Date.now() / 800) * 0.2,
      lateral: Math.sin(Date.now() / 1200) * 0.1,
      twist: Math.sin(Date.now() / 1500) * 0.05,
      acceleration: 2 + Math.abs(Math.sin(Date.now() / 500)) * 3
    };

    ingestSensorData(baseData);
  };

  const updateExerciseVideo = (exerciseId: string) => {
    const videoSource = exerciseVideos[exerciseId];
    if (!videoRef.current || !gifRef.current) return;

    if (videoSource) {
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

  const ingestSensorData = (rawSensorData: SensorData) => {
    if (!isAnalyzing || !rawSensorData) return;

    const adaptedData = adaptSensorData(rawSensorData);
    setSensorData(adaptedData);

    // Update FlexTail visualization
    const flextailMeasurement = convertToFlexTailMeasurement(rawSensorData);
    setMeasurement(flextailMeasurement);

    // Recognize exercise
    const recognitionResult = analyzer.recognizeExercise(adaptedData);
    setRecognition(recognitionResult);
    
    // Use selected exercise if not auto-detect
    const exercise = selectedExercise !== 'auto' 
      ? exerciseMap[selectedExercise] || selectedExercise
      : recognitionResult.exercise;

    // Assess quality
    const qualityMetrics = analyzer.assessQuality(adaptedData, exercise);
    setMetrics(qualityMetrics);

    // Generate feedback
    const feedbackItems = analyzer.generateFeedback(exercise, qualityMetrics);
    setFeedback(feedbackItems);
  };

  const startAnalysis = () => {
    setIsAnalyzing(true);
    analyzer.reset();
  };

  const stopAnalysis = () => {
    setIsAnalyzing(false);
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

  const risk = getRiskLevel(metrics.overall);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-4">
            <Button variant="ghost" onClick={() => navigate(`/exercises?category=${category}`)}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                Exercise Analysis
              </h1>
              <p className="text-sm text-muted-foreground">AI-Powered Exercise Analysis & Quality Assessment</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Visualizations Card */}
            <Card>
              <CardHeader>
                <CardTitle>3D Visualizations</CardTitle>
                <CardDescription>Real-time body and spine analysis</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Exercise Video */}
                  <div className="space-y-2">
                    <h3 className="text-sm font-semibold">Exercise Demo</h3>
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
                  </div>

                  {/* Spine Visualization */}
                  <div className="space-y-2">
                    <h3 className="text-sm font-semibold">Spine Visualization</h3>
                    <div className="aspect-video bg-muted rounded-lg overflow-hidden">
                      <FlexTail3DViewer measurement={measurement} />
                    </div>
                  </div>
                </div>

                {/* Controls */}
                <div className="flex items-center gap-4 pt-4 border-t">
                  <Button
                    onClick={isAnalyzing ? stopAnalysis : startAnalysis}
                    disabled={selectedExercise === 'auto' && !recognition?.exercise}
                    className="flex-1"
                  >
                    {isAnalyzing ? (
                      <>
                        <Square className="h-4 w-4 mr-2" />
                        Stop Analysis
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Start Analysis
                      </>
                    )}
                  </Button>
                  <Select value={selectedExercise} onValueChange={setSelectedExercise}>
                    <SelectTrigger className="w-48">
                      <SelectValue placeholder="Select exercise" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto-detect</SelectItem>
                      <SelectItem value="squat">Squat</SelectItem>
                      <SelectItem value="deadlift">Deadlift</SelectItem>
                      <SelectItem value="plank">Plank</SelectItem>
                      <SelectItem value="pushup">Push-up</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Side Panel */}
          <div className="space-y-6">
            {/* Exercise Recognition */}
            <Card>
              <CardHeader>
                <CardTitle>Exercise Recognition</CardTitle>
                <CardDescription>Detected exercise and confidence</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">
                    {recognition?.exercise 
                      ? recognition.exercise.charAt(0).toUpperCase() + recognition.exercise.slice(1)
                      : 'Detecting...'}
                  </div>
                  {recognition?.confidence && (
                    <div className="text-sm text-muted-foreground">
                      Confidence: {(recognition.confidence * 100).toFixed(0)}%
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Form Analysis */}
            <Card>
              <CardHeader>
                <CardTitle>Real-Time Form Analysis</CardTitle>
                <CardDescription>AI-powered feedback from FlexTail sensors</CardDescription>
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
                  <div className={`p-3 rounded-lg ${risk.badge} bg-opacity-10 border ${risk.badge} border-opacity-20`}>
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Risk Level</p>
                        <p className="font-semibold">{risk.level}</p>
                      </div>
                      <Badge className={risk.badge}>{risk.text}</Badge>
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
              </CardContent>
            </Card>

            {/* AI Feedback */}
            <Card>
              <CardHeader>
                <CardTitle>AI Coach Feedback</CardTitle>
                <CardDescription>Real-time guidance and corrections</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {feedback.length > 0 ? (
                    feedback.map((item, index) => (
                      <div
                        key={index}
                        className="flex items-start gap-2 p-2 rounded-lg bg-muted/50"
                      >
                        {getFeedbackIcon(item.type)}
                        <p className="text-sm flex-1">{item.message}</p>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground">No feedback yet</p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Sensor Data */}
            <Card>
              <CardHeader>
                <CardTitle>Sensor Measurements</CardTitle>
                <CardDescription>Live sensor data from FlexTail device</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Lumbar Lordosis</span>
                    <span className="font-semibold">
                      {sensorData?.lumbarLordosis !== undefined 
                        ? `${sensorData.lumbarLordosis.toFixed(1)}°` 
                        : '--'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Sagittal Tilt</span>
                    <span className="font-semibold">
                      {sensorData?.sagittalTilt !== undefined 
                        ? `${sensorData.sagittalTilt.toFixed(1)}°` 
                        : '--'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Lateral Tilt</span>
                    <span className="font-semibold">
                      {sensorData?.lateralTilt !== undefined 
                        ? `${sensorData.lateralTilt.toFixed(1)}°` 
                        : '--'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Rotation</span>
                    <span className="font-semibold">
                      {sensorData?.rotation !== undefined 
                        ? `${sensorData.rotation.toFixed(1)}°` 
                        : '--'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Acceleration</span>
                    <span className="font-semibold">
                      {sensorData?.acceleration 
                        ? (typeof sensorData.acceleration === 'number'
                            ? `${sensorData.acceleration.toFixed(2)} m/s²`
                            : `${Math.sqrt(
                                sensorData.acceleration.x**2 + 
                                sensorData.acceleration.y**2 + 
                                sensorData.acceleration.z**2
                              ).toFixed(2)} m/s²`)
                        : '--'}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ExerciseAnalysis;

