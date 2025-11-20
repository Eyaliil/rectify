import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';

const FormAnalysis = () => {
  // Mock data - will be connected to Flextail device
  const formScore = 87;
  const riskLevel = 'caution';
  const feedback = 'Keep your back straight and engage your core throughout the movement';

  const getRiskConfig = (level: string) => {
    switch (level) {
      case 'safe':
        return { 
          color: 'bg-success', 
          icon: <CheckCircle2 className="h-5 w-5" />,
          label: 'Safe',
          gradient: 'from-success/20 to-success/5'
        };
      case 'caution':
        return { 
          color: 'bg-warning', 
          icon: <AlertTriangle className="h-5 w-5" />,
          label: 'Caution',
          gradient: 'from-warning/20 to-warning/5'
        };
      case 'warning':
        return { 
          color: 'bg-warning', 
          icon: <AlertCircle className="h-5 w-5" />,
          label: 'Warning',
          gradient: 'from-warning/20 to-warning/5'
        };
      case 'danger':
        return { 
          color: 'bg-danger', 
          icon: <XCircle className="h-5 w-5" />,
          label: 'Dangerous',
          gradient: 'from-danger/20 to-danger/5'
        };
      default:
        return { 
          color: 'bg-muted', 
          icon: <AlertCircle className="h-5 w-5" />,
          label: 'Unknown',
          gradient: 'from-muted/20 to-muted/5'
        };
    }
  };

  const riskConfig = getRiskConfig(riskLevel);

  return (
    <Card className="border-border/50 shadow-card">
      <CardHeader>
        <CardTitle>Real-Time Form Analysis</CardTitle>
        <CardDescription>AI-powered feedback from Flextail sensors</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Form Score */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Form Correctness</span>
            <span className="font-semibold text-foreground">{formScore}%</span>
          </div>
          <Progress value={formScore} className="h-3" />
        </div>

        {/* Risk Level Indicator */}
        <div className={`p-4 rounded-lg bg-gradient-to-br ${riskConfig.gradient} border border-border`}>
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${riskConfig.color}`}>
              <div className="text-background">{riskConfig.icon}</div>
            </div>
            <div className="flex-1">
              <p className="text-sm text-muted-foreground">Risk Level</p>
              <p className="font-semibold text-foreground">{riskConfig.label}</p>
            </div>
            <Badge className={riskConfig.color}>
              {riskConfig.label.toUpperCase()}
            </Badge>
          </div>
        </div>

        {/* AI Feedback */}
        <div className="p-4 rounded-lg bg-card border border-border">
          <p className="text-sm font-medium text-muted-foreground mb-2">AI Coach Feedback</p>
          <p className="text-foreground">{feedback}</p>
        </div>

        {/* Joint Angles (placeholder) */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 rounded-lg bg-muted/50 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Left Knee</p>
            <p className="text-lg font-semibold text-foreground">142°</p>
          </div>
          <div className="p-3 rounded-lg bg-muted/50 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Right Knee</p>
            <p className="text-lg font-semibold text-foreground">138°</p>
          </div>
          <div className="p-3 rounded-lg bg-muted/50 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Hip Angle</p>
            <p className="text-lg font-semibold text-foreground">95°</p>
          </div>
          <div className="p-3 rounded-lg bg-muted/50 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Back Angle</p>
            <p className="text-lg font-semibold text-foreground">12°</p>
          </div>
        </div>

        <p className="text-xs text-muted-foreground text-center">
          Connect Flextail device to receive real-time measurements
        </p>
      </CardContent>
    </Card>
  );
};

export default FormAnalysis;
