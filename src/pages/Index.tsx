import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Activity, TrendingUp, Award, Zap } from 'lucide-react';

const Index = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/dashboard');
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center space-y-6 mb-16">
          <h1 className="text-5xl md:text-7xl font-bold">
            <span className="bg-gradient-primary bg-clip-text text-transparent">
              FlexCoach AI
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
            Your intelligent fitness companion. Real-time form analysis, AI coaching, and gamified workouts for safer, better performance.
          </p>
          <Button 
            size="lg" 
            className="text-lg px-8 py-6 shadow-glow"
            onClick={handleGetStarted}
          >
            <Zap className="h-5 w-5 mr-2" />
            Get Started
          </Button>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <div className="p-6 rounded-xl bg-card border border-border shadow-card">
            <div className="p-3 rounded-lg bg-gradient-primary w-fit mb-4">
              <Activity className="h-6 w-6 text-primary-foreground" />
            </div>
            <h3 className="text-xl font-semibold mb-2 text-foreground">Real-Time Analysis</h3>
            <p className="text-muted-foreground">
              Connect your Flextail device for instant form feedback and risk assessment during every exercise.
            </p>
          </div>

          <div className="p-6 rounded-xl bg-card border border-border shadow-card">
            <div className="p-3 rounded-lg bg-gradient-accent w-fit mb-4">
              <TrendingUp className="h-6 w-6 text-accent-foreground" />
            </div>
            <h3 className="text-xl font-semibold mb-2 text-foreground">AI Coaching</h3>
            <p className="text-muted-foreground">
              Get personalized feedback and corrections powered by advanced AI to perfect your form and prevent injuries.
            </p>
          </div>

          <div className="p-6 rounded-xl bg-card border border-border shadow-card">
            <div className="p-3 rounded-lg bg-gradient-primary w-fit mb-4">
              <Award className="h-6 w-6 text-primary-foreground" />
            </div>
            <h3 className="text-xl font-semibold mb-2 text-foreground">Gamification</h3>
            <p className="text-muted-foreground">
              Earn points, unlock achievements, and maintain streaks. Make fitness fun and engaging like your favorite games!
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
