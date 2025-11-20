import { Activity, Award, Flame, TrendingUp } from 'lucide-react';
import StatsCard from '@/components/dashboard/StatsCard';
import ExerciseCard from '@/components/dashboard/ExerciseCard';
import FormAnalysis from '@/components/dashboard/FormAnalysis';
import AchievementsList from '@/components/dashboard/AchievementsList';

const Dashboard = () => {
  // Mock profile data
  const profile = {
    username: 'User',
    total_points: 0,
    current_streak: 0,
    level: 1
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              FlexCoach AI
            </h1>
            <p className="text-sm text-muted-foreground">Welcome, {profile.username}!</p>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatsCard
            title="Total Points"
            value={profile.total_points}
            icon={<TrendingUp className="h-5 w-5" />}
            gradient="bg-gradient-primary"
          />
          <StatsCard
            title="Current Streak"
            value={`${profile.current_streak} days`}
            icon={<Flame className="h-5 w-5" />}
            gradient="bg-gradient-accent"
          />
          <StatsCard
            title="Level"
            value={profile.level}
            icon={<Award className="h-5 w-5" />}
            gradient="bg-gradient-primary"
          />
          <StatsCard
            title="Active Sessions"
            value={0}
            icon={<Activity className="h-5 w-5" />}
            gradient="bg-gradient-accent"
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Exercise & Form Analysis */}
          <div className="lg:col-span-2 space-y-6">
            <ExerciseCard />
            <FormAnalysis />
          </div>

          {/* Right Column: Achievements */}
          <div className="lg:col-span-1">
            <AchievementsList />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
