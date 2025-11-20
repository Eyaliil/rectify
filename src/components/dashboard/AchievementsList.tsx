import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Lock } from 'lucide-react';

interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  points: number;
  unlocked?: boolean;
}

const AchievementsList = () => {
  // Mock achievements data
  const achievements: Achievement[] = [
    {
      id: '1',
      name: 'First Steps',
      description: 'Complete your first exercise',
      icon: '🎯',
      points: 10,
      unlocked: false
    },
    {
      id: '2',
      name: 'Week Warrior',
      description: 'Maintain a 7-day streak',
      icon: '🔥',
      points: 50,
      unlocked: false
    },
    {
      id: '3',
      name: 'Perfect Form',
      description: 'Achieve 100% form score',
      icon: '⭐',
      points: 100,
      unlocked: false
    }
  ];

  return (
    <Card className="border-border/50 shadow-card h-fit">
      <CardHeader>
        <CardTitle>Achievements</CardTitle>
        <CardDescription>Unlock rewards as you progress</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {achievements.map((achievement) => (
          <div
            key={achievement.id}
            className={`p-4 rounded-lg border transition-all ${
              achievement.unlocked
                ? 'bg-gradient-to-br from-primary/10 to-primary/5 border-primary/20'
                : 'bg-muted/30 border-border opacity-60'
            }`}
          >
            <div className="flex items-start gap-3">
              <div className="text-3xl">{achievement.unlocked ? achievement.icon : <Lock className="h-6 w-6 text-muted-foreground" />}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2">
                  <h4 className="font-semibold text-foreground">{achievement.name}</h4>
                  <Badge variant={achievement.unlocked ? 'default' : 'secondary'} className="shrink-0">
                    {achievement.points} pts
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground mt-1">{achievement.description}</p>
              </div>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};

export default AchievementsList;
