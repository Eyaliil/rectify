import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  gradient: string;
}

const StatsCard = ({ title, value, icon, gradient }: StatsCardProps) => {
  return (
    <Card className="overflow-hidden border-border/50 shadow-card">
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground font-medium">{title}</p>
            <p className="text-3xl font-bold text-foreground">{value}</p>
          </div>
          <div className={cn('p-3 rounded-xl', gradient)}>
            <div className="text-primary-foreground">{icon}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default StatsCard;
