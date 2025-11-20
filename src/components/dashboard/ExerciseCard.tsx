import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Dumbbell, Heart, Flower2 } from 'lucide-react';

const ExerciseCard = () => {
  const navigate = useNavigate();

  const categories = [
    {
      id: 'weight lift',
      name: 'Weight Lifting',
      description: 'Build strength and muscle with weight training exercises',
      icon: <Dumbbell className="h-8 w-8" />,
      gradient: 'bg-gradient-primary'
    },
    {
      id: 'cardio',
      name: 'Cardio',
      description: 'Improve cardiovascular health and endurance',
      icon: <Heart className="h-8 w-8" />,
      gradient: 'bg-gradient-accent'
    },
    {
      id: 'yoga',
      name: 'Yoga',
      description: 'Enhance flexibility, balance, and mindfulness',
      icon: <Flower2 className="h-8 w-8" />,
      gradient: 'bg-gradient-primary'
    }
  ];

  const handleCategorySelect = (categoryId: string) => {
    navigate(`/exercises?category=${categoryId}`);
  };

  return (
    <Card className="border-border/50 shadow-card">
      <CardHeader>
        <CardTitle>Start Exercise</CardTitle>
        <CardDescription>Choose a category to begin your workout</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {categories.map((category) => (
            <Card
              key={category.id}
              className="cursor-pointer transition-all hover:shadow-lg hover:scale-105 border-border"
              onClick={() => handleCategorySelect(category.id)}
            >
              <CardContent className="p-6 text-center space-y-4">
                <div className={`p-4 rounded-lg ${category.gradient} w-fit mx-auto`}>
                  <div className="text-primary-foreground">{category.icon}</div>
                </div>
                <div>
                  <h3 className="font-semibold text-lg mb-2">{category.name}</h3>
                  <p className="text-sm text-muted-foreground">{category.description}</p>
                </div>
                <Button className="w-full" variant="outline">
                  Select Category
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default ExerciseCard;
