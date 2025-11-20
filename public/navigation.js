// Navigation and View Management

export class Navigation {
    constructor() {
        this.currentView = 'landing';
        this.selectedCategory = null;
        this.selectedExercise = null;
        
        this.exercises = {
            'weightlifting': [
                { id: 'squat', name: 'Squat', icon: '🏋️' },
                { id: 'deadlift', name: 'Deadlift', icon: '💪' },
                { id: 'row', name: 'Row', icon: '🚣' }
            ],
            'cardio': [
                { id: 'pushup', name: 'Push-up', icon: '🤸' },
                { id: 'plank', name: 'Plank', icon: '🧘' },
                { id: 'burpee', name: 'Burpee', icon: '⚡' }
            ],
            'yoga': [
                { id: 'warrior', name: 'Warrior Pose', icon: '🧘‍♀️' },
                { id: 'downward', name: 'Downward Dog', icon: '🐕' },
                { id: 'tree', name: 'Tree Pose', icon: '🌳' }
            ]
        };
    }

    showView(viewName, data = {}) {
        // Hide all views
        const views = document.querySelectorAll('.view');
        views.forEach(view => {
            view.style.display = 'none';
        });

        // Show requested view
        const targetView = document.getElementById(`${viewName}-view`);
        if (targetView) {
            targetView.style.display = 'block';
            this.currentView = viewName;
        }

        // Handle view-specific logic
        if (viewName === 'exercise') {
            this.renderExerciseSelection(data.category);
        } else if (viewName === 'analysis') {
            this.selectedExercise = data.exercise;
            // Initialize analysis page if needed
            if (window.app) {
                if (!window.app.initialized) {
                    window.app.initialize();
                }
                if (typeof window.app.setExercise === 'function') {
                    window.app.setExercise(data.exercise);
                }
            }
        }
    }

    renderExerciseSelection(category) {
        const container = document.getElementById('exercise-grid');
        if (!container) return;

        const exercises = this.exercises[category] || [];
        container.innerHTML = '';

        exercises.forEach(exercise => {
            const card = document.createElement('div');
            card.className = 'exercise-card';
            card.innerHTML = `
                <div class="exercise-icon">${exercise.icon}</div>
                <h3>${exercise.name}</h3>
            `;
            card.addEventListener('click', () => {
                this.selectedExercise = exercise.id;
                this.showView('analysis', { exercise: exercise.id });
            });
            container.appendChild(card);
        });
    }

    goBack() {
        if (this.currentView === 'analysis') {
            this.showView('exercise', { category: this.selectedCategory });
        } else if (this.currentView === 'exercise') {
            this.showView('landing');
            this.selectedCategory = null;
        }
    }
}

