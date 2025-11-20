// Navigation and View Management

export class Navigation {
    constructor() {
        this.currentView = 'landing';
        this.selectedCategory = null;
        this.selectedExercise = null;
        
        this.exercises = {
            'weightlifting': [
                { id: 'squat', name: 'Squat', icon: '🏋️', video: 'videos/squat.mp4' },
                { id: 'deadlift', name: 'Deadlift', icon: '💪', video: 'videos/deadlift.mp4' },
                { id: 'row', name: 'Row', icon: '🚣', video: 'videos/row.mp4' }
            ],
            'cardio': [
                { id: 'pushup', name: 'Push-up', icon: '🤸', video: 'videos/pushup.mp4' },
                { id: 'plank', name: 'Plank', icon: '🧘', video: 'videos/plank.mp4' },
                { id: 'burpee', name: 'Burpee', icon: '⚡', video: 'videos/burpee.mp4' }
            ],
            'yoga': [
                { id: 'warrior', name: 'Warrior Pose', icon: '🧘‍♀️', video: 'videos/warrior-pose.gif' },
                { id: 'downward', name: 'Downward Dog', icon: '🐕', video: 'videos/downward-dog-pose.gif' },
                { id: 'tree', name: 'Tree Pose', icon: '🌳', video: 'videos/tree-pose.gif' }
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
            let preview;
            if (exercise.video) {
                const isGif = exercise.video.toLowerCase().endsWith('.gif');
                if (isGif) {
                    preview = `<img class="exercise-preview-gif" src="${exercise.video}" alt="${exercise.name} preview" loading="lazy">`;
                } else {
                    preview = `<video class="exercise-preview-video" src="${exercise.video}" playsinline muted loop preload="metadata"></video>`;
                }
            } else {
                preview = `<div class="exercise-icon">${exercise.icon}</div>`;
            }

            card.innerHTML = `
                <div class="exercise-preview">${preview}</div>
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

