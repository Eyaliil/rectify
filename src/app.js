// Main App Initialization and Navigation Setup

import { Navigation } from './navigation.js';

// Initialize navigation
let navigation;

// Setup category selection
document.addEventListener('DOMContentLoaded', () => {
    // Initialize navigation
    navigation = new Navigation();
    
    const categoryCards = document.querySelectorAll('.category-card');
    categoryCards.forEach(card => {
        card.addEventListener('click', () => {
            const category = card.dataset.category;
            navigation.selectedCategory = category;
            
            // Update category title
            const categoryTitle = document.getElementById('category-title');
            if (categoryTitle) {
                const titles = {
                    'weightlifting': 'Weight Lifting Exercises',
                    'cardio': 'Cardio Exercises',
                    'yoga': 'Yoga Poses'
                };
                categoryTitle.textContent = titles[category] || 'Select Exercise';
            }
            
            navigation.showView('exercise', { category });
        });
    });

    // Setup back buttons
    const backToCategories = document.getElementById('back-to-categories');
    if (backToCategories) {
        backToCategories.addEventListener('click', () => {
            navigation.showView('landing');
        });
    }

    const backToExercises = document.getElementById('back-to-exercises');
    if (backToExercises) {
        backToExercises.addEventListener('click', () => {
            navigation.goBack();
        });
    }

    // Make navigation available globally
    window.navigation = navigation;
});

