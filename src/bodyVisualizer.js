// 3D Body Visualization using Three.js

import * as THREE from 'three';

export class BodyVisualizer {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.bodyParts = {};
        this.animationId = null;
        this.init();
    }

    init() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        // Camera
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(0, 5, 10);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);

        // Grid helper
        const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
        this.scene.add(gridHelper);

        // Create skeleton
        this.createSkeleton();

        // Handle resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Start animation loop
        this.animate();
    }

    createSkeleton() {
        // Create a simplified human skeleton representation
        const material = new THREE.MeshStandardMaterial({ 
            color: 0x4a90e2,
            metalness: 0.3,
            roughness: 0.7
        });

        const jointMaterial = new THREE.MeshStandardMaterial({ 
            color: 0xff6b6b,
            metalness: 0.5,
            roughness: 0.5
        });

        // Head
        const headGeometry = new THREE.SphereGeometry(0.3, 16, 16);
        this.bodyParts.head = new THREE.Mesh(headGeometry, material);
        this.bodyParts.head.position.set(0, 3.5, 0);
        this.scene.add(this.bodyParts.head);

        // Torso (spine)
        const torsoGeometry = new THREE.CylinderGeometry(0.2, 0.25, 2, 8);
        this.bodyParts.torso = new THREE.Mesh(torsoGeometry, material);
        this.bodyParts.torso.position.set(0, 2, 0);
        this.scene.add(this.bodyParts.torso);

        // Pelvis
        const pelvisGeometry = new THREE.BoxGeometry(0.4, 0.3, 0.3);
        this.bodyParts.pelvis = new THREE.Mesh(pelvisGeometry, material);
        this.bodyParts.pelvis.position.set(0, 1, 0);
        this.scene.add(this.bodyParts.pelvis);

        // Left arm
        this.bodyParts.leftUpperArm = this.createLimb(0.15, 0.8, material);
        this.bodyParts.leftUpperArm.position.set(-0.5, 2.5, 0);
        this.scene.add(this.bodyParts.leftUpperArm);

        this.bodyParts.leftForearm = this.createLimb(0.12, 0.7, material);
        this.bodyParts.leftForearm.position.set(-0.5, 1.7, 0);
        this.scene.add(this.bodyParts.leftForearm);

        // Right arm
        this.bodyParts.rightUpperArm = this.createLimb(0.15, 0.8, material);
        this.bodyParts.rightUpperArm.position.set(0.5, 2.5, 0);
        this.scene.add(this.bodyParts.rightUpperArm);

        this.bodyParts.rightForearm = this.createLimb(0.12, 0.7, material);
        this.bodyParts.rightForearm.position.set(0.5, 1.7, 0);
        this.scene.add(this.bodyParts.rightForearm);

        // Left leg
        this.bodyParts.leftThigh = this.createLimb(0.18, 1.2, material);
        this.bodyParts.leftThigh.position.set(-0.2, 0.4, 0);
        this.scene.add(this.bodyParts.leftThigh);

        this.bodyParts.leftShin = this.createLimb(0.15, 1.0, material);
        this.bodyParts.leftShin.position.set(-0.2, -0.8, 0);
        this.scene.add(this.bodyParts.leftShin);

        // Right leg
        this.bodyParts.rightThigh = this.createLimb(0.18, 1.2, material);
        this.bodyParts.rightThigh.position.set(0.2, 0.4, 0);
        this.scene.add(this.bodyParts.rightThigh);

        this.bodyParts.rightShin = this.createLimb(0.15, 1.0, material);
        this.bodyParts.rightShin.position.set(0.2, -0.8, 0);
        this.scene.add(this.bodyParts.rightShin);

        // Joints
        this.createJoints(jointMaterial);
    }

    createLimb(radius, length, material) {
        const geometry = new THREE.CylinderGeometry(radius, radius, length, 8);
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        return mesh;
    }

    createJoints(material) {
        const jointGeometry = new THREE.SphereGeometry(0.1, 8, 8);
        const joints = ['shoulderL', 'shoulderR', 'elbowL', 'elbowR', 'hipL', 'hipR', 'kneeL', 'kneeR'];
        
        joints.forEach(jointName => {
            const joint = new THREE.Mesh(jointGeometry, material);
            this.bodyParts[jointName] = joint;
            this.scene.add(joint);
        });

        // Position joints
        this.bodyParts.shoulderL.position.set(-0.5, 2.9, 0);
        this.bodyParts.shoulderR.position.set(0.5, 2.9, 0);
        this.bodyParts.elbowL.position.set(-0.5, 1.3, 0);
        this.bodyParts.elbowR.position.set(0.5, 1.3, 0);
        this.bodyParts.hipL.position.set(-0.2, 1, 0);
        this.bodyParts.hipR.position.set(0.2, 1, 0);
        this.bodyParts.kneeL.position.set(-0.2, -0.2, 0);
        this.bodyParts.kneeR.position.set(0.2, -0.2, 0);
    }

    // Update body pose based on sensor data
    updatePose(sensorData) {
        const { lumbarLordosis, sagittalTilt, lateralTilt, rotation } = sensorData;

        // Convert sensor data to body rotations
        // Lumbar lordosis affects lower back curvature
        const lordosisRad = THREE.MathUtils.degToRad(lumbarLordosis - 30);
        
        // Sagittal tilt (forward/backward lean)
        const sagittalRad = THREE.MathUtils.degToRad(sagittalTilt);
        
        // Lateral tilt (side lean)
        const lateralRad = THREE.MathUtils.degToRad(lateralTilt);
        
        // Rotation (twist)
        const rotationRad = THREE.MathUtils.degToRad(rotation);

        // Update torso based on sagittal and lateral tilt
        this.bodyParts.torso.rotation.x = sagittalRad;
        this.bodyParts.torso.rotation.z = lateralRad;
        this.bodyParts.torso.rotation.y = rotationRad;

        // Update pelvis
        this.bodyParts.pelvis.rotation.x = sagittalRad * 0.5;
        this.bodyParts.pelvis.rotation.z = lateralRad * 0.5;

        // Update head to maintain relative position
        this.bodyParts.head.rotation.x = sagittalRad * 0.3;
        this.bodyParts.head.rotation.z = lateralRad * 0.3;

        // Update legs based on lordosis (squat depth)
        const squatDepth = (lumbarLordosis - 20) / 40; // Normalize
        const legBend = Math.max(0, Math.min(1, squatDepth)) * Math.PI * 0.4;

        this.bodyParts.leftThigh.rotation.x = -legBend;
        this.bodyParts.rightThigh.rotation.x = -legBend;
        this.bodyParts.leftShin.rotation.x = legBend * 0.8;
        this.bodyParts.rightShin.rotation.x = legBend * 0.8;

        // Update knee positions
        const kneeOffset = Math.sin(legBend) * 0.6;
        this.bodyParts.kneeL.position.y = -0.2 - kneeOffset;
        this.bodyParts.kneeR.position.y = -0.2 - kneeOffset;
    }

    // Visual feedback for quality (color coding)
    updateQualityFeedback(metrics) {
        const color = this.getQualityColor(metrics.overall);
        const material = new THREE.MeshStandardMaterial({ 
            color: color,
            metalness: 0.3,
            roughness: 0.7
        });

        // Update torso color based on overall quality
        this.bodyParts.torso.material = material;
        this.bodyParts.pelvis.material = material;
    }

    getQualityColor(score) {
        if (score >= 80) return 0x4caf50; // Green
        if (score >= 60) return 0xff9800; // Orange
        return 0xf44336; // Red
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        window.removeEventListener('resize', this.onWindowResize);
        if (this.container && this.renderer.domElement) {
            this.container.removeChild(this.renderer.domElement);
        }
    }
}

