import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/**
 * FlexTail3DViewer - A Three.js visualization for FlexTail measurement data
 * 
 * Ported from the Flutter implementation to provide plug-and-play 3D visualization
 */
export class FlexTail3DViewer {
  constructor(container, options = {}) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    if (!this.container) {
      throw new Error('Container element not found');
    }

    // Configuration options
    this.options = {
      flexTailColor: options.flexTailColor || 0xff6600,
      backgroundColor: options.backgroundColor || 0x1a1a1a,
      groundPlaneColor: options.groundPlaneColor || 0x404040,
      showGroundPlane: options.showGroundPlane !== false,
      useAccelerometer: options.useAccelerometer !== false,
      autoRotate: options.autoRotate || false,
      width: options.width || this.container.clientWidth,
      height: options.height || this.container.clientHeight,
    };

    // State
    this.lastOrientation = { pitch: 0, roll: 0, yaw: 0 };
    this.flextailMesh = null;
    
    this._init();
  }

  _init() {
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.options.backgroundColor);
    this.scene.position.set(-30, -180, 0);

    // Create camera
    const aspect = this.options.width / this.options.height;
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.5, 2000);
    this.camera.position.set(450, 100, 0);
    this.camera.fov = 64;
    this.camera.lookAt(new THREE.Vector3(0, 0, 0));

    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.options.width, this.options.height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = false;
    this.container.appendChild(this.renderer.domElement);

    // Add lights
    this._setupLights();

    // Create controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 10;
    this.controls.maxDistance = 750;
    this.controls.enablePan = true;
    this.controls.autoRotate = this.options.autoRotate;

    // Create FlexTail mesh with default shape
    this.flextailMesh = this._createFlexTailMesh();
    this.scene.add(this.flextailMesh);

    // Add ground plane
    if (this.options.showGroundPlane) {
      this.groundPlane = this._createGroundPlane();
      this.scene.add(this.groundPlane);
    }

    // Set default scene rotation
    this.scene.rotation.set(0.5 * Math.PI, Math.PI, 0.5);

    // Handle window resize
    this._onWindowResize = this._handleResize.bind(this);
    window.addEventListener('resize', this._onWindowResize);

    // Start animation loop
    this._animate();
  }

  _setupLights() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);

    // Directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.1);
    directionalLight.position.set(450, 100, 0);
    directionalLight.castShadow = true;
    this.scene.add(directionalLight);

    // Additional directional lights for better visibility
    const dirLight1 = new THREE.DirectionalLight(0xcccccc);
    dirLight1.position.set(1, 1, 1);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xcccccc);
    dirLight2.position.set(-1, -1, -1);
    this.scene.add(dirLight2);
  }

  _createFlexTailMesh() {
    const geometry = this._createDefaultGeometry();
    const material = new THREE.MeshPhongMaterial({
      color: this.options.flexTailColor,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 1,
    });
    
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.rotation.set(-Math.PI / 2, 0, 0);
    
    return mesh;
  }

  _createDefaultGeometry() {
    // Create default shape matching the Flutter implementation
    const reconstruction = {
      left: {
        x: [-10, -10],
        y: [0, 40],
        z: [0, 300]
      },
      center: {
        x: [0, 0],
        y: [0, 0],
        z: [0, 360]
      },
      right: {
        x: [10, 10],
        y: [0, 40],
        z: [0, 300]
      }
    };
    
    return this._coordsToThreeShape(reconstruction);
  }

  _createGroundPlane() {
    const planeSize = 100.0;
    const planeThickness = 10;
    
    const geometry = new THREE.BoxGeometry(planeSize, planeThickness, planeSize);
    const material = new THREE.MeshPhongMaterial({
      color: this.options.groundPlaneColor,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.7,
    });
    
    const plane = new THREE.Mesh(geometry, material);
    plane.receiveShadow = true;
    plane.position.set(0, -planeThickness / 2, 0);
    plane.rotateX(Math.PI / 2);
    
    return plane;
  }

  /**
   * Convert reconstruction coordinates to Three.js geometry
   * Ported from ThreeConvert.coordsToThreeShape in Dart
   */
  _coordsToThreeShape(reconstruction) {
    const r = reconstruction;
    const n = r.left.x.length - 1;
    const thickness = 5.0;

    // Create curves for left and right edges
    const leftPoints = [];
    const rightPoints = [];
    
    for (let i = 0; i <= n; i++) {
      // Invert Y to flip bend direction
      leftPoints.push(new THREE.Vector3(r.left.x[i], -r.left.y[i], r.left.z[i]));
      rightPoints.push(new THREE.Vector3(r.right.x[i], -r.right.y[i], r.right.z[i]));
    }

    const leftCurve = new THREE.CatmullRomCurve3(leftPoints);
    const rightCurve = new THREE.CatmullRomCurve3(rightPoints);

    const geometry = new THREE.BufferGeometry();
    const uSegments = 30;
    const vSegments = 8;

    const vertices = [];
    const indices = [];
    const normals = [];

    // Calculate surface normals for each u segment
    const surfaceNormals = [];
    for (let u = 0; u <= uSegments; u++) {
      const uParam = u / uSegments;
      
      const tangentU = u < uSegments 
        ? leftCurve.getTangent(uParam)
        : leftCurve.getTangent(uParam - 0.001);

      const leftPoint = leftCurve.getPoint(uParam);
      const rightPoint = rightCurve.getPoint(uParam);
      
      const tangentV = new THREE.Vector3(
        rightPoint.x - leftPoint.x,
        rightPoint.y - leftPoint.y,
        rightPoint.z - leftPoint.z
      ).normalize();

      const surfaceNormal = tangentU.clone().cross(tangentV).normalize();
      surfaceNormals.push(surfaceNormal);
    }

    // Generate vertices for top and bottom surfaces
    for (let layer = 0; layer < 2; layer++) {
      const thicknessSign = layer === 0 ? -1.0 : 1.0;

      for (let u = 0; u <= uSegments; u++) {
        const uParam = u / uSegments;
        const leftPoint = leftCurve.getPoint(uParam);
        const rightPoint = rightCurve.getPoint(uParam);
        const surfaceNormal = surfaceNormals[u];

        for (let v = 0; v <= vSegments; v++) {
          const vParam = v / vSegments;

          const centerX = leftPoint.x + (rightPoint.x - leftPoint.x) * vParam;
          const centerY = leftPoint.y + (rightPoint.y - leftPoint.y) * vParam;
          const centerZ = leftPoint.z + (rightPoint.z - leftPoint.z) * vParam;

          const thicknessOffset = thickness / 2 * thicknessSign;
          const x = centerX + surfaceNormal.x * thicknessOffset;
          const y = centerY + surfaceNormal.y * thicknessOffset;
          const z = centerZ + surfaceNormal.z * thicknessOffset;

          vertices.push(x, y, z);
          normals.push(
            surfaceNormal.x * thicknessSign,
            surfaceNormal.y * thicknessSign,
            surfaceNormal.z * thicknessSign
          );
        }
      }
    }

    // Generate indices for main body
    for (let layer = 0; layer < 2; layer++) {
      const vertexOffset = layer * (uSegments + 1) * (vSegments + 1);

      for (let u = 0; u < uSegments; u++) {
        for (let v = 0; v < vSegments; v++) {
          const a = vertexOffset + u * (vSegments + 1) + v;
          const b = vertexOffset + u * (vSegments + 1) + v + 1;
          const c = vertexOffset + (u + 1) * (vSegments + 1) + v + 1;
          const d = vertexOffset + (u + 1) * (vSegments + 1) + v;

          if (layer === 0) {
            indices.push(a, b, c, a, c, d);
          } else {
            indices.push(a, c, b, a, d, c);
          }
        }
      }
    }

    // Set geometry attributes
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
    geometry.setIndex(indices);
    geometry.computeBoundingSphere();

    return geometry;
  }

  /**
   * Update the visualization with new measurement data
   * @param {Object} measurement - Measurement object containing reconstruction and orientation
   */
  updateMeasurement(measurement) {
    if (!measurement) return;

    // Update geometry if reconstruction data is provided
    if (measurement.reconstruction) {
      const newGeometry = this._coordsToThreeShape(measurement.reconstruction);
      this.flextailMesh.geometry.dispose();
      this.flextailMesh.geometry = newGeometry;
    }

    // Update orientation if accelerometer data is provided
    if (this.options.useAccelerometer && measurement.orientation) {
      const o = measurement.orientation;
      
      // Average with last orientation for smoothing (matching Flutter implementation)
      const avgPitch = (o.pitch + this.lastOrientation.pitch) / 2;
      const avgRoll = (-o.roll - this.lastOrientation.roll) / 2;
      
      // Adjusted offset so 90 degrees pitch corresponds to laying flat and pointing right
      this.flextailMesh.rotation.set(avgPitch, avgRoll, 0);
      
      this.lastOrientation = o;
    }

    this.render();
  }

  /**
   * Render a single frame
   */
  render() {
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Animation loop
   */
  _animate() {
    if (this._disposed) return;

    requestAnimationFrame(() => this._animate());
    
    this.controls.update();
    this.render();
  }

  /**
   * Handle window resize
   */
  _handleResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  /**
   * Reset camera to default position
   */
  resetCamera() {
    this.camera.position.set(450, 100, 0);
    this.camera.lookAt(new THREE.Vector3(0, 0, 0));
    this.controls.reset();
    this.render();
  }

  /**
   * Toggle accelerometer-based orientation
   */
  toggleAccelerometer() {
    this.options.useAccelerometer = !this.options.useAccelerometer;
    if (!this.options.useAccelerometer) {
      this.flextailMesh.rotation.set(-Math.PI / 2, 0, 0);
    }
    this.render();
  }

  /**
   * Toggle ground plane visibility
   */
  toggleGroundPlane() {
    if (this.groundPlane) {
      this.groundPlane.visible = !this.groundPlane.visible;
      this.render();
    }
  }

  /**
   * Clean up resources
   */
  dispose() {
    this._disposed = true;
    
    window.removeEventListener('resize', this._onWindowResize);
    
    if (this.flextailMesh) {
      this.flextailMesh.geometry.dispose();
      this.flextailMesh.material.dispose();
    }
    
    if (this.groundPlane) {
      this.groundPlane.geometry.dispose();
      this.groundPlane.material.dispose();
    }
    
    this.controls.dispose();
    this.renderer.dispose();
    
    if (this.container && this.renderer.domElement.parentNode === this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}

export default FlexTail3DViewer;

