import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface FlexTail3DViewerProps {
  flexTailColor?: number;
  backgroundColor?: number;
  groundPlaneColor?: number;
  showGroundPlane?: boolean;
  useAccelerometer?: boolean;
  autoRotate?: boolean;
  width?: number;
  height?: number;
  measurement?: {
    reconstruction?: {
      left: { x: number[]; y: number[]; z: number[] };
      right: { x: number[]; y: number[]; z: number[] };
      center: { x: number[]; y: number[]; z: number[] };
    };
    orientation?: {
      pitch: number;
      roll: number;
      yaw: number;
    };
  } | null;
}

export const FlexTail3DViewer = ({
  flexTailColor = 0xff6600,
  backgroundColor = 0x1a1a1a,
  groundPlaneColor = 0x404040,
  showGroundPlane = true,
  useAccelerometer = true,
  autoRotate = false,
  measurement = null,
}: FlexTail3DViewerProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const flextailMeshRef = useRef<THREE.Mesh | null>(null);
  const groundPlaneRef = useRef<THREE.Mesh | null>(null);
  const lastOrientationRef = useRef({ pitch: 0, roll: 0, yaw: 0 });
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(backgroundColor);
    scene.position.set(-30, -180, 0);
    sceneRef.current = scene;

    // Create camera
    const aspect = width / height;
    const camera = new THREE.PerspectiveCamera(45, aspect, 0.5, 2000);
    camera.position.set(450, 100, 0);
    camera.fov = 64;
    camera.lookAt(new THREE.Vector3(0, 0, 0));
    cameraRef.current = camera;

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = false;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.1);
    directionalLight.position.set(450, 100, 0);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const dirLight1 = new THREE.DirectionalLight(0xcccccc);
    dirLight1.position.set(1, 1, 1);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xcccccc);
    dirLight2.position.set(-1, -1, -1);
    scene.add(dirLight2);

    // Create controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 10;
    controls.maxDistance = 750;
    controls.enablePan = true;
    controls.autoRotate = autoRotate;
    controlsRef.current = controls;

    // Create FlexTail mesh
    const flextailMesh = createFlexTailMesh(flexTailColor);
    scene.add(flextailMesh);
    flextailMeshRef.current = flextailMesh;

    // Add ground plane
    if (showGroundPlane) {
      const groundPlane = createGroundPlane(groundPlaneColor);
      scene.add(groundPlane);
      groundPlaneRef.current = groundPlane;
    }

    // Set default scene rotation
    scene.rotation.set(0.5 * Math.PI, Math.PI, 0.5);

    // Handle window resize
    const handleResize = () => {
      if (!container || !camera || !renderer) return;
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };
    window.addEventListener('resize', handleResize);

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (flextailMeshRef.current) {
        flextailMeshRef.current.geometry.dispose();
        (flextailMeshRef.current.material as THREE.Material).dispose();
      }
      if (groundPlaneRef.current) {
        groundPlaneRef.current.geometry.dispose();
        (groundPlaneRef.current.material as THREE.Material).dispose();
      }
      if (controlsRef.current) {
        controlsRef.current.dispose();
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
        if (container && rendererRef.current.domElement.parentNode === container) {
          container.removeChild(rendererRef.current.domElement);
        }
      }
    };
  }, []);

  // Update measurement
  useEffect(() => {
    if (!measurement || !flextailMeshRef.current) return;

    // Update geometry if reconstruction data is provided
    if (measurement.reconstruction) {
      const newGeometry = coordsToThreeShape(measurement.reconstruction);
      const oldGeometry = flextailMeshRef.current.geometry;
      flextailMeshRef.current.geometry = newGeometry;
      oldGeometry.dispose();
    }

    // Update orientation if accelerometer data is provided
    if (useAccelerometer && measurement.orientation) {
      const o = measurement.orientation;
      const lastOrientation = lastOrientationRef.current;
      
      // Average with last orientation for smoothing
      const avgPitch = (o.pitch + lastOrientation.pitch) / 2;
      const avgRoll = (-o.roll - lastOrientation.roll) / 2;
      
      flextailMeshRef.current.rotation.set(avgPitch - Math.PI / 2, avgRoll, 0);
      
      lastOrientationRef.current = o;
    }

    // Render
    if (sceneRef.current && cameraRef.current && rendererRef.current) {
      rendererRef.current.render(sceneRef.current, cameraRef.current);
    }
  }, [measurement, useAccelerometer]);

  return <div ref={containerRef} className="w-full h-full" />;
};

function createFlexTailMesh(flexTailColor: number): THREE.Mesh {
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
  
  const geometry = coordsToThreeShape(reconstruction);
  const material = new THREE.MeshPhongMaterial({
    color: flexTailColor,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 1,
  });
  
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.rotation.set(-Math.PI / 2, 0, 0);
  
  return mesh;
}

function createGroundPlane(groundPlaneColor: number): THREE.Mesh {
  const planeSize = 100.0;
  const planeThickness = 10;
  
  const geometry = new THREE.BoxGeometry(planeSize, planeThickness, planeSize);
  const material = new THREE.MeshPhongMaterial({
    color: groundPlaneColor,
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

function coordsToThreeShape(reconstruction: {
  left: { x: number[]; y: number[]; z: number[] };
  right: { x: number[]; y: number[]; z: number[] };
  center: { x: number[]; y: number[]; z: number[] };
}): THREE.BufferGeometry {
  const r = reconstruction;
  const n = r.left.x.length - 1;
  const thickness = 5.0;

  // Create curves for left and right edges
  const leftPoints: THREE.Vector3[] = [];
  const rightPoints: THREE.Vector3[] = [];
  
  for (let i = 0; i <= n; i++) {
    leftPoints.push(new THREE.Vector3(r.left.x[i], r.left.y[i], r.left.z[i]));
    rightPoints.push(new THREE.Vector3(r.right.x[i], r.right.y[i], r.right.z[i]));
  }

  const leftCurve = new THREE.CatmullRomCurve3(leftPoints);
  const rightCurve = new THREE.CatmullRomCurve3(rightPoints);

  const geometry = new THREE.BufferGeometry();
  const uSegments = 30;
  const vSegments = 8;

  const vertices: number[] = [];
  const indices: number[] = [];
  const normals: number[] = [];

  // Calculate surface normals for each u segment
  const surfaceNormals: THREE.Vector3[] = [];
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

