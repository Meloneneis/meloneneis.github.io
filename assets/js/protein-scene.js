/**
 * Protein backbone ribbon — procedural alpha-helix for the hero.
 * Deliberately not a neon particle neural net.
 */
(function () {
  "use strict";

  function prefersReducedMotion() {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function initProteinScene() {
    var mount = document.getElementById("protein-canvas");
    if (!mount || typeof THREE === "undefined") return;

    var reduced = prefersReducedMotion();
    var width = mount.clientWidth || window.innerWidth;
    var height = mount.clientHeight || window.innerHeight;

    var renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.set(0, 0.35, 8.2);

    var group = new THREE.Group();
    scene.add(group);

    // Ambient + warm key + cool fill (lab lighting, not neon)
    scene.add(new THREE.AmbientLight(0xb8c4d4, 0.55));
    var key = new THREE.DirectionalLight(0xf3efe6, 0.95);
    key.position.set(4, 6, 5);
    scene.add(key);
    var fill = new THREE.DirectionalLight(0x1a8a94, 0.35);
    fill.position.set(-5, -2, -3);
    scene.add(fill);

    var helixPoints = [];
    var turns = 9;
    var pointsPerTurn = 28;
    var total = turns * pointsPerTurn;
    var radius = 1.15;
    var rise = 0.38;

    for (var i = 0; i < total; i++) {
      var t = i / pointsPerTurn;
      var angle = t * Math.PI * 2;
      var x = Math.cos(angle) * radius;
      var y = (i / total) * 7.2 - 3.6;
      var z = Math.sin(angle) * radius;
      // slight breathing warp
      var wobble = 0.08 * Math.sin(t * 3.1);
      helixPoints.push(new THREE.Vector3(x + wobble, y, z));
    }

    var curve = new THREE.CatmullRomCurve3(helixPoints, false, "catmullrom", 0.35);
    var tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 280, 0.11, 12, false),
      new THREE.MeshStandardMaterial({
        color: 0xd7c4a3,
        metalness: 0.18,
        roughness: 0.42,
        emissive: 0x0d5c63,
        emissiveIntensity: 0.08,
      })
    );
    group.add(tube);

    // Residue markers along the backbone
    var residueGeo = new THREE.SphereGeometry(0.085, 12, 12);
    var residueMat = new THREE.MeshStandardMaterial({
      color: 0xc45c26,
      metalness: 0.2,
      roughness: 0.35,
      emissive: 0xc45c26,
      emissiveIntensity: 0.15,
    });
    var residues = new THREE.Group();
    for (var r = 0; r < helixPoints.length; r += 4) {
      var sphere = new THREE.Mesh(residueGeo, residueMat);
      sphere.position.copy(helixPoints[r]);
      residues.add(sphere);
    }
    group.add(residues);

    // Secondary ribbon strand (pairing / co-evolution hint)
    var strandPoints = [];
    for (var s = 0; s < 60; s++) {
      var u = s / 59;
      var a = u * Math.PI * 4 + 1.2;
      strandPoints.push(
        new THREE.Vector3(
          Math.cos(a) * 2.35,
          u * 5.5 - 2.75,
          Math.sin(a) * 2.35
        )
      );
    }
    var strandCurve = new THREE.CatmullRomCurve3(strandPoints);
    var strand = new THREE.Mesh(
      new THREE.TubeGeometry(strandCurve, 120, 0.035, 8, false),
      new THREE.MeshStandardMaterial({
        color: 0x1a8a94,
        metalness: 0.1,
        roughness: 0.55,
        transparent: true,
        opacity: 0.7,
      })
    );
    group.add(strand);

    // Soft contact arcs between nearby residues
    var arcMat = new THREE.LineBasicMaterial({
      color: 0x1a8a94,
      transparent: true,
      opacity: 0.28,
    });
    for (var c = 0; c < helixPoints.length - 12; c += 10) {
      var p1 = helixPoints[c];
      var p2 = helixPoints[c + 11];
      var mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
      mid.x *= 1.35;
      mid.z *= 1.35;
      var arc = new THREE.QuadraticBezierCurve3(p1, mid, p2);
      var arcGeo = new THREE.BufferGeometry().setFromPoints(arc.getPoints(16));
      group.add(new THREE.Line(arcGeo, arcMat));
    }

    group.rotation.x = 0.35;
    group.rotation.z = -0.25;
    group.position.x = 1.6;

    var mouse = { x: 0, y: 0 };
    var target = { x: 0, y: 0 };

    function onPointerMove(e) {
      var cx = e.clientX || (e.touches && e.touches[0] && e.touches[0].clientX) || 0;
      var cy = e.clientY || (e.touches && e.touches[0] && e.touches[0].clientY) || 0;
      mouse.x = (cx / window.innerWidth) * 2 - 1;
      mouse.y = (cy / window.innerHeight) * 2 - 1;
    }

    if (!reduced) {
      window.addEventListener("pointermove", onPointerMove, { passive: true });
    }

    function onResize() {
      width = mount.clientWidth || window.innerWidth;
      height = mount.clientHeight || window.innerHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }
    window.addEventListener("resize", onResize);

    var clock = new THREE.Clock();
    var frame = 0;

    function animate() {
      frame = requestAnimationFrame(animate);
      var t = clock.getElapsedTime();

      if (!reduced) {
        target.x += (mouse.x * 0.35 - target.x) * 0.04;
        target.y += (mouse.y * 0.25 - target.y) * 0.04;
        group.rotation.y = t * 0.12 + target.x;
        group.rotation.x = 0.35 + target.y * 0.35;
        group.position.y = Math.sin(t * 0.4) * 0.12;
        residues.rotation.y = Math.sin(t * 0.2) * 0.02;
      }

      renderer.render(scene, camera);
    }

    if (reduced) {
      renderer.render(scene, camera);
    } else {
      animate();
    }

    // Pause when tab hidden
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
        cancelAnimationFrame(frame);
      } else if (!reduced) {
        animate();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      // Wait for deferred three.js
      setTimeout(initProteinScene, 50);
    });
  } else {
    setTimeout(initProteinScene, 50);
  }
})();
