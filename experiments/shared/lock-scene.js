import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

export function mountLockScene(container, { strength = 0.8, autoplay = true } = {}) {
  const noop = { setStrength() {}, destroy() {} };
  const clampStrength = value => {
    const number = Number(value);
    return Number.isFinite(number) ? THREE.MathUtils.clamp(number, 0, 1) : 0;
  };

  const canvas = document.createElement('canvas');
  const contextOptions = {
    alpha: false,
    antialias: true,
    powerPreference: 'low-power'
  };

  let context;
  let renderer;

  try {
    context =
      canvas.getContext('webgl2', contextOptions) ||
      canvas.getContext('webgl', contextOptions);
    if (!context) return noop;

    renderer = new THREE.WebGLRenderer({
      canvas,
      context,
      ...contextOptions
    });
  } catch {
    context?.getExtension('WEBGL_lose_context')?.loseContext();
    return noop;
  }

  const colors = {
    ground: '#15110E',
    ink: '#D9D2CA',
    orange: '#C9561C',
    blue: '#4A86E8'
  };

  renderer.setClearColor(colors.ground, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  canvas.setAttribute('role', 'img');
  canvas.setAttribute(
    'aria-label',
    'A locked system prompt asks for English-only replies. A stream of pale, orderly particles passes through five model layers. An orange direction injected at the blue-edged third layer transforms the downstream stream into orange, abstract calligraphic glyphs, illustrating a switch to Chinese despite the system prompt.'
  );
  canvas.style.cssText = 'display:block;width:100%;height:100%;position:absolute;inset:0;';

  const root = document.createElement('div');
  root.style.cssText = 'width:100%;min-width:0;';

  const viewport = document.createElement('div');
  viewport.style.cssText = `
    position:relative;width:100%;aspect-ratio:16/9;
    overflow:hidden;background:${colors.ground};border-radius:6px;
  `;
  viewport.append(canvas);

  const controls = document.createElement('div');
  controls.style.cssText = `
    display:flex;align-items:center;gap:12px;flex-wrap:wrap;
    padding:12px 0 2px;color:var(--ink,${colors.ink});
    font-family:var(--font-mono,monospace);font-size:12px;line-height:1.5;
  `;

  const label = document.createElement('label');
  label.style.cssText = `
    display:flex;align-items:center;gap:12px;flex:1 1 230px;
    flex-wrap:wrap;min-width:0;
  `;

  const labelText = document.createElement('span');
  labelText.textContent = 'Steering strength';

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = '100';
  slider.step = '1';
  slider.style.cssText = `
    flex:1 1 110px;min-width:90px;margin:0;min-height:28px;
    accent-color:var(--control,${colors.orange});cursor:pointer;
  `;
  label.append(labelText, slider);

  const readout = document.createElement('output');
  readout.setAttribute('aria-label', 'Current steering strength');
  readout.setAttribute('aria-live', 'off');
  readout.style.cssText = `
    min-width:4ch;text-align:right;font-variant-numeric:tabular-nums;
    color:var(--ink-soft,${colors.ink});
  `;

  const playButton = document.createElement('button');
  playButton.type = 'button';
  playButton.style.cssText = `
    font:inherit;color:var(--ink,${colors.ink});background:transparent;
    border:1px solid var(--rule,#65584e);border-radius:4px;
    padding:5px 10px;min-height:32px;cursor:pointer;
  `;

  controls.append(label, readout, playButton);
  root.append(viewport, controls);

  const hiddenImages = Array.from(container.querySelectorAll('img'), image => ({
    image,
    hidden: image.getAttribute('hidden'),
    display: image.style.getPropertyValue('display'),
    priority: image.style.getPropertyPriority('display')
  }));

  for (const { image } of hiddenImages) {
    image.hidden = true;
    image.style.setProperty('display', 'none', 'important');
  }
  container.append(root);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(colors.ground);

  const camera = new THREE.PerspectiveCamera(32, 16 / 9, 0.1, 80);
  const geometries = new Set();
  const materials = new Set();

  const keepGeometry = geometry => {
    geometries.add(geometry);
    return geometry;
  };
  const keepMaterial = material => {
    materials.add(material);
    return material;
  };

  scene.add(new THREE.HemisphereLight(0xfff5e7, 0x554337, 2));
  const keyLight = new THREE.DirectionalLight(0xfff4e9, 2.3);
  keyLight.position.set(-3, 7, 9);
  scene.add(keyLight);

  function roundedShape(width, height, radius) {
    const x = -width / 2;
    const y = -height / 2;
    const shape = new THREE.Shape();
    shape.moveTo(x + radius, y);
    shape.lineTo(x + width - radius, y);
    shape.quadraticCurveTo(x + width, y, x + width, y + radius);
    shape.lineTo(x + width, y + height - radius);
    shape.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    shape.lineTo(x + radius, y + height);
    shape.quadraticCurveTo(x, y + height, x, y + height - radius);
    shape.lineTo(x, y + radius);
    shape.quadraticCurveTo(x, y, x + radius, y);
    return shape;
  }

  function addPanel(width, height, radius, opacity) {
    const group = new THREE.Group();
    const shape = roundedShape(width, height, radius);
    const face = new THREE.Mesh(
      keepGeometry(new THREE.ShapeGeometry(shape, 10)),
      keepMaterial(new THREE.MeshPhysicalMaterial({
        color: colors.ink,
        roughness: 0.85,
        metalness: 0,
        transparent: true,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: false
      }))
    );
    group.add(face);

    const outline = new THREE.Line(
      keepGeometry(new THREE.BufferGeometry().setFromPoints(shape.getPoints(12))),
      keepMaterial(new THREE.LineBasicMaterial({
        color: colors.ink,
        transparent: true,
        opacity: 0.28,
        depthWrite: false
      }))
    );
    outline.position.z = 0.006;
    group.add(outline);
    return group;
  }

  const prompt = addPanel(1.55, 1.9, 0.16, 0.12);
  prompt.position.set(-6.4, 0.24, 0.22);
  scene.add(prompt);

  const lockMaterial = keepMaterial(new THREE.MeshStandardMaterial({
    color: colors.ink,
    roughness: 0.65,
    metalness: 0.12
  }));

  const lockBody = new THREE.Mesh(
    keepGeometry(new THREE.BoxGeometry(0.52, 0.39, 0.1)),
    lockMaterial
  );
  lockBody.position.set(0, -0.08, 0.12);
  prompt.add(lockBody);

  // The upper half of a torus forms the padlock's shackle.
  const shackle = new THREE.Mesh(
    keepGeometry(new THREE.TorusGeometry(0.2, 0.034, 8, 32, Math.PI)),
    lockMaterial
  );
  shackle.position.set(0, 0.12, 0.12);
  prompt.add(shackle);

  const layerPositions = [-4.8, -3.1, -1.4, 0.3, 2.0];
  const steeringX = layerPositions[2];
  const layerDepth = 2.25;
  const layerHeight = 3.3;

  layerPositions.forEach((x, index) => {
    const panel = addPanel(layerDepth, layerHeight, 0.12, 0.065);
    panel.rotation.y = Math.PI / 2;
    panel.position.x = x;

    if (index === 2) {
      const edge = new THREE.Line(
        keepGeometry(new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-layerDepth / 2, -layerHeight / 2 + 0.13, 0.015),
          new THREE.Vector3(-layerDepth / 2, layerHeight / 2 - 0.13, 0.015)
        ])),
        keepMaterial(new THREE.LineBasicMaterial({
          color: colors.blue,
          transparent: true,
          opacity: 0.95,
          depthWrite: false
        }))
      );
      panel.add(edge);
    }
    scene.add(panel);
  });

  const arrowMaterial = keepMaterial(new THREE.MeshBasicMaterial({
    color: colors.orange
  }));
  const arrow = new THREE.Group();
  arrow.position.set(steeringX, 0, 0);
  const shaft = new THREE.Mesh(
    keepGeometry(new THREE.CylinderGeometry(0.024, 0.024, 0.46, 10)),
    arrowMaterial
  );
  shaft.position.y = 2.17;

  const arrowhead = new THREE.Mesh(
    keepGeometry(new THREE.ConeGeometry(0.105, 0.23, 16)),
    arrowMaterial
  );
  arrowhead.rotation.z = Math.PI;
  arrowhead.position.y = 1.84;
  arrow.add(shaft, arrowhead);
  scene.add(arrow);

  const baseline = new THREE.Line(
    keepGeometry(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-7.2, -1.88, 0),
      new THREE.Vector3(6.85, -1.88, 0)
    ])),
    keepMaterial(new THREE.LineBasicMaterial({
      color: colors.ink,
      transparent: true,
      opacity: 0.1,
      depthWrite: false
    }))
  );
  scene.add(baseline);

  let randomState = 71293;
  function random() {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    return randomState / 4294967296;
  }

  // Five deliberately non-linguistic quadratic brush strokes per cluster.
  const strokeTemplates = [
    [[-0.43, 0.73], [-0.12, 0.99], [0.32, 0.62]],
    [[-0.3, 0.43], [-0.51, -0.05], [-0.35, -0.65]],
    [[-0.08, 0.36], [0.4, 0.25], [0.23, -0.17]],
    [[0.19, 0.65], [-0.24, -0.28], [0.07, -0.78]],
    [[-0.18, -0.4], [0.12, -0.74], [0.46, -0.48]]
  ];
  const strokeA = [];
  const strokeB = [];
  const strokeC = [];

  for (let glyph = 0; glyph < 6; glyph++) {
    const angle = (random() - 0.5) * 0.3;
    const scaleX = 0.9 + random() * 0.15;
    const scaleY = 0.9 + random() * 0.23;
    const shear = (random() - 0.5) * 0.22;

    for (const stroke of strokeTemplates) {
      const transformed = stroke.map(([x, y]) => {
        const px = x * scaleX + shear * y + (random() - 0.5) * 0.065;
        const py = y * scaleY + (random() - 0.5) * 0.075;
        return new THREE.Vector2(
          px * Math.cos(angle) - py * Math.sin(angle),
          px * Math.sin(angle) + py * Math.cos(angle)
        );
      });
      strokeA.push(transformed[0]);
      strokeB.push(transformed[1]);
      strokeC.push(transformed[2]);
    }
  }

  const particleCount = 1600;
  const phases = new Float32Array(particleCount);
  const rows = new Float32Array(particleCount);
  const strokes = new Float32Array(particleCount);
  const seeds = new Float32Array(particleCount);

  for (let i = 0; i < particleCount; i++) {
    phases[i] = (i + random() * 0.6) / particleCount;
    rows[i] = (i % 10) / 9;
    strokes[i] = Math.floor(random() * 5);
    seeds[i] = random();
  }

  const particleGeometry = keepGeometry(new THREE.BufferGeometry());
  particleGeometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(particleCount * 3), 3)
  );
  particleGeometry.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));
  particleGeometry.setAttribute('aRow', new THREE.BufferAttribute(rows, 1));
  particleGeometry.setAttribute('aStroke', new THREE.BufferAttribute(strokes, 1));
  particleGeometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));

  const uniforms = {
    uTime: { value: 0 },
    uStrength: { value: 0 },
    uPointScale: { value: 1 },
    uPixelRatio: { value: 1 },
    uGrey: { value: new THREE.Color(colors.ink) },
    uOrange: { value: new THREE.Color(colors.orange) },
    uStrokeA: { value: strokeA },
    uStrokeB: { value: strokeB },
    uStrokeC: { value: strokeC }
  };

  const particleMaterial = keepMaterial(new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    vertexShader: `
      attribute float aPhase;
      attribute float aRow;
      attribute float aStroke;
      attribute float aSeed;

      uniform float uTime;
      uniform float uStrength;
      uniform float uPointScale;
      uniform float uPixelRatio;
      uniform vec3 uGrey;
      uniform vec3 uOrange;
      uniform vec2 uStrokeA[30];
      uniform vec2 uStrokeB[30];
      uniform vec2 uStrokeC[30];

      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        const float startX = -7.2;
        const float endX = 6.7;
        const float steeringX = -1.4;
        const float pitch = 1.35;

        float travel = fract(aPhase + uTime * 0.026);
        float x = mix(startX, endX, travel);
        vec3 rowPosition = vec3(
          x,
          mix(-1.12, 1.12, aRow),
          (aSeed - 0.5) * 0.22
        );

        float downstream = smoothstep(steeringX, steeringX + 0.52, x);
        float blend = uStrength * downstream;

        float glyphProgress = clamp(
          (x - steeringX) / pitch, 0.0, 5.99999
        );
        float glyphIndex = floor(glyphProgress);
        float t = fract(glyphProgress);
        int strokeIndex = int(glyphIndex * 5.0 + aStroke);

        vec2 a = uStrokeA[strokeIndex];
        vec2 b = uStrokeB[strokeIndex];
        vec2 c = uStrokeC[strokeIndex];
        vec2 curve = (1.0 - t) * (1.0 - t) * a
                   + 2.0 * (1.0 - t) * t * b
                   + t * t * c;

        vec3 glyphPosition = vec3(
          steeringX + (glyphIndex + 0.5) * pitch + curve.x,
          curve.y,
          (aSeed - 0.5) * 0.07
        );

        vec3 transformed = mix(rowPosition, glyphPosition, blend);
        vec4 viewPosition = modelViewMatrix * vec4(transformed, 1.0);
        gl_Position = projectionMatrix * viewPosition;

        float brushWidth = 0.65 + 0.5 * sin(3.14159265 * t);
        float size = mix(0.055, 0.068 * brushWidth, blend);
        size *= 0.85 + aSeed * 0.3;
        gl_PointSize = clamp(
          uPointScale * size / max(0.1, -viewPosition.z),
          1.0, 8.0 * uPixelRatio
        );

        // Fade stroke endpoints so particles can re-enter the next cluster.
        float strokeFade = smoothstep(0.0, 0.075, t)
                         * (1.0 - smoothstep(0.925, 1.0, t));
        float edgeFade = smoothstep(0.0, 0.025, travel)
                       * (1.0 - smoothstep(0.975, 1.0, travel));
        float textRhythm = 0.76 + 0.24 * smoothstep(
          0.12, 0.3, fract(x * 2.2 + aRow * 1.7)
        );
        float glyphFade = mix(
          1.0, strokeFade,
          downstream * smoothstep(0.0, 0.15, uStrength)
        );

        vAlpha = edgeFade * glyphFade
               * mix(textRhythm * 0.76, 0.96, blend);
        vColor = mix(uGrey, uOrange, blend);
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        float radius = length(gl_PointCoord - vec2(0.5));
        float coverage = 1.0 - smoothstep(0.18, 0.5, radius);
        float alpha = coverage * vAlpha;
        if (alpha < 0.01) discard;
        gl_FragColor = vec4(vColor, alpha);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `
  }));

  const particles = new THREE.Points(particleGeometry, particleMaterial);
  particles.frustumCulled = false;
  particles.renderOrder = 2;
  scene.add(particles);

  const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  let reducedMotion = motionQuery.matches;
  let requestedStrength = clampStrength(strength);
  let desiredStrength = reducedMotion || !autoplay ? requestedStrength : 0;
  let displayedStrength = desiredStrength;
  let automaticRamp = Boolean(autoplay) && !reducedMotion;
  let rampElapsed = 0;
  let playing = Boolean(autoplay);
  let elapsed = 0;
  let destroyed = false;
  let intersecting = true;
  let hasSize = false;
  let dirty = true;
  let frameId = null;
  let lastTimestamp = null;

  uniforms.uStrength.value = displayedStrength;

  function updateReadout() {
    const percent = Math.round(desiredStrength * 100);
    slider.value = String(percent);
    slider.setAttribute('aria-valuetext', `${percent} percent`);
    readout.value = `${percent}%`;
  }

  function updateButton() {
    playButton.disabled = reducedMotion;
    playButton.textContent = reducedMotion ? 'Play' : playing ? 'Pause' : 'Play';
    playButton.setAttribute(
      'aria-label',
      reducedMotion
        ? 'Animation disabled by reduced-motion preference'
        : playing ? 'Pause animation' : 'Play animation'
    );
    playButton.title = reducedMotion
      ? 'Animation is disabled by your reduced-motion preference.'
      : '';
    playButton.style.cursor = reducedMotion ? 'default' : 'pointer';
    playButton.style.opacity = reducedMotion ? '0.55' : '1';
  }

  function updateCamera() {
    const azimuth = 0.21 + (reducedMotion ? 0 : Math.sin(elapsed * 0.075) * 0.035);
    const radius = 11.5;
    camera.position.set(
      Math.sin(azimuth) * radius - 0.2,
      3.0,
      Math.cos(azimuth) * radius
    );
    camera.lookAt(0.6, 0.1, 0);
  }

  function isVisible() {
    return !destroyed && intersecting && !document.hidden && hasSize;
  }

  function isAnimating() {
    return isVisible() && playing && !reducedMotion;
  }

  function schedule() {
    if (!isVisible() || frameId !== null) return;
    if (dirty || isAnimating()) frameId = requestAnimationFrame(frame);
  }

  function frame(timestamp) {
    frameId = null;
    if (!isVisible()) {
      lastTimestamp = null;
      return;
    }

    const animating = isAnimating();
    const delta = lastTimestamp === null
      ? 0
      : Math.min((timestamp - lastTimestamp) / 1000, 0.05);
    lastTimestamp = animating ? timestamp : null;

    if (animating) {
      elapsed += delta;
      uniforms.uTime.value = elapsed;

      if (automaticRamp) {
        rampElapsed += delta;
        const progress = THREE.MathUtils.clamp((rampElapsed - 0.5) / 7.5, 0, 1);
        const eased = progress * progress * (3 - 2 * progress);
        desiredStrength = requestedStrength * eased;
        if (progress === 1) automaticRamp = false;
        updateReadout();
      }

      displayedStrength +=
        (desiredStrength - displayedStrength) * (1 - Math.exp(-delta * 9));

      if (Math.abs(desiredStrength - displayedStrength) < 0.0001) {
        displayedStrength = desiredStrength;
      }
      uniforms.uStrength.value = displayedStrength;
      updateCamera();
    }

    if (dirty || animating) renderer.render(scene, camera);
    dirty = false;
    schedule();
  }

  function refreshVisibility() {
    lastTimestamp = null;
    if (!isVisible() && frameId !== null) {
      cancelAnimationFrame(frameId);
      frameId = null;
    }
    dirty = true;
    schedule();
  }

  function setStrength(value) {
    if (destroyed) return;
    automaticRamp = false;
    requestedStrength = clampStrength(value);
    desiredStrength = requestedStrength;

    if (!isAnimating()) {
      displayedStrength = desiredStrength;
      uniforms.uStrength.value = displayedStrength;
    }

    updateReadout();
    dirty = true;
    schedule();
  }

  function onSliderInput() {
    setStrength(Number(slider.value) / 100);
  }

  function stopAutomaticRamp() {
    automaticRamp = false;
  }

  function onTogglePlayback() {
    if (destroyed || reducedMotion) return;
    playing = !playing;
    lastTimestamp = null;

    if (!playing) {
      displayedStrength = desiredStrength;
      uniforms.uStrength.value = displayedStrength;
    }

    updateButton();
    dirty = true;
    schedule();
  }

  function onMotionPreferenceChange() {
    reducedMotion = motionQuery.matches;
    lastTimestamp = null;

    if (reducedMotion) {
      automaticRamp = false;
      desiredStrength = requestedStrength;
      displayedStrength = requestedStrength;
      uniforms.uStrength.value = requestedStrength;
      uniforms.uTime.value = 0;
    } else {
      uniforms.uTime.value = elapsed;
    }

    updateCamera();
    updateReadout();
    updateButton();
    dirty = true;
    schedule();
  }

  function resize() {
    if (destroyed) return;
    const width = viewport.getBoundingClientRect().width;
    hasSize = width > 0;

    if (!hasSize) {
      refreshVisibility();
      return;
    }

    const height = width * 9 / 16;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    uniforms.uPixelRatio.value = pixelRatio;
    uniforms.uPointScale.value =
      renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));

    dirty = true;
    schedule();
  }

  slider.addEventListener('input', onSliderInput);
  slider.addEventListener('pointerdown', stopAutomaticRamp);
  slider.addEventListener('keydown', stopAutomaticRamp);
  playButton.addEventListener('click', onTogglePlayback);
  document.addEventListener('visibilitychange', refreshVisibility);
  motionQuery.addEventListener('change', onMotionPreferenceChange);

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(viewport);

  const intersectionObserver = new IntersectionObserver(entries => {
    if (destroyed) return;
    intersecting = entries[entries.length - 1].isIntersecting;
    refreshVisibility();
  }, { threshold: 0 });
  intersectionObserver.observe(viewport);

  updateReadout();
  updateButton();
  updateCamera();
  resize();

  return {
    setStrength,

    destroy() {
      if (destroyed) return;
      destroyed = true;
      if (frameId !== null) cancelAnimationFrame(frameId);

      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      document.removeEventListener('visibilitychange', refreshVisibility);
      motionQuery.removeEventListener('change', onMotionPreferenceChange);
      slider.removeEventListener('input', onSliderInput);
      slider.removeEventListener('pointerdown', stopAutomaticRamp);
      slider.removeEventListener('keydown', stopAutomaticRamp);
      playButton.removeEventListener('click', onTogglePlayback);

      for (const geometry of geometries) geometry.dispose();
      for (const material of materials) material.dispose();
      scene.clear();
      renderer.dispose();
      renderer.forceContextLoss();
      root.remove();

      for (const { image, hidden, display, priority } of hiddenImages) {
        if (hidden === null) image.removeAttribute('hidden');
        else image.setAttribute('hidden', hidden);

        if (display) image.style.setProperty('display', display, priority);
        else image.style.removeProperty('display');
      }
    }
  };
}
