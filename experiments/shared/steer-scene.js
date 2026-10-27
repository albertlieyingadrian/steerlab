// steer-scene.js
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

export function mountSteerScene(
  container,
  { strength = 0.85, autoplay = true } = {}
) {
  const noop = () => {};
  const fallback = { setStrength: noop, destroy: noop };
  const clamp = (value) => {
    const number = Number(value);
    return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 0;
  };
  const smoothstep = (a, b, value) => {
    const t = clamp((value - a) / (b - a));
    return t * t * (3 - 2 * t);
  };

  const canvas = document.createElement('canvas');
  const contextOptions = { alpha: false, antialias: true };
  let renderer;

  try {
    const context =
      canvas.getContext('webgl2', contextOptions) ||
      canvas.getContext('webgl', contextOptions);

    if (!context) return fallback;
    renderer = new THREE.WebGLRenderer({ canvas, context, ...contextOptions });
  } catch {
    renderer?.dispose();
    return fallback;
  }

  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const hiddenImages = [...container.querySelectorAll('img')].map((image) => ({
    image,
    display: image.style.getPropertyValue('display'),
    priority: image.style.getPropertyPriority('display')
  }));
  hiddenImages.forEach(({ image }) => {
    image.style.setProperty('display', 'none', 'important');
  });

  const root = document.createElement('div');
  root.style.cssText = 'width:100%;min-width:0;';

  const viewport = document.createElement('div');
  viewport.style.cssText =
    'position:relative;width:100%;aspect-ratio:16/9;overflow:hidden;border-radius:8px;';
  canvas.style.cssText = 'display:block;width:100%;height:100%;';
  canvas.setAttribute('role', 'img');
  canvas.setAttribute(
    'aria-label',
    'Activation steering: particles pass through five model layers. An orange direction at the third layer bends the downstream stream into the Golden Gate Bridge.'
  );
  viewport.append(canvas);

  const controls = document.createElement('div');
  controls.style.cssText = [
    'display:flex',
    'align-items:center',
    'flex-wrap:wrap',
    'gap:10px 14px',
    'padding:12px 2px',
    'color:var(--ink,#454b4c)',
    'font-family:var(--font-mono,monospace)',
    'font-size:12px',
    'line-height:1.5'
  ].join(';');

  const label = document.createElement('label');
  label.style.cssText =
    'display:flex;align-items:center;flex-wrap:wrap;gap:10px;flex:1;min-width:0;';
  const labelText = document.createElement('span');
  labelText.textContent = 'Steering strength';

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = '100';
  slider.step = '1';
  slider.style.cssText = [
    'flex:1',
    'min-width:90px',
    'max-width:280px',
    'height:24px',
    'margin:0',
    'cursor:pointer',
    'accent-color:var(--accent,#C8562A)'
  ].join(';');
  label.append(labelText, slider);

  const readout = document.createElement('output');
  readout.setAttribute('aria-hidden', 'true');
  readout.style.cssText =
    'min-width:4ch;text-align:right;color:var(--ink-soft,#697173);font-variant-numeric:tabular-nums;';

  const playButton = document.createElement('button');
  playButton.type = 'button';
  playButton.style.cssText = [
    'font:inherit',
    'color:inherit',
    'background:transparent',
    'border:1px solid var(--rule,#c9c6be)',
    'border-radius:5px',
    'padding:5px 11px',
    'min-height:32px',
    'cursor:pointer'
  ].join(';');

  controls.append(label, readout, playButton);
  root.append(viewport, controls);
  container.append(root);

  const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');

  let reducedMotion = motionPreference.matches;
  let playing = Boolean(autoplay);
  let requestedStrength = clamp(strength);
  let introActive = playing && !reducedMotion;
  let currentStrength = introActive ? 0 : requestedStrength;
  let introTime = 0;
  let elapsed = 0;
  let visible = false;
  let destroyed = false;
  let frameId = 0;
  let previousTime = null;
  let dark = false;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(33, 16 / 9, 0.1, 100);
  const lookAt = new THREE.Vector3(0.9, 1.1, 0);
  const orange = new THREE.Color('#C8562A');

  const geometries = new Set();
  const materials = new Set();
  const textures = new Set();
  const ownGeometry = (geometry) => {
    geometries.add(geometry);
    return geometry;
  };
  const ownMaterial = (material) => {
    materials.add(material);
    return material;
  };

  scene.add(new THREE.HemisphereLight(0xffffff, 0x85817a, 2));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
  keyLight.position.set(-5, 10, 12);
  scene.add(keyLight);

  const panelGeometry = ownGeometry(new THREE.PlaneGeometry(3.4, 5.4));
  const panelEdges = ownGeometry(new THREE.EdgesGeometry(panelGeometry));
  const panelMaterial = ownMaterial(
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 1,
      metalness: 0,
      transparent: true,
      opacity: 0.11,
      depthWrite: false,
      side: THREE.DoubleSide
    })
  );
  const edgeMaterial = ownMaterial(
    new THREE.LineBasicMaterial({
      color: 0x9ba5a5,
      transparent: true,
      opacity: 0.23,
      depthWrite: false
    })
  );

  for (const x of [-8, -4, 0, 4, 8]) {
    const panel = new THREE.Group();
    panel.position.set(x, 1.1, 0);
    panel.rotation.y = Math.PI / 2;
    panel.add(new THREE.Mesh(panelGeometry, panelMaterial));
    panel.add(new THREE.LineSegments(panelEdges, edgeMaterial));
    scene.add(panel);
  }

  const arrowMaterial = ownMaterial(
    new THREE.MeshBasicMaterial({ color: orange })
  );
  const shaft = new THREE.Mesh(
    ownGeometry(new THREE.CylinderGeometry(0.025, 0.025, 0.75, 12)),
    arrowMaterial
  );
  shaft.position.set(0, 1.96, 0);
  const arrowhead = new THREE.Mesh(
    ownGeometry(new THREE.ConeGeometry(0.14, 0.34, 20)),
    arrowMaterial
  );
  arrowhead.rotation.z = Math.PI;
  arrowhead.position.set(0, 1.43, 0);
  scene.add(shaft, arrowhead);

  const bridgeStart = 0.45;
  const bridgeEnd = 11.4;
  const deckY = 0.1;
  const towerXs = [3, 8.2];
  const towerTop = 4.4;
  const cableLow = 1.35;
  const bridgeHalfWidth = 0.68;
  const linePositions = [];

  const segment = (a, b) => linePositions.push(...a, ...b);
  const polyline = (points) => {
    for (let i = 1; i < points.length; i++) segment(points[i - 1], points[i]);
  };

  // Normalized cosh curves keep the cable endpoints exactly on the towers.
  function cableY(x) {
    const [left, right] = towerXs;
    if (x < left) {
      const u = (x - bridgeStart) / (left - bridgeStart);
      return deckY + (towerTop - deckY) *
        (Math.cosh(1.2 * u) - 1) / (Math.cosh(1.2) - 1);
    }
    if (x > right) {
      const u = (bridgeEnd - x) / (bridgeEnd - right);
      return deckY + (towerTop - deckY) *
        (Math.cosh(1.2 * u) - 1) / (Math.cosh(1.2) - 1);
    }
    const u = (x - (left + right) / 2) / ((right - left) / 2);
    return cableLow + (towerTop - cableLow) *
      (Math.cosh(1.65 * u) - 1) / (Math.cosh(1.65) - 1);
  }

  // Stepped caps give the narrow tower legs an art-deco silhouette.
  function towerOutline(x, z) {
    return [
      [x - 0.19, deckY, z],
      [x - 0.19, 3.95, z],
      [x - 0.12, 3.95, z],
      [x - 0.12, towerTop, z],
      [x + 0.12, towerTop, z],
      [x + 0.12, 3.95, z],
      [x + 0.19, 3.95, z],
      [x + 0.19, deckY, z]
    ];
  }

  const routes = [];
  const routeSamples = 1024;

  // Arc-length lookup tables allow continuous motion around every bridge member.
  function addRoute(points, weight) {
    const lengths = [0];
    for (let i = 1; i < points.length; i++) {
      lengths.push(lengths[i - 1] + Math.hypot(
        points[i][0] - points[i - 1][0],
        points[i][1] - points[i - 1][1],
        points[i][2] - points[i - 1][2]
      ));
    }
    const data = new Float32Array((routeSamples + 1) * 3);
    let edge = 1;
    for (let i = 0; i <= routeSamples; i++) {
      const distance = lengths.at(-1) * i / routeSamples;
      while (edge < points.length - 1 && lengths[edge] < distance) edge++;
      const fraction = (distance - lengths[edge - 1]) /
        Math.max(1e-8, lengths[edge] - lengths[edge - 1]);
      for (let axis = 0; axis < 3; axis++) {
        data[i * 3 + axis] = THREE.MathUtils.lerp(
          points[edge - 1][axis], points[edge][axis], fraction
        );
      }
    }
    routes.push({ data, weight });
  }

  for (const z of [-bridgeHalfWidth, bridgeHalfWidth]) {
    const deckRoute = [[bridgeStart, deckY, z], [bridgeEnd, deckY, z]];
    polyline(deckRoute);
    segment([bridgeStart, -0.12, z], [bridgeEnd, -0.12, z]);
    addRoute(deckRoute, 0.22);

    const cableRoute = [];
    for (let i = 0; i <= 240; i++) {
      const x = THREE.MathUtils.lerp(bridgeStart, bridgeEnd, i / 240);
      cableRoute.push([x, cableY(x), z]);
    }
    polyline(cableRoute);
    addRoute(cableRoute, 0.31);

    const towerRoute = [[bridgeStart, deckY, z]];
    for (const x of towerXs) {
      const outline = towerOutline(x, z);
      polyline(outline);
      towerRoute.push(...outline);
    }
    towerRoute.push([bridgeEnd, deckY, z]);
    addRoute(towerRoute, 0.27);

    const hangerRoutes = Array.from({ length: 3 }, () => [
      [bridgeStart, deckY, z]
    ]);
    let hanger = 0;
    for (let x = bridgeStart + 0.45; x < bridgeEnd - 0.2; x += 0.45) {
      const bottom = [x, deckY, z];
      const top = [x, cableY(x), z];
      segment(bottom, top);
      hangerRoutes[hanger++ % 3].push(bottom, top, bottom);
      segment([x - 0.45, -0.12, z], bottom);
    }
    for (const route of hangerRoutes) {
      route.push([bridgeEnd, deckY, z]);
      addRoute(route, 0.20 / 3);
    }
  }

  for (const x of towerXs) {
    for (const y of [1.45, 2.85, 4.05]) {
      segment([x, y, -bridgeHalfWidth], [x, y, bridgeHalfWidth]);
      segment([x, y + 0.12, -bridgeHalfWidth], [x, y + 0.12, bridgeHalfWidth]);
    }
  }
  for (let x = bridgeStart; x <= bridgeEnd; x += 0.7) {
    segment([x, deckY, -bridgeHalfWidth], [x, deckY, bridgeHalfWidth]);
  }

  const bridgeGeometry = ownGeometry(new THREE.BufferGeometry());
  bridgeGeometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(linePositions, 3)
  );
  const bridgeMaterial = ownMaterial(
    new THREE.LineBasicMaterial({
      color: orange,
      transparent: true,
      opacity: 0,
      depthWrite: false
    })
  );
  scene.add(new THREE.LineSegments(bridgeGeometry, bridgeMaterial));

  let randomState = 123456789;
  function random() {
    randomState ^= randomState << 13;
    randomState ^= randomState >>> 17;
    randomState ^= randomState << 5;
    return (randomState >>> 0) / 4294967296;
  }

  const count = 1500;
  const positions = new Float32Array(count * 3);
  const steering = new Float32Array(count);
  const opacity = new Float32Array(count);
  const sizes = new Float32Array(count);
  const phases = new Float32Array(count);
  const streamY = new Float32Array(count);
  const streamZ = new Float32Array(count);
  const particleRoutes = new Uint8Array(count);
  const totalWeight = routes.reduce((sum, route) => sum + route.weight, 0);

  for (let i = 0; i < count; i++) {
    phases[i] = (i + random()) / count;
    streamY[i] = deckY + (random() - 0.5) * 0.42;
    streamZ[i] = (random() - 0.5) * 0.85;
    sizes[i] = 2.5 + random() * 1.1;
    let choice = random() * totalWeight;
    let route = 0;
    while (route < routes.length - 1 && choice > routes[route].weight) {
      choice -= routes[route++].weight;
    }
    particleRoutes[i] = route;
  }

  const particleGeometry = ownGeometry(new THREE.BufferGeometry());
  const positionAttribute = new THREE.BufferAttribute(positions, 3);
  const steeringAttribute = new THREE.BufferAttribute(steering, 1);
  const opacityAttribute = new THREE.BufferAttribute(opacity, 1);
  for (const attribute of [positionAttribute, steeringAttribute, opacityAttribute]) {
    attribute.setUsage(THREE.DynamicDrawUsage);
  }
  particleGeometry.setAttribute('position', positionAttribute);
  particleGeometry.setAttribute('steering', steeringAttribute);
  particleGeometry.setAttribute('opacity', opacityAttribute);
  particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

  const particleMaterial = ownMaterial(
    new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        ink: { value: new THREE.Color('#4B5051') },
        accent: { value: orange },
        pixelRatio: { value: renderer.getPixelRatio() }
      },
      vertexShader: `
        attribute float steering;
        attribute float opacity;
        attribute float size;
        uniform float pixelRatio;
        varying float vSteering;
        varying float vOpacity;
        void main() {
          vSteering = steering;
          vOpacity = opacity;
          vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * viewPosition;
          gl_PointSize = size * pixelRatio *
            clamp(28.0 / -viewPosition.z, 0.65, 1.6);
        }
      `,
      fragmentShader: `
        uniform vec3 ink;
        uniform vec3 accent;
        varying float vSteering;
        varying float vOpacity;
        void main() {
          float radius = length(gl_PointCoord * 2.0 - 1.0);
          float softness = 1.0 - smoothstep(0.25, 1.0, radius);
          float alpha = softness * vOpacity * 0.87;
          if (alpha < 0.01) discard;
          gl_FragColor = vec4(mix(ink, accent, vSteering), alpha);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }
      `
    })
  );
  const particles = new THREE.Points(particleGeometry, particleMaterial);
  particles.frustumCulled = false;
  scene.add(particles);

  const puffCanvas = document.createElement('canvas');
  puffCanvas.width = puffCanvas.height = 128;
  const puffContext = puffCanvas.getContext('2d');
  const gradient = puffContext.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, 'rgba(255,255,255,0.8)');
  gradient.addColorStop(0.45, 'rgba(255,255,255,0.35)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  puffContext.fillStyle = gradient;
  puffContext.fillRect(0, 0, 128, 128);

  const puffTexture = new THREE.CanvasTexture(puffCanvas);
  puffTexture.colorSpace = THREE.SRGBColorSpace;
  textures.add(puffTexture);
  const fogMaterial = ownMaterial(
    new THREE.SpriteMaterial({
      map: puffTexture,
      color: 0xffffff,
      transparent: true,
      opacity: 0.15,
      depthWrite: false
    })
  );
  const puffs = [];
  for (let i = 0; i < 5; i++) {
    const puff = new THREE.Sprite(fogMaterial);
    puff.scale.set(4.7 + random(), 1.5 + random() * 0.5, 1);
    scene.add(puff);
    puffs.push(puff);
  }

  function updateReadout() {
    const value = Math.round(currentStrength * 100);
    slider.value = String(value);
    readout.value = `${value}%`;
  }

  function updateButton() {
    const active = playing && !reducedMotion;
    playButton.textContent = active ? 'Pause' : 'Play';
    playButton.setAttribute('aria-label', active ? 'Pause animation' : 'Play animation');
    playButton.disabled = reducedMotion;
    playButton.title = reducedMotion ? 'Animation disabled by reduced-motion preference' : '';
    playButton.style.cursor = reducedMotion ? 'default' : 'pointer';
    playButton.style.opacity = reducedMotion ? '0.55' : '1';
  }

  function schedule() {
    if (!destroyed && visible && !document.hidden && !frameId) {
      frameId = requestAnimationFrame(renderFrame);
    }
  }

  function stopFrame() {
    cancelAnimationFrame(frameId);
    frameId = 0;
    previousTime = null;
  }

  function updateScene(time) {
    const orbit = reducedMotion ? 0 : Math.sin(time * 0.065) * 0.045;
    const angle = Math.atan2(6, 27) + orbit;
    const radius = Math.hypot(6, 27) * 0.9;
    camera.position.set(Math.sin(angle) * radius, 8.2, Math.cos(angle) * radius);
    camera.lookAt(lookAt);

    for (let i = 0; i < count; i++) {
      const phase = (phases[i] + time * 0.014) % 1;
      const x = -12 + phase * 24;
      const u = clamp(x / 12);
      const blend = currentStrength * smoothstep(0, 0.09, u);
      const offset = i * 3;
      let tx = x;
      let ty = streamY[i];
      let tz = streamZ[i];

      if (x > 0) {
        const data = routes[particleRoutes[i]].data;
        const sample = u * routeSamples;
        const index = Math.min(routeSamples - 1, Math.floor(sample));
        const fraction = sample - index;
        const a = index * 3;
        tx = THREE.MathUtils.lerp(data[a], data[a + 3], fraction);
        ty = THREE.MathUtils.lerp(data[a + 1], data[a + 4], fraction) +
          (streamY[i] - deckY) * 0.1;
        tz = THREE.MathUtils.lerp(data[a + 2], data[a + 5], fraction) +
          streamZ[i] * 0.045;
      }

      positions[offset] = THREE.MathUtils.lerp(x, tx, blend);
      positions[offset + 1] = THREE.MathUtils.lerp(streamY[i], ty, blend);
      positions[offset + 2] = THREE.MathUtils.lerp(streamZ[i], tz, blend);
      steering[i] = blend;
      opacity[i] = smoothstep(-12, -11.3, x) *
        (1 - smoothstep(11.3, 12, x));
    }

    positionAttribute.needsUpdate = true;
    steeringAttribute.needsUpdate = true;
    opacityAttribute.needsUpdate = true;
    bridgeMaterial.opacity = currentStrength * (dark ? 0.52 : 0.43);
    fogMaterial.opacity = currentStrength * (dark ? 0.055 : 0.17);

    puffs.forEach((puff, i) => {
      puff.position.set(
        1.1 + i * 2.45 + Math.sin(time * 0.055 + i * 1.7) * 0.45,
        -0.75 - (i % 2) * 0.25 + Math.sin(time * 0.04 + i) * 0.07,
        0.6 + (i % 3) * 0.35
      );
    });
  }

  function renderFrame(now) {
    frameId = 0;
    if (destroyed || !visible || document.hidden) {
      previousTime = null;
      return;
    }

    const animated = playing && !reducedMotion;
    const delta = previousTime === null ? 0 : Math.min((now - previousTime) / 1000, 0.05);
    previousTime = animated ? now : null;

    if (animated) {
      elapsed += delta;
      if (introActive) {
        introTime += delta;
        currentStrength = requestedStrength * smoothstep(0.65, 7.5, introTime);
        updateReadout();
        if (introTime >= 7.5) introActive = false;
      }
    }

    updateScene(reducedMotion ? 0 : elapsed);
    renderer.render(scene, camera);
    if (animated) schedule();
  }

  function setStrength(value) {
    if (destroyed) return;
    requestedStrength = clamp(value);
    currentStrength = requestedStrength;
    introActive = false;
    updateReadout();
    schedule();
  }

  function onSliderInput() {
    setStrength(Number(slider.value) / 100);
  }

  function onSliderTouch() {
    introActive = false;
    requestedStrength = currentStrength;
  }

  function onPlayClick() {
    playing = !playing;
    previousTime = null;
    updateButton();
    schedule();
  }

  function applyTheme() {
    const override = document.documentElement.dataset.theme;
    dark = override === 'dark';
    scene.background = new THREE.Color(dark ? '#121516' : '#F3EEE4');
    particleMaterial.uniforms.ink.value.set(dark ? '#A7B1B3' : '#4B5051');
    panelMaterial.color.set(dark ? '#8B999B' : '#FFFFFF');
    panelMaterial.opacity = dark ? 0.065 : 0.11;
    edgeMaterial.color.set(dark ? '#A7B1B3' : '#929C9D');
    edgeMaterial.opacity = dark ? 0.16 : 0.23;

    controls.style.color = `var(--ink,${dark ? '#D7DDDD' : '#454B4C'})`;
    readout.style.color = `var(--ink-soft,${dark ? '#A7B1B3' : '#697173'})`;
    playButton.style.borderColor = `var(--rule,${dark ? '#394245' : '#C9C6BE'})`;
    viewport.style.background = dark ? '#121516' : '#F3EEE4';
    schedule();
  }

  function onMotionChange() {
    reducedMotion = motionPreference.matches;
    if (reducedMotion) {
      introActive = false;
      currentStrength = requestedStrength;
      updateReadout();
    }
    previousTime = null;
    updateButton();
    schedule();
  }

  function onVisibilityChange() {
    if (document.hidden) stopFrame();
    else schedule();
  }

  function resize() {
    if (destroyed) return;
    const width = Math.max(1, viewport.clientWidth);
    const height = Math.max(1, viewport.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);
    particleMaterial.uniforms.pixelRatio.value = renderer.getPixelRatio();
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    schedule();
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(viewport);

  const intersectionObserver = new IntersectionObserver((entries) => {
    visible = entries[entries.length - 1].isIntersecting;
    if (visible) schedule();
    else stopFrame();
  });
  intersectionObserver.observe(container);

  const themeObserver = new MutationObserver(applyTheme);
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme']
  });

  slider.addEventListener('input', onSliderInput);
  slider.addEventListener('pointerdown', onSliderTouch);
  slider.addEventListener('keydown', onSliderTouch);
  playButton.addEventListener('click', onPlayClick);
  motionPreference.addEventListener('change', onMotionChange);
  document.addEventListener('visibilitychange', onVisibilityChange);

  updateReadout();
  updateButton();
  applyTheme();
  resize();

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    stopFrame();

    resizeObserver.disconnect();
    intersectionObserver.disconnect();
    themeObserver.disconnect();
    slider.removeEventListener('input', onSliderInput);
    slider.removeEventListener('pointerdown', onSliderTouch);
    slider.removeEventListener('keydown', onSliderTouch);
    playButton.removeEventListener('click', onPlayClick);
    motionPreference.removeEventListener('change', onMotionChange);
    document.removeEventListener('visibilitychange', onVisibilityChange);

    scene.clear();
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
    textures.forEach((texture) => texture.dispose());
    renderer.dispose();
    renderer.forceContextLoss();
    root.remove();

    hiddenImages.forEach(({ image, display, priority }) => {
      if (display) image.style.setProperty('display', display, priority);
      else image.style.removeProperty('display');
    });
  }

  return { setStrength, destroy };
}
