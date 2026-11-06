// A watercolour meadow at sunset. Every number you might want to change lives in PARAMS,
// so a web page can turn them into sliders (a playground where you drag the sun and the painting repaints).
const PARAMS = {
  sunX: 290,        // 0..400
  sunY: 120,        // 0..400
  wind: 0.4,        // -1..1, how far grass and flowers lean
  flowerCount: 9,   // 0..30
  seed: 7,
};

function setup() {
  createCanvas(400, 400, WEBGL);
  brush.scaleBrushes(2);
  noLoop();
}

function wash(hex, opacity, points, bleed) {
  brush.noStroke();
  brush.fill(hex, opacity);
  brush.fillBleed(bleed);
  brush.fillTexture(0.5, 0.4);
  brush.beginShape(0.4);
  for (const [x, y] of points) brush.vertex(x, y);
  brush.endShape(true);
}

function draw() {
  translate(-width / 2, -height / 2);
  background("#f4efe4");
  randomSeed(PARAMS.seed);

  // 1. sky, back to front
  wash("#f2c6a0", 150, [[0, 0], [400, 0], [400, 250], [0, 250]], 0.25);
  wash("#e9a3a0", 90, [[0, 150], [400, 120], [400, 250], [0, 250]], 0.3);

  // 2. the sun
  brush.noStroke();
  brush.fill("#f4a259", 200);
  brush.fillBleed(0.35);
  brush.circle(PARAMS.sunX, PARAMS.sunY, 38);

  // 3. far hills, then the meadow
  wash("#9fb48a", 170, [[0, 240], [120, 205], [260, 230], [400, 200], [400, 300], [0, 300]], 0.2);
  wash("#6f9a5a", 200, [[0, 270], [150, 255], [300, 275], [400, 260], [400, 400], [0, 400]], 0.15);

  // 4. grass blades lean with the wind
  for (let i = 0; i < 26; i++) {
    const x = random(0, 400);
    const y = random(300, 395);
    const h = random(18, 40);
    wash("#4f7a3f", 160, [[x - 2, y], [x + PARAMS.wind * h * 0.6, y - h], [x + 2, y]], 0.05);
  }

  // 5. flowers: a stem, then a blossom
  const colours = ["#d1495b", "#edae49", "#f7f3e3", "#8f5aa8"];
  for (let i = 0; i < PARAMS.flowerCount; i++) {
    const x = 20 + random(0, 360);
    const y = 300 + random(0, 80);
    const tipX = x + PARAMS.wind * 14;
    wash("#3f6b35", 180, [[x - 1.5, y], [tipX, y - 30], [x + 1.5, y]], 0.05);
    brush.fill(colours[i % colours.length], 210);
    brush.fillBleed(0.2);
    brush.circle(tipX, y - 32, random(9, 14));
  }
}
