/* Animated covers, one behaviour everywhere they appear.

   A big cover (a note's hero, the research page's Start here lead, a day's lead in Latest) plays ONCE and rests on
   the full figure. It waits until 60% of it is in view, and if it scrolls away mid-play it jumps to the finished
   frame, so it is never left half-drawn. Under it: a 2px progress line and a pill, "Skip animation" while it runs,
   "Replay" after. A small cover (thumbnail strip, cards) stays still and loops only on hover or keyboard focus, and
   only where hovering exists; a small play mark says so.

   Covers are either inline SVG (svg.cover-anim, whose keyframes run only under .is-live) or a muted video named by
   data-anim="<base>" on the cover image (<base>.webm, <base>.mp4; the image is the poster, the full figure).
   With reduced motion, or without this script, every cover is the still full figure and no control is shown.

     SteerCover.player(figure, media, {start})  // media: {svgs: [...]} or {video: el}; start: share in view (0.6)
     SteerCover.video(img)               // a muted <video> for a data-anim image, poster = the image
     SteerCover.hover(img)               // a small cover that loops on hover or focus
*/
(function () {
  "use strict";
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  var canHover = window.matchMedia("(hover: hover)");
  var IO = "IntersectionObserver" in window;
  var DURATION = 8; // seconds; every cover is drawn over the same 8 s
  var ICON = {
    skip: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 3.5 9.5 8 4 12.5z"/><path d="M12 3.5v9"/></svg>',
    replay: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.2 8a4.8 4.8 0 1 0 1.5-3.5"/><path d="M3 2.5v2.8h2.8"/></svg>',
    play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5.5 3.5 11.5 8l-6 4.5z"/></svg>',
  };

  function motionOK() { return !reduce.matches && IO; }

  function video(img) {
    var v = document.createElement("video"), base = img.getAttribute("data-anim");
    v.className = img.className;
    v.width = img.width; v.height = img.height;
    v.muted = true; v.playsInline = true; v.preload = "none";
    v.setAttribute("muted", ""); v.setAttribute("playsinline", ""); v.setAttribute("aria-hidden", "true");
    v.poster = img.getAttribute("src");
    [["webm", "video/webm"], ["mp4", "video/mp4"]].forEach(function (f) {
      var s = document.createElement("source");
      s.src = base + "." + f[0]; s.type = f[1];
      v.appendChild(s);
    });
    return v;
  }

  // What a player drives: start from the beginning, jump to the end, and how far along it is (0..1).
  function svgMedia(svgs) {
    var t0 = 0;
    return {
      idle: function () { svgs.forEach(function (s) { s.classList.add("is-live", "is-idle"); }); },
      play: function () {
        svgs.forEach(function (s) { s.classList.remove("is-live", "is-idle"); void s.getBoundingClientRect(); s.classList.add("is-live"); });
        t0 = performance.now();
      },
      finish: function () { svgs.forEach(function (s) { s.classList.remove("is-live", "is-idle"); }); },
      progress: function () { return (performance.now() - t0) / 1000 / DURATION; },
    };
  }
  function videoMedia(v) {
    return {
      idle: function () {},
      play: function () {
        if (v.preload === "none") { v.preload = "auto"; v.load(); }
        try { v.currentTime = 0; } catch (e) {}
        var p = v.play();
        if (p && p.catch) p.catch(function () {});
      },
      finish: function () {
        v.pause();
        if (v.duration) { try { v.currentTime = Math.max(0, v.duration - 0.05); } catch (e) {} }
      },
      progress: function () { return v.duration ? v.currentTime / v.duration : 0; },
    };
  }

  function player(fig, src, opts) {
    if (!motionOK()) return null;
    var start = (opts && opts.start) || 0.6; // share of the figure in view that starts it
    var media = src.video ? videoMedia(src.video) : svgMedia(src.svgs);
    var bar = document.createElement("div"), fill = document.createElement("span"), btn = document.createElement("button");
    bar.className = "cover-bar";
    bar.innerHTML = '<span class="cover-prog" aria-hidden="true"></span>';
    bar.firstChild.appendChild(fill);
    btn.type = "button"; btn.className = "cover-ctl";
    bar.appendChild(btn);
    var cap = fig.querySelector("figcaption");
    fig.insertBefore(bar, cap || null);
    var state = "idle", raf = 0, seen;

    function label(s) {
      btn.innerHTML = s === "run" ? ICON.skip + "<span>Skip animation</span>" : ICON.replay + "<span>Replay</span>";
    }
    function tick() {
      var p = Math.min(1, media.progress());
      fill.style.transform = "scaleX(" + p + ")";
      if (state !== "run") return;
      if (p >= 1) { done(); return; }
      raf = requestAnimationFrame(tick);
    }
    function run() {
      state = "run"; label("run");
      media.play();
      cancelAnimationFrame(raf); raf = requestAnimationFrame(tick);
    }
    function done() {
      cancelAnimationFrame(raf);
      state = "done"; label("done");
      media.finish();
      fill.style.transform = "scaleX(1)";
    }
    btn.addEventListener("click", function () { if (state === "run") done(); else run(); });
    if (src.video) src.video.addEventListener("ended", function () { if (state === "run") done(); });
    media.idle(); label("run");
    seen = new IntersectionObserver(function (entries) {
      var e = entries[entries.length - 1];
      if (e.isIntersecting && e.intersectionRatio >= start - 0.001 && state === "idle") run();
      else if (!e.isIntersecting && state === "run") done();
    }, { threshold: [0, start] });
    seen.observe(fig);
    return {
      replay: function () { run(); },
      finish: function () { if (state === "run") done(); },
      destroy: function () { seen.disconnect(); cancelAnimationFrame(raf); if (state === "run") media.finish(); bar.remove(); },
      bar: bar,
    };
  }

  function hover(img) {
    if (!motionOK() || !canHover.matches) return;
    var link = img.closest("a");
    if (!link) return;
    var wrap = document.createElement("span"), mark = document.createElement("span"), v = null;
    wrap.className = "cover-thumb";
    mark.className = "cover-mark"; mark.setAttribute("aria-hidden", "true"); mark.innerHTML = ICON.play;
    img.parentNode.insertBefore(wrap, img);
    wrap.appendChild(img); wrap.appendChild(mark);
    function on() {
      if (!v) {
        v = video(img); v.className = ""; v.loop = true; v.preload = "auto";
        v.style.borderRadius = getComputedStyle(img).borderRadius;
        wrap.insertBefore(v, mark);
      }
      v.style.display = "";
      try { v.currentTime = 0; } catch (e) {}
      var p = v.play();
      if (p && p.catch) p.catch(function () {});
    }
    function off() { if (v) { v.pause(); v.style.display = "none"; } }
    link.addEventListener("mouseenter", on);
    link.addEventListener("mouseleave", off);
    link.addEventListener("focus", function () { if (link.matches(":focus-visible")) on(); });
    link.addEventListener("blur", off);
  }

  window.SteerCover = { player: player, video: video, hover: hover, motionOK: motionOK };

  // A note's hero: figure.hero-anim with its inline SVGs. It plays on load, as soon as a quarter of it shows (the
  // hero often starts just above the fold); the research page's leads wait for 60%.
  document.querySelectorAll("figure.hero-anim").forEach(function (fig) {
    var svgs = Array.prototype.slice.call(fig.querySelectorAll("svg.cover-anim"));
    if (svgs.length) player(fig, { svgs: svgs }, { start: 0.25 });
  });
})();
