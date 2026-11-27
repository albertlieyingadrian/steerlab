// "Which abstract is real?": hides each pair's answer until the reader clicks a card, then marks the real one.
// The text and answers are in the HTML (make_quiz.py); without scripts every answer is simply visible.
(function () {
  var root = document.getElementById("quiz");
  if (!root) return;
  root.classList.add("qz-on");
  var score = document.getElementById("qz-score");
  var right = 0, done = 0;
  root.querySelectorAll(".qz-item").forEach(function (item) {
    var real = item.getAttribute("data-real");
    item.querySelectorAll(".qz-card").forEach(function (card) {
      card.addEventListener("click", function () {
        if (item.classList.contains("qz-done")) return;
        item.classList.add("qz-done");
        done += 1;
        if (card.getAttribute("data-side") === real) right += 1;
        item.querySelectorAll(".qz-card").forEach(function (c) {
          c.classList.add(c.getAttribute("data-side") === real ? "qz-real" : "qz-wrong");
        });
        if (score) score.textContent = "You picked the real abstract " + right + " of " + done + " times.";
      });
    });
  });
})();
