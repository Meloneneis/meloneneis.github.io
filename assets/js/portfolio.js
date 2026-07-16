/**
 * Portfolio interactions: scroll reveals, nav contrast, soft counters.
 */
(function () {
  "use strict";

  function prefersReducedMotion() {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function initReveals() {
    var nodes = document.querySelectorAll(".reveal");
    if (!nodes.length) return;

    if (prefersReducedMotion() || !("IntersectionObserver" in window)) {
      nodes.forEach(function (n) {
        n.classList.add("is-in");
      });
      return;
    }

    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-in");
            io.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.12 }
    );

    nodes.forEach(function (n) {
      io.observe(n);
    });
  }

  function animateCounters() {
    var stats = document.querySelectorAll("[data-count]");
    if (!stats.length) return;

    function run(el) {
      var end = parseFloat(el.getAttribute("data-count"));
      var suffix = el.getAttribute("data-suffix") || "";
      var prefix = el.getAttribute("data-prefix") || "";
      var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
      if (prefersReducedMotion()) {
        el.textContent = prefix + end.toFixed(decimals) + suffix;
        return;
      }
      var start = 0;
      var duration = 1100;
      var t0 = null;
      function step(ts) {
        if (!t0) t0 = ts;
        var p = Math.min(1, (ts - t0) / duration);
        var eased = 1 - Math.pow(1 - p, 3);
        var val = start + (end - start) * eased;
        el.textContent = prefix + val.toFixed(decimals) + suffix;
        if (p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }

    if (!("IntersectionObserver" in window)) {
      stats.forEach(run);
      return;
    }

    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            run(entry.target);
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    stats.forEach(function (s) {
      io.observe(s);
    });
  }

  function yearStamp() {
    var el = document.getElementById("year-stamp");
    if (el) el.textContent = String(new Date().getFullYear());
  }

  document.addEventListener("DOMContentLoaded", function () {
    initReveals();
    animateCounters();
    yearStamp();
  });
})();
