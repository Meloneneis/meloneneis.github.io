/**
 * Blog post motion: progressive figure reveals + reading progress.
 */
(function () {
  "use strict";

  function prefersReducedMotion() {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function ensureStyles() {
    if (document.getElementById("blog-motion-style")) return;
    var style = document.createElement("style");
    style.id = "blog-motion-style";
    style.textContent = [
      ".blog-progress{position:fixed;top:0;left:0;height:3px;width:0;z-index:200;background:linear-gradient(90deg,#0d5c63,#c45c26);transform-origin:left center;}",
      ".page__content figure{opacity:0;transform:translateY(18px) scale(0.985);transition:opacity .85s cubic-bezier(.22,1,.36,1),transform .85s cubic-bezier(.22,1,.36,1);}",
      ".page__content figure.is-in{opacity:1;transform:none;}",
      ".page__content h2,.page__content h3{position:relative;}",
      ".page__content h2::before{content:'';position:absolute;left:-1rem;top:.15em;width:4px;height:1.1em;background:#0d5c63;opacity:0;transform:scaleY(0);transform-origin:top;transition:opacity .4s ease,transform .5s cubic-bezier(.22,1,.36,1);}",
      ".page__content h2.is-in::before{opacity:1;transform:scaleY(1);}",
      "@media (max-width:860px){.page__content h2::before{display:none;}}",
      "@media (prefers-reduced-motion:reduce){.page__content figure{opacity:1;transform:none;transition:none;}.page__content h2::before{opacity:1;transform:none;transition:none;}}",
    ].join("");
    document.head.appendChild(style);
  }

  function initProgress() {
    var bar = document.createElement("div");
    bar.className = "blog-progress";
    bar.setAttribute("aria-hidden", "true");
    document.body.appendChild(bar);

    function update() {
      var doc = document.documentElement;
      var scrollTop = doc.scrollTop || document.body.scrollTop;
      var height = doc.scrollHeight - doc.clientHeight;
      var p = height > 0 ? (scrollTop / height) * 100 : 0;
      bar.style.width = p + "%";
    }
    window.addEventListener("scroll", update, { passive: true });
    update();
  }

  function initReveals() {
    var figs = document.querySelectorAll(".page__content figure");
    var heads = document.querySelectorAll(".page__content h2");
    if (prefersReducedMotion() || !("IntersectionObserver" in window)) {
      figs.forEach(function (f) {
        f.classList.add("is-in");
      });
      heads.forEach(function (h) {
        h.classList.add("is-in");
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
      { threshold: 0.15, rootMargin: "0px 0px -6% 0px" }
    );
    figs.forEach(function (f) {
      io.observe(f);
    });
    heads.forEach(function (h) {
      io.observe(h);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (!document.body.classList.contains("layout--single") && !document.querySelector(".page__content")) {
      return;
    }
    // Only enhance post pages with figures / long content
    if (!document.querySelector(".page__content figure") && !document.querySelector(".page__content h2")) {
      return;
    }
    ensureStyles();
    initProgress();
    initReveals();
  });
})();
