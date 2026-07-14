/**
 * Panneau admin : confirmation, menus, modales, avatars, rebuild polling.
 */
(function () {
  "use strict";

  function hashHue(str) {
    var h = 0;
    for (var i = 0; i < str.length; i++) {
      h = (h * 31 + str.charCodeAt(i)) % 360;
    }
    return h;
  }

  document.querySelectorAll(".admin-avatar[data-username]").forEach(function (el) {
    var u = el.getAttribute("data-username") || "";
    el.style.setProperty("--av-h", String(hashHue(u)));
  });

  var modal = document.getElementById("admin-modal");
  var confirmBtn = document.getElementById("admin-modal-confirm");
  var pendingForm = null;

  function openModal(form) {
    if (!modal) return;
    pendingForm = form;
    modal.hidden = false;
    modal.setAttribute("aria-hidden", "false");
  }

  function closeModal() {
    if (!modal) return;
    pendingForm = null;
    modal.hidden = true;
    modal.setAttribute("aria-hidden", "true");
  }

  if (modal) {
    modal.querySelectorAll("[data-modal-close]").forEach(function (el) {
      el.addEventListener("click", closeModal);
    });
    if (confirmBtn) {
      confirmBtn.addEventListener("click", function () {
        if (pendingForm) pendingForm.submit();
        closeModal();
      });
    }
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !modal.hidden) closeModal();
    });
  }

  document.querySelectorAll("form[data-confirm]").forEach(function (form) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      openModal(form);
    });
  });

  document.querySelectorAll("[data-admin-menu]").forEach(function (menu) {
    var trigger = menu.querySelector(".admin-menu__trigger");
    var panel = menu.querySelector(".admin-menu__panel");
    if (!trigger || !panel) return;

    trigger.addEventListener("click", function (e) {
      e.stopPropagation();
      var open = menu.classList.contains("is-open");
      document.querySelectorAll("[data-admin-menu].is-open").forEach(function (m) {
        m.classList.remove("is-open");
        var p = m.querySelector(".admin-menu__panel");
        var t = m.querySelector(".admin-menu__trigger");
        if (p) p.hidden = true;
        if (t) t.setAttribute("aria-expanded", "false");
      });
      if (!open) {
        menu.classList.add("is-open");
        panel.hidden = false;
        trigger.setAttribute("aria-expanded", "true");
      }
    });
  });

  document.addEventListener("click", function () {
    document.querySelectorAll("[data-admin-menu].is-open").forEach(function (menu) {
      menu.classList.remove("is-open");
      var panel = menu.querySelector(".admin-menu__panel");
      var trigger = menu.querySelector(".admin-menu__trigger");
      if (panel) panel.hidden = true;
      if (trigger) trigger.setAttribute("aria-expanded", "false");
    });
  });

  var addBtn = document.getElementById("btn-add-user");
  var addModal = document.getElementById("add-user-modal");
  if (addBtn && addModal) {
    addBtn.addEventListener("click", function () {
      addModal.hidden = false;
      addModal.setAttribute("aria-hidden", "false");
    });
    addModal.querySelectorAll("[data-close-add-user]").forEach(function (el) {
      el.addEventListener("click", function () {
        addModal.hidden = true;
        addModal.setAttribute("aria-hidden", "true");
      });
    });
  }

  var logEl = document.getElementById("rebuild-log");
  if (logEl && window.location.pathname.indexOf("/admin/corpus") !== -1) {
    function poll() {
      fetch("/admin/corpus/rebuild/status", { credentials: "same-origin" })
        .then(function (r) {
          return r.text();
        })
        .then(function (text) {
          logEl.textContent = text;
          logEl.scrollTop = logEl.scrollHeight;
        })
        .catch(function () {});
    }
    poll();
    setInterval(poll, 3000);
  }
})();
