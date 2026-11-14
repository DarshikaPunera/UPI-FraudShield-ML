
"use strict";

document.addEventListener("DOMContentLoaded", function () {
    // Mobile navigation toggle.
    const menuToggle = document.querySelector(".menu-toggle");
    const navigation = document.querySelector(".main-nav");

    if (menuToggle && navigation) {
        menuToggle.addEventListener("click", function () {
            const isOpen = navigation.classList.toggle("nav-open");

            menuToggle.setAttribute("aria-expanded", String(isOpen));
            menuToggle.setAttribute(
                "aria-label",
                isOpen ? "Close navigation menu" : "Open navigation menu"
            );
        });

        navigation.querySelectorAll("a").forEach(function (link) {
            link.addEventListener("click", function () {
                navigation.classList.remove("nav-open");
                menuToggle.setAttribute("aria-expanded", "false");
            });
        });

        document.addEventListener("click", function (event) {
            if (
                !navigation.contains(event.target) &&
                !menuToggle.contains(event.target)
            ) {
                navigation.classList.remove("nav-open");
                menuToggle.setAttribute("aria-expanded", "false");
            }
        });
    }

    // Allow users to dismiss flash messages.
    document.querySelectorAll(".flash-message").forEach(function (message) {
        const closeButton = message.querySelector("[data-close-flash]");

        if (closeButton) {
            closeButton.addEventListener("click", function () {
                message.remove();
            });
        }

        window.setTimeout(function () {
            if (message.isConnected) {
                message.classList.add("flash-dismissed");

                window.setTimeout(function () {
                    if (message.isConnected) message.remove();
                }, 300);
            }
        }, 6000);
    });

    // Prevent repeated submissions on forms marked for protection.
    document.querySelectorAll("form[data-prevent-double-submit]").forEach(function (form) {
        form.addEventListener("submit", function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                form.reportValidity();
                return;
            }

            if (form.dataset.submitting === "true") {
                event.preventDefault();
                return;
            }

            form.dataset.submitting = "true";

            const button = form.querySelector('button[type="submit"]');

            if (button) {
                button.disabled = true;
                button.setAttribute("aria-busy", "true");
            }
        });
    });
});
