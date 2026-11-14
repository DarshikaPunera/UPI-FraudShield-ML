
"use strict";

document.addEventListener("DOMContentLoaded", function () {
    // Show or hide password fields.
    document.querySelectorAll("[data-toggle-password]").forEach(function (button) {
        button.addEventListener("click", function () {
            const fieldId = button.getAttribute("data-toggle-password");
            const field = document.getElementById(fieldId);

            if (!field) return;

            const showPassword = field.type === "password";
            field.type = showPassword ? "text" : "password";
            button.textContent = showPassword ? "Hide" : "Show";
            button.setAttribute(
                "aria-label",
                showPassword ? "Hide password" : "Show password"
            );
        });
    });

    // Validate password confirmation on registration.
    const registerForm = document.getElementById("registerForm");
    const password = document.getElementById("password");
    const confirmPassword = document.getElementById("confirm_password");

    if (registerForm && password && confirmPassword) {
        function validatePasswordMatch() {
            if (
                confirmPassword.value !== "" &&
                password.value !== confirmPassword.value
            ) {
                confirmPassword.setCustomValidity("Passwords do not match.");
            } else {
                confirmPassword.setCustomValidity("");
            }
        }

        password.addEventListener("input", validatePasswordMatch);
        confirmPassword.addEventListener("input", validatePasswordMatch);

        registerForm.addEventListener("submit", function (event) {
            validatePasswordMatch();

            if (!registerForm.checkValidity()) {
                event.preventDefault();
                registerForm.reportValidity();
            }
        });
    }

    // Prevent accidental repeated form submissions.
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

            const submitButton = form.querySelector('button[type="submit"]');

            if (submitButton) {
                submitButton.disabled = true;
                submitButton.setAttribute("aria-busy", "true");
            }
        });
    });
});
