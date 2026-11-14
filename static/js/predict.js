
"use strict";

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predictionForm");

    if (!form) {
        return;
    }

    const amountInput = document.getElementById("amount");
    const submitButton = document.getElementById("predictButton");

    // Validate transaction amount
    if (amountInput) {
        amountInput.addEventListener("input", function () {
            const amount = Number(amountInput.value);

            if (
                amountInput.value.trim() !== "" &&
                (!Number.isFinite(amount) || amount < 0)
            ) {
                amountInput.setCustomValidity(
                    "Please enter a valid, non-negative amount."
                );
            } else {
                amountInput.setCustomValidity("");
            }
        });
    }

    // Validate numeric fields without changing their values
    const numericFields = form.querySelectorAll('input[type="number"]');

    numericFields.forEach(function (field) {
        field.addEventListener("input", function () {
            const value = field.value.trim();

            if (value === "") {
                field.setCustomValidity("");
                return;
            }

            const number = Number(value);

            if (!Number.isFinite(number) || number < 0) {
                field.setCustomValidity(
                    "Please enter a valid, non-negative number."
                );
            } else {
                field.setCustomValidity("");
            }
        });
    });

    // Show a loading state during submission
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

        if (submitButton) {
            submitButton.disabled = true;
            submitButton.setAttribute("aria-busy", "true");
            submitButton.innerHTML = "Analyzing...";
        }
    });
});
