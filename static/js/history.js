
"use strict";

document.addEventListener("DOMContentLoaded", function () {
    const searchInput = document.getElementById("historySearch");
    const filterSelect = document.getElementById("historyFilter");
    const rows = Array.from(
        document.querySelectorAll("[data-transaction-row]")
    );
    const emptyMessage = document.getElementById("historyEmpty");

    if (!searchInput || !filterSelect) return;

    function filterTransactions() {
        const searchTerm = searchInput.value.trim().toLowerCase();
        const selectedFilter = filterSelect.value;
        let visibleCount = 0;

        rows.forEach(function (row) {
            const rowText = row.textContent.toLowerCase();
            const prediction = (
                row.getAttribute("data-prediction") || "genuine"
            ).toLowerCase();

            const matchesSearch = rowText.includes(searchTerm);
            const matchesFilter =
                selectedFilter === "all" ||
                prediction === selectedFilter;

            const isVisible = matchesSearch && matchesFilter;
            row.hidden = !isVisible;

            if (isVisible) visibleCount++;
        });

        if (emptyMessage) {
            emptyMessage.hidden = visibleCount !== 0;
        }
    }

    searchInput.addEventListener("input", filterTransactions);
    filterSelect.addEventListener("change", filterTransactions);

    filterTransactions();
});
