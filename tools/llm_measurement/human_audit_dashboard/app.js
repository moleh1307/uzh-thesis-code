"use strict";

const state = {
  rows: [],
  filtered: [],
  stats: {},
  currentId: null,
  filter: "unscored",
  saving: false,
  contentClassRequired: false,
};

const elements = {
  codingPanel: document.getElementById("coding-panel"),
  emptyState: document.getElementById("empty-state"),
  unitType: document.getElementById("unit-type"),
  position: document.getElementById("position"),
  auditId: document.getElementById("audit-id"),
  questionSection: document.getElementById("question-section"),
  questionText: document.getElementById("question-text"),
  targetHeading: document.getElementById("target-heading"),
  targetText: document.getElementById("target-text"),
  contentClassSection: document.getElementById("content-class-section"),
  contentClass: document.getElementById("content-class"),
  notes: document.getElementById("notes"),
  saveStatus: document.getElementById("save-status"),
  progressCount: document.getElementById("progress-count"),
  progressBar: document.getElementById("progress-bar"),
  preCount: document.getElementById("pre-count"),
  qaCount: document.getElementById("qa-count"),
  previousButton: document.getElementById("previous-button"),
  nextButton: document.getElementById("next-button"),
  clearButton: document.getElementById("clear-button"),
  unscorableButton: document.getElementById("unscorable-button"),
};

function currentRow() {
  return state.rows.find((row) => row.audit_id === state.currentId) || null;
}

function applyFilter(preserveId = true) {
  const previousId = preserveId ? state.currentId : null;
  state.filtered = state.rows.filter((row) => {
    if (state.filter === "all") return true;
    if (state.filter === "unscored") return !row.label;
    return row.unit_type === state.filter;
  });
  if (previousId && state.filtered.some((row) => row.audit_id === previousId)) {
    state.currentId = previousId;
  } else {
    state.currentId = state.filtered[0]?.audit_id || null;
  }
  render();
}

function renderStats() {
  const stats = state.stats;
  elements.progressCount.textContent = `${stats.scored || 0} / ${stats.total || 0}`;
  elements.progressBar.style.width = stats.total ? `${(stats.scored / stats.total) * 100}%` : "0%";
  elements.preCount.textContent = `${stats.pre_scored || 0} / ${stats.pre_total || 0}`;
  elements.qaCount.textContent = `${stats.qa_scored || 0} / ${stats.qa_total || 0}`;
}

function renderSelection(label) {
  document.querySelectorAll(".score-button").forEach((button) => {
    button.classList.toggle("selected", Boolean(label && label.human_ok === 1 && Number(button.dataset.score) === label.human_specificity));
  });
  elements.unscorableButton.classList.toggle("selected", Boolean(label && label.human_ok === 0));
}

function render() {
  renderStats();
  const row = currentRow();
  const empty = !row;
  elements.emptyState.hidden = !empty;
  elements.codingPanel.hidden = empty;
  elements.previousButton.disabled = empty || state.filtered.findIndex((item) => item.audit_id === state.currentId) <= 0;
  elements.nextButton.disabled = empty || state.filtered.findIndex((item) => item.audit_id === state.currentId) >= state.filtered.length - 1;
  if (empty) {
    elements.position.textContent = `0 of ${state.filtered.length}`;
    elements.auditId.textContent = "—";
    return;
  }

  const index = state.filtered.findIndex((item) => item.audit_id === row.audit_id);
  elements.unitType.textContent = row.unit_type === "qa" ? "Q&A" : "PRE";
  elements.position.textContent = `${index + 1} of ${state.filtered.length}`;
  elements.auditId.textContent = row.audit_id;
  elements.questionSection.hidden = row.unit_type !== "qa";
  elements.questionText.textContent = row.analyst_question || "";
  elements.targetHeading.textContent = row.unit_type === "qa" ? "CEO answer" : "CEO presentation segment";
  elements.targetText.textContent = row.unit_type === "qa" ? row.ceo_answer : row.ceo_presentation_segment;
  elements.contentClassSection.hidden = !state.contentClassRequired;
  elements.contentClass.value = row.label?.human_content_class || "";
  elements.notes.value = row.label?.human_notes || "";
  renderSelection(row.label);
  localStorage.setItem("specificityAuditCurrentId", row.audit_id);
}

function setSaveStatus(text, error = false) {
  elements.saveStatus.textContent = text;
  elements.saveStatus.style.color = error ? "#8f322b" : "#287a55";
}

async function saveLabel(humanOk, specificity, clear = false) {
  const row = currentRow();
  if (!row || state.saving) return;
  const contentClass = state.contentClassRequired ? elements.contentClass.value : "";
  if (!clear && state.contentClassRequired && !contentClass) {
    setSaveStatus("Select content class", true);
    return;
  }
  state.saving = true;
  setSaveStatus("Saving…");
  try {
    const response = await fetch("/api/label", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        audit_id: row.audit_id,
        human_ok: humanOk,
        human_specificity: specificity,
        human_content_class: contentClass,
        human_notes: elements.notes.value.trim(),
        clear,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Save failed");
    if (clear) {
      row.label = null;
    } else {
      row.label = {
        human_ok: humanOk,
        human_specificity: specificity,
        human_content_class: contentClass,
        human_notes: elements.notes.value.trim(),
      };
    }
    state.stats = result.stats;
    setSaveStatus("Saved");
    if (!clear && state.filter === "unscored") {
      applyFilter(false);
    } else {
      render();
    }
  } catch (error) {
    setSaveStatus(error.message || "Save failed", true);
  } finally {
    state.saving = false;
  }
}

function move(delta) {
  const index = state.filtered.findIndex((row) => row.audit_id === state.currentId);
  const target = state.filtered[index + delta];
  if (target) {
    state.currentId = target.audit_id;
    render();
    window.scrollTo({top: 0, behavior: "smooth"});
  }
}

document.querySelectorAll(".score-button").forEach((button) => {
  button.addEventListener("click", () => saveLabel(1, Number(button.dataset.score)));
});

document.querySelectorAll(".filter-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".filter-button").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.filter = button.dataset.filter;
    applyFilter(false);
  });
});

elements.unscorableButton.addEventListener("click", () => {
  if (state.contentClassRequired) elements.contentClass.value = "procedural_only";
  saveLabel(0, 0);
});
elements.clearButton.addEventListener("click", () => saveLabel(null, null, true));
elements.previousButton.addEventListener("click", () => move(-1));
elements.nextButton.addEventListener("click", () => move(1));
elements.notes.addEventListener("blur", () => {
  const row = currentRow();
  if (row?.label) saveLabel(row.label.human_ok, row.label.human_specificity);
});
elements.contentClass.addEventListener("change", () => {
  const row = currentRow();
  if (state.contentClassRequired && row?.label) {
    saveLabel(row.label.human_ok, row.label.human_specificity);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.target === elements.notes || event.target === elements.contentClass) return;
  if (["1", "2", "3", "4", "5"].includes(event.key)) {
    event.preventDefault();
    saveLabel(1, Number(event.key));
  } else if (event.key.toLowerCase() === "u") {
    event.preventDefault();
    saveLabel(0, 0);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    move(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    move(1);
  }
});

async function initialize() {
  try {
    const response = await fetch("/api/state", {cache: "no-store"});
    if (!response.ok) throw new Error("Could not load audit state");
    const result = await response.json();
    state.rows = result.rows;
    state.stats = result.stats;
    state.contentClassRequired = Boolean(result.content_class_required);
    const savedId = localStorage.getItem("specificityAuditCurrentId");
    state.currentId = state.rows.some((row) => row.audit_id === savedId) ? savedId : null;
    applyFilter(true);
    setSaveStatus("Ready");
  } catch (error) {
    setSaveStatus(error.message || "Load failed", true);
  }
}

initialize();
