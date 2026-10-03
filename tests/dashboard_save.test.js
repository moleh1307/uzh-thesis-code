"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

function fixture() {
  const controls = new Map();
  const documentEvents = {};
  const element = (id) => {
    if (!controls.has(id)) controls.set(id, {
      id, value: "", disabled: false, style: {}, dataset: {}, events: {},
      classList: { toggle() {}, add() {}, remove() {} },
      addEventListener(name, handler) { this.events[name] = handler; },
    });
    return controls.get(id);
  };
  const scores = [1, 2, 3, 4, 5].map((score) => {
    const control = element(`score-${score}`);
    control.dataset.score = String(score);
    return control;
  });
  const filters = ["all", "unscored", "pre", "qa"].map((filter) => {
    const control = element(`filter-${filter}`);
    control.dataset.filter = filter;
    return control;
  });
  const buttons = ["previous-button", "next-button", "clear-button", "unscorable-button", "source-missing-button"]
    .map(element).concat(scores, filters);
  const requests = [];
  const context = vm.createContext({
    document: {
      getElementById: element,
      querySelectorAll(selector) {
        if (selector === ".score-button") return scores;
        if (selector === ".filter-button") return filters;
        if (selector === "button") return buttons;
        throw new Error(`Unexpected selector ${selector}`);
      },
      addEventListener(name, handler) { documentEvents[name] = handler; },
    },
    localStorage: { setItem() {}, getItem() { return null; } },
    window: { scrollTo() {} },
    fetch: (url, options) => new Promise((resolve, reject) => {
      requests.push({url, payload: JSON.parse(options.body), resolve, reject});
    }),
  });
  const source = fs.readFileSync(path.join(__dirname, "../tools/llm_measurement/human_audit_dashboard/app.js"), "utf8");
  // Isolate the save workflow from the initial GET; exercise the unchanged production functions.
  vm.runInContext(source.replace(/\ninitialize\(\);\s*$/, ""), context);
  vm.runInContext(`state.rows = [
    {audit_id: "A", unit_type: "qa", ceo_answer: "Synthetic answer A", label: {human_ok: 1, human_specificity: 3, human_content_class: "substantive", human_notes: "old A"}},
    {audit_id: "B", unit_type: "qa", ceo_answer: "Synthetic answer B", label: {human_ok: 1, human_specificity: 4, human_content_class: "substantive", human_notes: "old B"}}
  ]; state.filter = "all"; state.currentId = "A"; state.contentClassRequired = true; applyFilter();`, context);
  return { context, element, buttons, filters, requests, documentEvents,
    run: (code) => vm.runInContext(code, context),
    respond(index, ok = true) {
      requests[index].resolve({ok, json: async () => ok ? {stats: {total: 2, reviewed: 2}} : {error: "External file conflict"}});
    },
  };
}

test("pending save locks score, class, notes, navigation, filters and keyboard until success", async () => {
  const f = fixture();
  f.element("notes").value = "draft A";
  f.element("content-class").value = "mixed";
  const saving = f.run("saveLabel(1, 5)");
  assert.equal(f.requests.length, 1);
  assert.ok(f.buttons.every((button) => button.disabled));
  assert.equal(f.element("notes").disabled, true);
  assert.equal(f.element("content-class").disabled, true);
  f.run("move(1)");
  f.filters[1].events.click();
  let prevented = false;
  f.documentEvents.keydown({key: "4", preventDefault() { prevented = true; }});
  assert.equal(prevented, true);
  assert.equal(f.run("state.currentId"), "A");
  assert.equal(f.run("state.filter"), "all");
  assert.equal(f.requests.length, 1);
  f.respond(0);
  await saving;
  assert.equal(f.run("state.saving"), false);
  assert.equal(f.element("notes").value, "draft A");
  assert.equal(f.run("currentRow().label.human_specificity"), 5);
  assert.equal(f.run("currentRow().label.human_content_class"), "mixed");
  assert.equal(f.element("next-button").disabled, false);
  f.run("move(1)");
  assert.equal(f.run("state.currentId"), "B");
  assert.equal(f.element("notes").value, "old B");
  f.element("notes").value = "draft B";
  f.element("content-class").value = "mixed";
  const saveB = f.run("saveLabel(1, 2)");
  assert.equal(f.requests[1].payload.audit_id, "B");
  assert.equal(f.requests[1].payload.human_notes, "draft B");
  f.respond(1);
  await saveB;
  assert.equal(f.run("currentRow().label.human_specificity"), 2);
});

for (const failure of ["http", "network"]) {
  test(`${failure} failure preserves draft class, notes and previous saved label for retry`, async () => {
    const f = fixture();
    f.element("notes").value = "retain this draft";
    f.element("content-class").value = "mixed";
    const saving = f.run("saveLabel(1, 5)");
    if (failure === "http") f.respond(0, false);
    else f.requests[0].reject(new Error("Network interrupted"));
    await saving;
    assert.equal(f.element("notes").value, "retain this draft");
    assert.equal(f.element("content-class").value, "mixed");
    assert.equal(f.run("currentRow().label.human_specificity"), 3);
    assert.equal(f.run('state.drafts.get("A").human_specificity'), 5);
    assert.equal(f.element("notes").disabled, false);
    assert.equal(f.element("content-class").disabled, false);
    f.run("move(1); move(-1);");
    assert.equal(f.element("notes").value, "retain this draft");
    assert.equal(f.element("content-class").value, "mixed");
    const retry = f.run("saveLabel(1, 5)");
    assert.equal(f.requests[1].payload.human_notes, "retain this draft");
    f.respond(1);
    await retry;
    assert.equal(f.run("currentRow().label.human_specificity"), 5);
  });
}

test("source-quality validation and successful clear remain intact", async () => {
  const f = fixture();
  f.element("content-class").value = "uncertain";
  f.element("notes").value = "";
  await f.run('saveLabel("", "")');
  assert.equal(f.requests.length, 0);
  assert.match(f.element("save-status").textContent, /reason required/);
  f.element("notes").value = "Synthetic incomplete source";
  const saving = f.run('saveLabel("", "")');
  f.respond(0);
  await saving;
  assert.equal(f.run("currentRow().label.human_ok"), "");
  const clear = f.run("saveLabel(null, null, true)");
  f.respond(1);
  await clear;
  assert.equal(f.run("currentRow().label"), null);
  assert.equal(f.element("notes").value, "");
});

test("unscored filter advances only after success and preserves a failed draft", async () => {
  const f = fixture();
  f.run('state.rows.forEach(row => row.label = null); state.filter = "unscored"; applyFilter();');
  f.element("content-class").value = "substantive";
  f.element("notes").value = "new A";
  const failed = f.run("saveLabel(1, 4)");
  f.respond(0, false);
  await failed;
  assert.equal(f.run("state.currentId"), "A");
  assert.equal(f.element("notes").value, "new A");
  const saved = f.run("saveLabel(1, 4)");
  f.respond(1);
  await saved;
  assert.equal(f.run("state.currentId"), "B");
  assert.equal(f.element("notes").value, "");
});
