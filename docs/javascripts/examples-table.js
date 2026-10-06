document.addEventListener("DOMContentLoaded", function () {
  // Relative to examples.html, the only page that loads this file; written by
  // write_examples_index in conf.py.
  var JSON_URL = "_static/examples.json";
  var FIELDS = [
    { id: "model",             label: "Model" },
    { id: "observation_model", label: "Observation model" },
    { id: "signal",            label: "Signal" },
    { id: "topic",             label: "Topic" },
    { id: "data",              label: "Data" },
  ];

  // field id -> Set of selected values, mirrored in the query string so a
  // filtered view can be linked to, e.g. ?signal=spike-counts&model=GLM,PopulationGLM
  var selected = {};

  function asList(value) {
    return Array.isArray(value) ? value : [value];
  }

  function readQuery() {
    var params = new URLSearchParams(window.location.search);
    FIELDS.forEach(function (f) {
      var raw = params.get(f.id);
      selected[f.id] = new Set(raw ? raw.split(",") : []);
    });
  }

  function writeQuery() {
    var parts = [];
    FIELDS.forEach(function (f) {
      if (selected[f.id].size) parts.push(f.id + "=" + Array.from(selected[f.id]).join(","));
    });
    history.replaceState(null, "", parts.length ? "?" + parts.join("&") : window.location.pathname);
  }

  // OR within a field, AND across fields
  function matches(example) {
    return FIELDS.every(function (f) {
      var wanted = selected[f.id];
      return !wanted.size || asList(example[f.id]).some(function (v) { return wanted.has(v); });
    });
  }

  // ── filters ──────────────────────────────────────────────────────────────

  function buildFilters(examples, table) {
    var container = document.getElementById("examples-filters");

    function refresh() {
      container.querySelectorAll(".examples-chip").forEach(function (chip) {
        chip.setAttribute("aria-pressed", selected[chip.dataset.field].has(chip.dataset.value));
      });
      writeQuery();
      table.draw();
    }

    FIELDS.forEach(function (f) {
      // values some example carries, plus any asked for in the URL, so every
      // active filter has a chip to clear it with
      var values = new Set(selected[f.id]);
      examples.forEach(function (e) {
        asList(e[f.id]).forEach(function (v) { values.add(v); });
      });

      var row = document.createElement("div");
      row.className = "examples-filter-row";
      var label = document.createElement("span");
      label.className = "examples-filter-label";
      label.textContent = f.label;
      row.appendChild(label);

      Array.from(values).sort().forEach(function (v) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "examples-chip";
        chip.textContent = v;
        chip.dataset.field = f.id;
        chip.dataset.value = v;
        chip.addEventListener("click", function () {
          if (selected[f.id].has(v)) selected[f.id].delete(v);
          else selected[f.id].add(v);
          refresh();
        });
        row.appendChild(chip);
      });
      container.appendChild(row);
    });

    var clear = document.createElement("button");
    clear.type = "button";
    clear.className = "examples-clear";
    clear.textContent = "Clear filters";
    clear.addEventListener("click", function () {
      FIELDS.forEach(function (f) { selected[f.id].clear(); });
      refresh();
    });
    container.appendChild(clear);

    refresh();
  }

  // ── table ────────────────────────────────────────────────────────────────

  function tagList(value) {
    return asList(value).join(", ");
  }

  readQuery();
  fetch(JSON_URL)
    .then(function (response) { return response.json(); })
    .then(function (examples) {
      DataTable.ext.search.push(function (settings, searchData, index, example) {
        return matches(example);
      });
      var table = new DataTable("#examples-table", {
        data: examples,
        paging: false,
        columns: [
          {
            title: "Example",
            data: "title",
            render: function (title, type, example) {
              if (type !== "display") return title + " " + example.description;
              return '<a href="' + example.url + '">' + title + "</a>" +
                     '<div class="examples-description">' + example.description + "</div>";
            },
          },
          { title: "Model",             data: "model",             render: tagList },
          { title: "Observation model", data: "observation_model", render: tagList },
          { title: "Signal",            data: "signal",            render: tagList },
          { title: "Topic",             data: "topic",             render: tagList },
          { title: "Data",              data: "data",              render: tagList },
        ],
      });
      buildFilters(examples, table);
    });
});
