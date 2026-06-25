document.addEventListener("DOMContentLoaded", function () {
  var CSV_URL = "_static/aggregate_summary.csv";
  var TABLE_COLS = ["device", "solver_name", "fit_time_s", "converged", "iter_num", "compile_time_fraction"];

  // recordings first so it drives the sort order
  var CATEGORIES = [
    { id: "recordings", label: "Recordings", source: null },
    { id: "small",      label: "Small sim",  source: "synthetic", sample_size: 100,    feature_dim: 1,   pop_size: 1  },
    { id: "large",      label: "Large sim",  source: "synthetic", sample_size: 100000, feature_dim: 100, pop_size: 20 },
  ];
  var COLORS = { recordings: "#54A24B", small: "#4C78A8", large: "#F58518" };

  function fmt(c, val) {
    if (typeof val !== "number") return val != null ? String(val) : "";
    if (c === "fit_time_s" || c === "compile_time_fraction") return val.toFixed(3);
    if (c === "iter_num") return val.toFixed(1);
    return String(val);
  }

  function matchCategory(row, cat) {
    if (cat.source === null) return row.data_source !== "synthetic";
    return row.data_source === "synthetic" &&
           row.sample_size == cat.sample_size &&
           row.feature_dim == cat.feature_dim &&
           row.pop_size    == cat.pop_size;
  }

  // ── chart ────────────────────────────────────────────────────────────────

  function buildChart(rows, datasetFilter) {
    var ridgeRows = rows.filter(function (r) {
      return (!r.regularizer || r.regularizer === "Ridge") &&
             r.converged !== false && r.converged !== "False";
    });

    var activeCats = datasetFilter === "all"
      ? CATEGORIES
      : CATEGORIES.filter(function (c) { return c.id === datasetFilter; });

    // sort solvers by recordings GPU time, fall back to large sim GPU
    function bestTime(solver, dev, cats) {
      for (var i = 0; i < cats.length; i++) {
        var m = ridgeRows.filter(function (r) {
          return r.solver_name === solver && r.device === dev && matchCategory(r, cats[i]);
        });
        if (m.length) return m[0].fit_time_s;
      }
      return Infinity;
    }

    var sortDev = document.getElementById("filter-sort").value;
    var solvers = Array.from(new Set(ridgeRows.map(function (r) { return r.solver_name; })));
    solvers.sort(function (a, b) {
      return bestTime(a, sortDev, activeCats) - bestTime(b, sortDev, activeCats);
    });

    var traces = [];
    ["gpu", "cpu"].forEach(function (dev, di) {
      var ax = di === 0 ? "" : "2";
      activeCats.forEach(function (cat) {
        var x = [], y = [];
        solvers.forEach(function (s) {
          var m = ridgeRows.filter(function (r) {
            return r.solver_name === s && r.device === dev && matchCategory(r, cat);
          });
          if (m.length) { x.push(s); y.push(m[0].fit_time_s); }
        });
        if (x.length) traces.push({
          x: x, y: y,
          name: cat.label,
          legendgroup: cat.id,
          showlegend: di === 0,
          type: "bar",
          marker: { color: COLORS[cat.id] },
          xaxis: "x" + ax,
          yaxis: "y" + ax,
        });
      });
    });

    var scaleType = document.getElementById("filter-scale").value;
    var yAxisBase = scaleType === "log"
      ? { type: "log", exponentformat: "power", minor: { ticks: "inside", tickmode: "auto", nticks: 9 }, autorange: true }
      : { type: "linear", autorange: true };

    Plotly.react("benchmark-chart", traces, {
      barmode: "group",
      grid: { rows: 1, columns: 2, pattern: "independent" },
      xaxis:  { tickangle: -35, domain: [0, 0.47] },
      yaxis:  Object.assign({ title: "Fit time (s)" }, yAxisBase),
      xaxis2: { tickangle: -35, domain: [0.53, 1], anchor: "y2" },
      yaxis2: Object.assign(
        document.getElementById("filter-share-y").checked ? { matches: "y" } : {},
        yAxisBase, { anchor: "x2" }
      ),
      annotations: [
        { text: "GPU", xref: "paper", yref: "paper", x: 0.235, y: 1.05,
          showarrow: false, font: { size: 15, weight: "bold" }, xanchor: "center" },
        { text: "CPU", xref: "paper", yref: "paper", x: 0.765, y: 1.05,
          showarrow: false, font: { size: 15, weight: "bold" }, xanchor: "center" },
      ],
      legend: { orientation: "h", y: -0.45, x: 0.5, xanchor: "center" },
      margin: { b: 180, t: 50, l: 70, r: 20 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor:  "rgba(0,0,0,0)",
    }, { responsive: true });
  }

  // ── tables ───────────────────────────────────────────────────────────────

  var _dtInstances = {};

  function buildTable(tableId, catIdx, rows) {
    var cat = CATEGORIES.find(function (c, i) { return i === catIdx; });
    // remap: tables use fixed order recordings=0, small=1, large=2
    var tableCatOrder = [
      CATEGORIES.find(function (c) { return c.id === "recordings"; }),
      CATEGORIES.find(function (c) { return c.id === "small"; }),
      CATEGORIES.find(function (c) { return c.id === "large"; }),
    ];
    cat = tableCatOrder[catIdx];
    var catRows = rows.filter(function (r) { return matchCategory(r, cat); });
    var tbody = document.querySelector("#" + tableId + " tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    catRows.forEach(function (r) {
      var tr = document.createElement("tr");
      var vTd = document.createElement("td");
      vTd.textContent = r.version != null ? r.version : "";
      tr.appendChild(vTd);
      TABLE_COLS.forEach(function (c) {
        var td = document.createElement("td");
        td.textContent = fmt(c, r[c]);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    if (_dtInstances[tableId]) { _dtInstances[tableId].destroy(); }
    document.querySelectorAll("#" + tableId + " thead th select").forEach(function (s) { s.remove(); });
    _dtInstances[tableId] = new DataTable("#" + tableId, {
      pageLength: 25,
      dom: "lrtip",
      columnDefs: [{ targets: 0, visible: false }],
      order: [[3, "asc"]],
      initComplete: function () {
        var api = this.api();
        [1, 4].forEach(function (colIdx) {
          var column = api.column(colIdx);
          var th = column.header();
          var sel = document.createElement("select");
          sel.style.cssText = "display:block;width:100%;margin-top:4px;font-size:0.82em;";
          var allOpt = document.createElement("option");
          allOpt.value = ""; allOpt.textContent = "All";
          sel.appendChild(allOpt);
          var vals = {};
          column.data().each(function (d) { vals[String(d)] = true; });
          Object.keys(vals).sort().forEach(function (v) {
            var opt = document.createElement("option");
            opt.value = v; opt.textContent = v;
            sel.appendChild(opt);
          });
          sel.addEventListener("change", function () {
            column.search(this.value).draw();
          });
          th.appendChild(sel);
        });
      },
    });
  }

  // ── wiring ───────────────────────────────────────────────────────────────

  Papa.parse(CSV_URL, {
    download: true, header: true, dynamicTyping: true,
    complete: function (results) {
      var allRows = results.data.filter(function (r) { return r.version; });

      var versionSel = document.getElementById("filter-version");
      var versions = Array.from(new Set(allRows.map(function (r) { return r.version; }))).sort();
      versions.forEach(function (v) {
        var opt = document.createElement("option");
        opt.value = v; opt.textContent = v;
        versionSel.appendChild(opt);
      });
      if (versions.indexOf("latest") !== -1) versionSel.value = "latest";

      function currentVersion() { return versionSel.value; }
      function currentDataset() { return document.getElementById("filter-dataset").value; }

      function refresh() {
        var ver = currentVersion();
        var filteredRows = allRows.filter(function (r) {
          return !ver || r.version === ver;
        });
        buildChart(filteredRows, currentDataset());
        buildTable("table-small-sim",  1, filteredRows);
        buildTable("table-large-sim",  2, filteredRows);
        buildTable("table-recordings", 0, filteredRows);
      }

      versionSel.addEventListener("change", refresh);
      document.getElementById("filter-dataset").addEventListener("change", refresh);
      document.getElementById("filter-sort").addEventListener("change", refresh);
      document.getElementById("filter-scale").addEventListener("change", refresh);
      document.getElementById("filter-share-y").addEventListener("change", refresh);
      refresh();
    },
  });
});
