document.addEventListener("DOMContentLoaded", function () {
  var CSV_URL = "_static/aggregate_summary.csv";
  var COLS = ["device", "solver_name", "fit_time_s", "converged", "iter_num", "compile_time_fraction"];

  function fmt(c, val) {
    if (typeof val !== "number") return val != null ? String(val) : "";
    if (c === "fit_time_s" || c === "compile_time_fraction") return val.toFixed(3);
    if (c === "iter_num") return val.toFixed(1);
    return String(val);
  }

  function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function makeVersionFilter(selectId, api) {
    var sel = document.getElementById(selectId);
    api.column(0).data().unique().sort().each(function (v) {
      var opt = document.createElement("option");
      opt.value = v; opt.textContent = v;
      sel.appendChild(opt);
    });
    var hasLatest = api.column(0).data().toArray().indexOf("latest") !== -1;
    if (hasLatest) {
      sel.value = "latest";
      api.column(0).search("^latest$", true, false).draw();
    }
    sel.addEventListener("change", function () {
      var v = this.value;
      api.column(0).search(v ? "^" + escapeRegex(v) + "$" : "", true, false).draw();
    });
  }

  function buildTable(tableId, selectId, rows) {
    var tbody = document.querySelector("#" + tableId + " tbody");
    if (!tbody) return;
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      var vTd = document.createElement("td");
      vTd.textContent = r["version"] != null ? r["version"] : "";
      tr.appendChild(vTd);
      COLS.forEach(function (c) {
        var td = document.createElement("td");
        td.textContent = fmt(c, r[c]);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    return new DataTable("#" + tableId, {
      pageLength: 25,
      searching: false,
      columnDefs: [{ targets: 0, visible: false }],
      order: [[3, "asc"]],
      initComplete: function () { makeVersionFilter(selectId, this.api()); },
    });
  }

  Papa.parse(CSV_URL, {
    download: true,
    header: true,
    dynamicTyping: true,
    complete: function (results) {
      var rows = results.data.filter(function (r) { return r.version; });

      buildTable("table-small-sim", "filter-small-sim",
        rows.filter(function (r) {
          return r.data_source === "synthetic" &&
                 r.sample_size == 100 && r.feature_dim == 1 && r.pop_size == 1;
        }));

      buildTable("table-large-sim", "filter-large-sim",
        rows.filter(function (r) {
          return r.data_source === "synthetic" &&
                 r.sample_size == 100000 && r.feature_dim == 100 && r.pop_size == 20;
        }));

      buildTable("table-recordings", "filter-recordings",
        rows.filter(function (r) { return r.data_source !== "synthetic"; }));
    },
  });
});
