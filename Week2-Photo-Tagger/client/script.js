const API = "http://127.0.0.1:8000/tag";

const filesEl  = document.getElementById("files");
const runBtn   = document.getElementById("run");
const statusEl = document.getElementById("status");
const outEl    = document.getElementById("out");

runBtn.addEventListener("click", async () => {
  const files = filesEl.files;
  if (!files || !files.length) {
    alert("Please choose a pic");
    return;
  }

  const fd = new FormData();
  for (const f of files) {
    fd.append("files", f);
  }

  statusEl.textContent = "Processing...";
  outEl.innerHTML = "";

  const r = await fetch(API, { method: "POST", body: fd });
  const data = await r.json();

  renderResults(data.results || []);
  statusEl.textContent = "Done ✅";
});

function renderResults(results) {
  outEl.innerHTML = "";
  for (const item of results) {
    const div = document.createElement("div");
    div.className = "p-4 bg-white border rounded";
    div.innerHTML = `
      <div class="font-bold mb-2">${item.file}</div>
      <div><strong>Room types:</strong> ${item.room_types.map(r => r.label).join(", ")}</div>
      <div><strong>Features:</strong> ${item.features.map(f => f.label).join(", ")}</div>
    `;
    outEl.appendChild(div);
  }
}
