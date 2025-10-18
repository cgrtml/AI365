// ====== config ======
const API_BASE = "http://127.0.0.1:8000";

const API_TAG = `${API_BASE}/tag`;
const API_FEEDBACK = `${API_BASE}/feedback`;

// ====== state ======
let selected = []; // [{file, url}]
const gallery = document.getElementById("gallery");
const results = document.getElementById("results");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

const topkRoomEl = document.getElementById("topk_room");
const topkFeatEl = document.getElementById("topk_feat");
const minfeatEl = document.getElementById("minfeat");

// ====== helpers ======
function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 1800);
}
function isImage(f) { return /^image\/(jpeg|png|webp)$/i.test(f.type); }
function kb(n) { return Math.round(n / 1024); }

function renderGallery() {
  gallery.innerHTML = "";
  analyzeBtn.disabled = selected.length === 0;
  for (const item of selected) {
    const card = document.createElement("div");
    card.className = "card border rounded-xl p-3";
    card.innerHTML = `
      <img src="${item.url}" class="w-full h-40 object-cover rounded-lg mb-3" alt="">
      <div class="flex items-center justify-between text-xs text-gray-600">
        <div class="truncate">${item.file.name}</div>
        <div>${kb(item.file.size)} KB</div>
      </div>
      <button class="mt-2 w-full px-3 py-1.5 rounded-lg border hover:bg-gray-50 remove">Kaldır</button>
    `;
    card.querySelector(".remove").addEventListener("click", () => {
      URL.revokeObjectURL(item.url);
      selected = selected.filter(x => x !== item);
      renderGallery();
    });
    gallery.appendChild(card);
  }
}

function addFiles(files) {
  let added = 0;
  for (const f of files) {
    if (!isImage(f)) continue;
    const url = URL.createObjectURL(f);
    selected.push({ file: f, url });
    added++;
  }
  if (added === 0) toast("Sadece .jpg / .png / .webp kabul ediliyor");
  renderGallery();
}

// ====== events: dropzone & input ======
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", e => {
  e.preventDefault(); dropzone.classList.remove("drag");
  addFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", e => addFiles(e.target.files));
clearBtn.addEventListener("click", () => {
  for (const s of selected) URL.revokeObjectURL(s.url);
  selected = []; renderGallery(); results.innerHTML = ""; statusEl.textContent = "";
});

// ====== analyze ======
analyzeBtn.addEventListener("click", async () => {
  if (selected.length === 0) return;
  statusEl.innerHTML = `<span class="spinner inline-block align-middle mr-2"></span> Analiz ediliyor…`;
  results.innerHTML = "";

  const fd = new FormData();
  for (const s of selected) fd.append("files", s.file);

  const qs = new URLSearchParams({
    topk_room: topkRoomEl.value || 3,
    topk_feat: topkFeatEl.value || 8,
    minfeat_score: minfeatEl.value || 0.02
  });

  try {
    const r = await fetch(`${API_TAG}?${qs.toString()}`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const data = await r.json();
    renderResults(data.results || []);
    statusEl.textContent = "Bitti ✅";
    toast("Analiz tamamlandı");
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Hata: Sunucuya ulaşılamadı";
    toast("Hata oluştu");
  }
});

// ====== results render + teach mode ======
function badge(text) { return `<span class="badge">${escapeHtml(text)}</span>`; }
function escapeHtml(str) { return (str || "").replace(/[&<>"']/g, s => ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;" }[s])); }
function pct(x){ return `${Math.round((x||0)*100)}%`; }

function renderResults(list) {
  results.innerHTML = "";
  for (const item of list) {
    const preview = selected.find(s => s.file.name === item.file);
    const imgURL = preview ? preview.url : "";

    const card = document.createElement("div");
    card.className = "card border rounded-2xl overflow-hidden";
    card.innerHTML = `
      <div class="p-3">
        <img src="${imgURL}" class="w-full h-48 object-contain rounded-lg bg-gray-50" alt="${item.file}">
      </div>
      <div class="px-4 pb-4">
        <div class="text-xs text-gray-500 mb-1 truncate">${item.file}</div>

        <div class="mb-2 font-semibold">Room types</div>
        <div class="flex flex-wrap gap-2 mb-4">
          ${(item.room_types||[]).map(r => badge(`${r.label} · ${pct(r.score)}`)).join("")}
        </div>

        <div class="mb-2 font-semibold">Features</div>
        <div class="flex flex-wrap gap-2 mb-4">
          ${(item.features||[]).map(f => badge(`${f.label} · ${pct(f.score)}`)).join("")}
        </div>

        <div class="grid gap-2 mb-2">
          <input class="roomFix border rounded-lg px-3 py-2 text-sm" placeholder="Doğru oda etiketleri (virgülle: kitchen, living room)">
          <input class="featFix border rounded-lg px-3 py-2 text-sm" placeholder="Doğru özellikler (virgülle: hardwood floors)">
        </div>

        <div class="flex items-center gap-2">
          <button class="save px-3 py-2 rounded-lg bg-black text-white text-sm">Teach Mode: Kaydet</button>
          <button class="copy px-3 py-2 rounded-lg border text-sm">Kopyala</button>
          <span class="grow"></span>
          <span class="sha text-xs text-gray-400">sha: ${item.sha.slice(0,8)}…</span>
        </div>
      </div>
    `;

    // copy all tags
    card.querySelector(".copy").addEventListener("click", async () => {
      const tags = [
        ...(item.room_types||[]).map(x => x.label),
        ...(item.features||[]).map(x => x.label),
      ];
      await navigator.clipboard.writeText(tags.join(", "));
      toast("Etiketler kopyalandı");
    });

    // feedback
    card.querySelector(".save").addEventListener("click", async () => {
      const roomInput = card.querySelector(".roomFix").value.trim();
      const featInput = card.querySelector(".featFix").value.trim();
      const params = new URLSearchParams({ sha: item.sha, accept: "true" });
      if (roomInput) {
        for (const r of roomInput.split(",").map(s => s.trim()).filter(Boolean)) {
          params.append("room_types", r);
        }
      }
      if (featInput) {
        for (const f of featInput.split(",").map(s => s.trim()).filter(Boolean)) {
          params.append("features", f);
        }
      }

      try {
        const res = await fetch(`${API_FEEDBACK}?${params.toString()}`, { method: "POST" });
        if (!res.ok) throw new Error(`${res.status}`);
        toast("Öğretme kaydedildi ✓");
      } catch (e) {
        console.error(e);
        toast("Kaydedilemedi");
      }
    });

    results.appendChild(card);
  }
}
