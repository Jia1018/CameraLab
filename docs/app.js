const runSelect = document.querySelector("#runSelect");
const summary = document.querySelector("#summary");
const ambiguities = document.querySelector("#ambiguities");
const pairs = document.querySelector("#pairs");
const clipGrid = document.querySelector("#clipGrid");
const REFRESH_MS = 15000;

let runIndex = null;
let activeManifest = null;

async function loadJson(path) {
  const sep = path.includes("?") ? "&" : "?";
  const response = await fetch(`${path}${sep}t=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load ${path}`);
  return response.json();
}

function clipById(manifest) {
  return Object.fromEntries(manifest.clips.map((clip) => [clip.clip_id, clip]));
}

function renderSummary(manifest) {
  const cameraCount = new Set(manifest.clips.map((clip) => clip.camera_id)).size;
  const physicsCount = new Set(manifest.clips.map((clip) => clip.physics_id)).size;
  summary.innerHTML = [
    ["Run", manifest.run_id],
    ["Clips", manifest.clips.length],
    ["Cameras", cameraCount],
    ["Physics", physicsCount],
  ]
    .map(([label, value]) => `<div class="stat"><strong>${value}</strong>${label}</div>`)
    .join("");
}

function renderVideo(runBase, clip) {
  return `
    <div>
      <video controls muted loop preload="metadata" src="${runBase}/${clip.video}"></video>
      <div class="label">
        <span>${clip.clip_id}</span>
        <span>${clip.camera_id} / ${clip.physics_id}</span>
      </div>
    </div>
  `;
}

function renderAmbiguities(manifest, runBase) {
  const groups = manifest.ambiguous_equivalence_groups || [];
  const clips = clipById(manifest);
  if (!groups.length) {
    ambiguities.innerHTML = "";
    return;
  }
  ambiguities.innerHTML = `
    <h2>Ambiguity Groups</h2>
    <div class="pairs">
      ${groups
        .map((group) => {
          const videos = group.clip_ids.map((clipId) => renderVideo(runBase, clips[clipId])).join("");
          return `
            <article class="pair ambiguity">
              <h3>${group.title}</h3>
              <p class="meta">not a training pair · ${group.reason}</p>
              <div class="pairVideos">${videos}</div>
            </article>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderPairs(manifest, runBase) {
  const clips = clipById(manifest);
  pairs.innerHTML = manifest.pair_groups
    .map((group) => {
      const videos = group.clip_ids.map((clipId) => renderVideo(runBase, clips[clipId])).join("");
      return `
        <article class="pair">
          <h3>${group.title}</h3>
          <p class="meta">controlled: ${group.controlled_factor} | varied: ${group.varied_factor}</p>
          <div class="pairVideos">${videos}</div>
        </article>
      `;
    })
    .join("");
}

function renderClips(manifest, runBase) {
  clipGrid.innerHTML = manifest.clips
    .map(
      (clip) => `
        <article class="clip">
          ${renderVideo(runBase, clip)}
          <div class="badgeRow">
            <span class="badge camera">${clip.camera_id}</span>
            <span class="badge physics">${clip.physics_id}</span>
            ${clip.pair_groups.map((group) => `<span class="badge">${group}</span>`).join("")}
          </div>
        </article>
      `,
    )
    .join("");
}

async function loadRun(run) {
  const manifestPath = `assets/runs/${run.manifest}`;
  const manifest = await loadJson(manifestPath);
  activeManifest = run.manifest;
  const runBase = manifestPath.replace(/\/manifest\.json$/, "");
  renderSummary(manifest);
  renderAmbiguities(manifest, runBase);
  renderPairs(manifest, runBase);
  renderClips(manifest, runBase);
}

function sameIndex(a, b) {
  return JSON.stringify(a?.runs ?? []) === JSON.stringify(b?.runs ?? []);
}

function renderRunSelect(index) {
  const selected = runSelect.value || index.runs[0]?.manifest;
  runSelect.innerHTML = index.runs.map((run) => `<option value="${run.manifest}">${run.run_id}</option>`).join("");
  if (index.runs.some((run) => run.manifest === selected)) {
    runSelect.value = selected;
  }
}

async function refreshIndex() {
  const nextIndex = await loadJson("assets/runs/index.json");
  if (!runIndex || !sameIndex(runIndex, nextIndex)) {
    runIndex = nextIndex;
    renderRunSelect(runIndex);
    const selectedRun = runIndex.runs.find((run) => run.manifest === runSelect.value) || runIndex.runs[0];
    await loadRun(selectedRun);
    return;
  }
  const activeRun = runIndex.runs.find((run) => run.manifest === activeManifest);
  if (activeRun) await loadRun(activeRun);
}

async function main() {
  try {
    await refreshIndex();
    runSelect.addEventListener("change", () => {
      const run = runIndex.runs.find((item) => item.manifest === runSelect.value);
      loadRun(run);
    });
    setInterval(() => {
      refreshIndex().catch((error) => {
        summary.innerHTML = `<div class="stat"><strong>Refresh failed</strong>${error.message}</div>`;
      });
    }, REFRESH_MS);
  } catch (error) {
    summary.innerHTML = `<div class="stat"><strong>No runs</strong>${error.message}</div>`;
  }
}

main();
