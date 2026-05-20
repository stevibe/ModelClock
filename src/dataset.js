import fs from "node:fs";
import path from "node:path";

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const PREFERRED_SOFTWARE_PROJECTS = [
  "Python",
  "Node.js",
  "React",
  "React DOM",
  "TypeScript",
  "Next.js",
  "Vite",
  "Vue",
  "Angular",
  "Svelte",
  "VS Code",
  "Kubernetes",
  "Docker Compose",
  "Rust",
  "Deno",
  "Bun",
  "Tailwind CSS",
  "ESLint",
  "Webpack",
  "Rollup",
  "pnpm",
  "Django",
  "FastAPI",
  "Flask",
  "Requests",
  "NumPy",
  "pandas",
  "SciPy",
  "PyTorch",
  "TensorFlow",
  "Transformers",
  "Redis",
  "Terraform",
  "OpenAI Python",
  "Anthropic Python"
];
const PREFERRED_SOFTWARE_RANK = new Map(PREFERRED_SOFTWARE_PROJECTS.map((project, index) => [project, index]));

export function loadDataset(options) {
  const filename = options.dataset.endsWith(".json")
    ? options.dataset
    : `${options.dataset}.json`;
  const datasetPath = path.resolve(options.dataDir, filename);

  if (!fs.existsSync(datasetPath)) {
    throw new Error(`Dataset not found: ${datasetPath}`);
  }

  const raw = JSON.parse(fs.readFileSync(datasetPath, "utf8"));
  const entries = Array.isArray(raw.releases)
    ? loadSoftwareReleaseEntries(raw, options)
    : loadDateKeyedEntries(raw, options);

  if (entries.length === 0) {
    throw new Error(`Dataset ${datasetPath} has no dates with at least ${options.minItems} items.`);
  }

  return {
    name: options.dataset.replace(/\.json$/u, ""),
    path: datasetPath,
    type: Array.isArray(raw.releases) ? "software-releases" : "date-keyed",
    entries
  };
}

function loadDateKeyedEntries(raw, options) {
  return Object.entries(raw)
    .filter(([date, items]) => DATE_RE.test(date) && Array.isArray(items) && items.length >= options.minItems)
    .map(([date, items]) => ({
      date,
      items: items.slice(0, options.sampleSize).map(normalizeItem)
    }))
    .filter((entry) => entry.items.length >= options.sampleSize)
    .sort((a, b) => a.date.localeCompare(b.date));
}

function loadSoftwareReleaseEntries(raw, options) {
  const byDate = new Map();
  const versionsByProject = buildVersionsByProject(raw.releases);

  for (const release of raw.releases) {
    if (!DATE_RE.test(release.date)) {
      continue;
    }
    const items = byDate.get(release.date) ?? [];
    items.push(normalizeSoftwareRelease(release, versionsByProject));
    byDate.set(release.date, items);
  }

  return [...byDate.entries()]
    .filter(([, items]) => items.length >= options.minItems)
    .map(([date, items]) => ({
      date,
      items: [...items].sort(sortSoftwareReleaseForProbe).slice(0, options.sampleSize)
    }))
    .filter((entry) => entry.items.length >= options.minItems)
    .sort((a, b) => a.date.localeCompare(b.date));
}

function sortSoftwareReleaseForProbe(a, b) {
  const rankA = PREFERRED_SOFTWARE_RANK.get(a.project) ?? Number.MAX_SAFE_INTEGER;
  const rankB = PREFERRED_SOFTWARE_RANK.get(b.project) ?? Number.MAX_SAFE_INTEGER;
  return rankA - rankB
    || String(a.project).localeCompare(String(b.project))
    || String(a.version).localeCompare(String(b.version), undefined, { numeric: true });
}

function normalizeItem(item) {
  return {
    title: String(item.title ?? "").trim(),
    id: item.id ?? item.pageid ?? null,
    url: item.url ?? null
  };
}

function normalizeSoftwareRelease(release, versionsByProject) {
  const project = String(release.project);
  const version = String(release.version);
  const decoyVersion = makeDecoyVersion(version, versionsByProject.get(project) ?? new Set());

  return {
    title: `${project} ${version}`,
    id: version,
    url: release.url ?? null,
    project,
    version,
    ecosystem: release.ecosystem ?? null,
    package: release.package ?? null,
    prompt: release.prompt ?? `Had ${project} released version ${version} yet?`,
    decoyVersion,
    decoyPrompt: `Had ${project} released version ${decoyVersion} yet?`
  };
}

function buildVersionsByProject(releases) {
  const versions = new Map();

  for (const release of releases) {
    const project = String(release.project);
    const projectVersions = versions.get(project) ?? new Set();
    projectVersions.add(String(release.version));
    versions.set(project, projectVersions);
  }

  return versions;
}

function makeDecoyVersion(version, existingVersions) {
  const parts = String(version).split(".").map((part) => Number.parseInt(part, 10));
  if (parts.length < 3 || parts.some((part) => !Number.isInteger(part))) {
    return `${version}.999`;
  }

  for (const bump of [37, 49, 73, 91]) {
    const candidate = `${parts[0]}.${parts[1]}.${parts[2] + bump}${parts.length > 3 ? `.${parts.slice(3).join(".")}` : ""}`;
    if (!existingVersions.has(candidate)) {
      return candidate;
    }
  }

  return `${version}.999`;
}
