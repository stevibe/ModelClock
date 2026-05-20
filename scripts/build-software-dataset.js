#!/usr/bin/env node

import fs from "node:fs/promises";

const OUTPUT_PATH = "data/software-releases.json";
const MIN_DATE = "2023-01-01";
const MAX_DATE = "2026-12-31";

const NPM_PACKAGES = [
  { project: "React", packageName: "react" },
  { project: "React DOM", packageName: "react-dom" },
  { project: "Next.js", packageName: "next" },
  { project: "TypeScript", packageName: "typescript" },
  { project: "Vite", packageName: "vite" },
  { project: "Tailwind CSS", packageName: "tailwindcss" },
  { project: "Vue", packageName: "vue" },
  { project: "Svelte", packageName: "svelte" },
  { project: "Angular", packageName: "@angular/core" },
  { project: "Express", packageName: "express" },
  { project: "ESLint", packageName: "eslint" },
  { project: "Prettier", packageName: "prettier" },
  { project: "Webpack", packageName: "webpack" },
  { project: "Rollup", packageName: "rollup" },
  { project: "esbuild", packageName: "esbuild" },
  { project: "Astro", packageName: "astro" },
  { project: "Remix", packageName: "@remix-run/react" },
  { project: "Nuxt", packageName: "nuxt" },
  { project: "Bun types", packageName: "bun-types" },
  { project: "pnpm", packageName: "pnpm" }
];

const PYPI_PACKAGES = [
  { project: "PyTorch", packageName: "torch" },
  { project: "TensorFlow", packageName: "tensorflow" },
  { project: "JAX", packageName: "jax" },
  { project: "Transformers", packageName: "transformers" },
  { project: "vLLM", packageName: "vllm" },
  { project: "LangChain", packageName: "langchain" },
  { project: "LangChain Core", packageName: "langchain-core" },
  { project: "LlamaIndex", packageName: "llama-index" },
  { project: "OpenAI Python", packageName: "openai" },
  { project: "Anthropic Python", packageName: "anthropic" },
  { project: "Django", packageName: "django" },
  { project: "FastAPI", packageName: "fastapi" },
  { project: "Flask", packageName: "flask" },
  { project: "Pydantic", packageName: "pydantic" },
  { project: "SQLAlchemy", packageName: "sqlalchemy" },
  { project: "NumPy", packageName: "numpy" },
  { project: "pandas", packageName: "pandas" },
  { project: "scikit-learn", packageName: "scikit-learn" },
  { project: "SciPy", packageName: "scipy" },
  { project: "Matplotlib", packageName: "matplotlib" },
  { project: "pytest", packageName: "pytest" },
  { project: "Ruff", packageName: "ruff" },
  { project: "Black", packageName: "black" },
  { project: "Poetry", packageName: "poetry" },
  { project: "uv", packageName: "uv" },
  { project: "httpx", packageName: "httpx" },
  { project: "Requests", packageName: "requests" }
];

const NODE_RELEASES_URL = "https://nodejs.org/download/release/index.json";
const PYTHON_RELEASES_URL = "https://www.python.org/downloads/source/";

const GITHUB_PROJECTS = [
  { project: "Rust", repo: "rust-lang/rust", versionPrefix: "" },
  { project: "Kubernetes", repo: "kubernetes/kubernetes", versionPrefix: "" },
  { project: "Docker Compose", repo: "docker/compose", versionPrefix: "" },
  { project: "Deno", repo: "denoland/deno", versionPrefix: "" },
  { project: "Bun", repo: "oven-sh/bun", versionPrefix: "bun-" },
  { project: "Redis", repo: "redis/redis", versionPrefix: "" },
  { project: "PostgreSQL", repo: "postgres/postgres", versionPrefix: "REL_" },
  { project: "Go", repo: "golang/go", versionPrefix: "go" },
  { project: "VS Code", repo: "microsoft/vscode", versionPrefix: "" },
  { project: "Terraform", repo: "hashicorp/terraform", versionPrefix: "" }
];

async function main() {
  const startedAt = new Date().toISOString();
  const [npmRecords, pypiRecords, nodeRecords, pythonRecords, githubRecords] = await Promise.all([
    collectNpmRecords(),
    collectPyPIRecords(),
    collectNodeRecords(),
    collectPythonRecords(),
    collectGitHubRecords()
  ]);

  const releases = [...npmRecords, ...pypiRecords, ...nodeRecords, ...pythonRecords, ...githubRecords]
    .filter((record) => record.date >= MIN_DATE && record.date <= MAX_DATE)
    .filter((record) => isStableVersion(record.version))
    .sort((left, right) => {
      const byDate = left.date.localeCompare(right.date);
      if (byDate !== 0) {
        return byDate;
      }
      return `${left.project} ${left.version}`.localeCompare(`${right.project} ${right.version}`);
    });

  const database = {
    metadata: {
      generatedAt: startedAt,
      minDate: MIN_DATE,
      maxDate: MAX_DATE,
      recordCount: releases.length,
      sources: [
        "https://registry.npmjs.org/",
        "https://pypi.org/pypi/{package}/json",
        NODE_RELEASES_URL,
        PYTHON_RELEASES_URL,
        "https://api.github.com/repos/{owner}/{repo}/releases"
      ],
      schemaVersion: 1
    },
    releases
  };

  await fs.writeFile(OUTPUT_PATH, `${JSON.stringify(database, null, 2)}\n`);
  printSummary(releases);
}

async function collectNpmRecords() {
  const nested = await Promise.all(NPM_PACKAGES.map(async (definition) => {
    const url = `https://registry.npmjs.org/${encodeURIComponent(definition.packageName)}`;
    const registry = await fetchJson(url);
    const versions = registry.versions ?? {};
    const times = registry.time ?? {};

    return Object.keys(versions).flatMap((version) => {
      const publishedAt = times[version];
      if (!publishedAt) {
        return [];
      }

      return [releaseRecord({
        project: definition.project,
        ecosystem: "npm",
        packageName: definition.packageName,
        version,
        date: publishedAt.slice(0, 10),
        source: "npm",
        url: `https://www.npmjs.com/package/${definition.packageName}/v/${version}`
      })];
    });
  }));

  return nested.flat();
}

async function collectPyPIRecords() {
  const nested = await Promise.all(PYPI_PACKAGES.map(async (definition) => {
    const url = `https://pypi.org/pypi/${encodeURIComponent(definition.packageName)}/json`;
    const registry = await fetchJson(url);
    const releases = registry.releases ?? {};

    return Object.entries(releases).flatMap(([version, files]) => {
      if (!Array.isArray(files) || files.length === 0) {
        return [];
      }

      const uploadedAt = files
        .map((file) => file.upload_time_iso_8601 ?? file.upload_time)
        .filter(Boolean)
        .sort()[0];

      if (!uploadedAt) {
        return [];
      }

      return [releaseRecord({
        project: definition.project,
        ecosystem: "pypi",
        packageName: definition.packageName,
        version,
        date: uploadedAt.slice(0, 10),
        source: "pypi",
        url: `https://pypi.org/project/${definition.packageName}/${version}/`
      })];
    });
  }));

  return nested.flat();
}

async function collectNodeRecords() {
  const releases = await fetchJson(NODE_RELEASES_URL);
  return releases.map((release) => releaseRecord({
    project: "Node.js",
    ecosystem: "runtime",
    packageName: "node",
    version: stripVersionPrefix(release.version),
    date: release.date,
    source: "nodejs.org",
    url: `https://nodejs.org/en/blog/release/${release.version}`,
    facts: release.lts ? [`LTS: ${release.lts}`] : []
  }));
}

async function collectPythonRecords() {
  const html = await fetchText(PYTHON_RELEASES_URL);
  const records = [];
  const releaseRe = /<a href="([^"]+)">Python (\d+\.\d+\.\d+) - ([A-Z][a-z]+\.? \d{1,2}, \d{4})<\/a>/gu;

  for (const match of html.matchAll(releaseRe)) {
    const [, href, version, releaseDate] = match;
    records.push(releaseRecord({
      project: "Python",
      ecosystem: "runtime",
      packageName: "python",
      version,
      date: parsePythonDate(releaseDate),
      source: "python.org",
      url: new URL(href, PYTHON_RELEASES_URL).toString()
    }));
  }

  return records;
}

async function collectGitHubRecords() {
  const nested = await Promise.all(GITHUB_PROJECTS.map(async (definition) => {
    const releases = await fetchGitHubReleases(definition.repo);
    return releases.map((release) => {
      const version = normalizeGitHubVersion(release.tag_name, definition.versionPrefix);
      return releaseRecord({
        project: definition.project,
        ecosystem: "github-release",
        packageName: definition.repo,
        version,
        date: release.published_at.slice(0, 10),
        source: "github",
        url: release.html_url,
        facts: release.name && release.name !== release.tag_name ? [release.name] : []
      });
    });
  }));

  return nested.flat();
}

async function fetchGitHubReleases(repo) {
  const records = [];
  let page = 1;

  while (page <= 3) {
    const url = `https://api.github.com/repos/${repo}/releases?per_page=100&page=${page}`;
    const pageRecords = await fetchJson(url, {
      headers: {
        accept: "application/vnd.github+json",
        "user-agent": "modelclock-dataset-builder",
        ...(process.env.GITHUB_TOKEN ? { authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {})
      }
    });

    if (!Array.isArray(pageRecords) || pageRecords.length === 0) {
      break;
    }

    records.push(...pageRecords);

    if (pageRecords.length < 100 || pageRecords.some((release) => release.published_at?.slice(0, 10) < MIN_DATE)) {
      break;
    }

    page += 1;
  }

  return records;
}

function releaseRecord({ project, ecosystem, packageName, version, date, source, url, facts = [] }) {
  return {
    date,
    project,
    version,
    ecosystem,
    package: packageName,
    source,
    url,
    prompt: `Had ${project} released version ${version} yet?`,
    facts
  };
}

async function fetchJson(url, init = {}) {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function fetchText(url, init = {}) {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

function parsePythonDate(value) {
  const normalized = value.replace(/\./gu, "");
  const timestamp = Date.parse(`${normalized} UTC`);
  if (!Number.isFinite(timestamp)) {
    throw new Error(`Unable to parse Python release date: ${value}`);
  }
  return new Date(timestamp).toISOString().slice(0, 10);
}

function normalizeGitHubVersion(tagName, versionPrefix) {
  let version = String(tagName);
  if (versionPrefix && version.startsWith(versionPrefix)) {
    version = version.slice(versionPrefix.length);
  }
  if (versionPrefix === "REL_") {
    version = version.replace(/_/gu, ".");
  }
  return stripVersionPrefix(version);
}

function stripVersionPrefix(version) {
  return String(version).replace(/^v/u, "");
}

function isStableVersion(version) {
  return /^\d+\.\d+\.\d+(?:\.\d+)?$/u.test(version);
}

function printSummary(releases) {
  const byProject = new Map();
  for (const release of releases) {
    byProject.set(release.project, (byProject.get(release.project) ?? 0) + 1);
  }

  console.log(`Wrote ${OUTPUT_PATH}`);
  console.log(`Records: ${releases.length}`);
  console.log(`Range: ${releases[0]?.date ?? "n/a"} to ${releases.at(-1)?.date ?? "n/a"}`);
  console.log("Projects:");
  for (const [project, count] of [...byProject.entries()].sort()) {
    console.log(`  ${project}: ${count}`);
  }
}

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exitCode = 1;
});
