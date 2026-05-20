import fs from "node:fs";
import path from "node:path";

export function loadDotEnv(filename, env) {
  if (!fs.existsSync(filename)) {
    return;
  }

  const lines = fs.readFileSync(filename, "utf8").split(/\r?\n/u);

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const equalsIndex = trimmed.indexOf("=");
    if (equalsIndex === -1) {
      continue;
    }

    const key = trimmed.slice(0, equalsIndex).trim();
    const value = unquote(trimmed.slice(equalsIndex + 1).trim());

    if (key && env[key] === undefined) {
      env[key] = value;
    }
  }
}

export function saveDotEnv(filename, updates) {
  fs.mkdirSync(path.dirname(filename), { recursive: true });
  const existing = fs.existsSync(filename)
    ? fs.readFileSync(filename, "utf8").split(/\r?\n/u)
    : [];
  const lines = existing.length === 1 && existing[0] === "" ? [] : existing;
  const remaining = new Set(Object.keys(updates));
  const nextLines = lines.map((line) => {
    const parsed = parseAssignment(line);
    if (!parsed || !remaining.has(parsed.key)) {
      return line;
    }
    remaining.delete(parsed.key);
    return `${parsed.key}=${quoteDotEnvValue(updates[parsed.key])}`;
  });

  if (nextLines.length > 0 && nextLines.at(-1) !== "" && remaining.size > 0) {
    nextLines.push("");
  }

  for (const key of remaining) {
    nextLines.push(`${key}=${quoteDotEnvValue(updates[key])}`);
  }

  fs.writeFileSync(filename, `${nextLines.join("\n").replace(/\n+$/u, "")}\n`);
}

export function parseCsvEnv(value) {
  if (!value) {
    return [];
  }
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function formatCsvEnv(values) {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))].join(",");
}

function parseAssignment(line) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) {
    return null;
  }
  const equalsIndex = trimmed.indexOf("=");
  if (equalsIndex === -1) {
    return null;
  }
  const key = trimmed.slice(0, equalsIndex).trim();
  return key ? { key } : null;
}

function unquote(value) {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value
      .slice(1, -1)
      .replace(/\\n/gu, "\n")
      .replace(/\\"/gu, '"')
      .replace(/\\\\/gu, "\\");
  }
  return value;
}

function quoteDotEnvValue(value) {
  const text = String(value ?? "");
  if (/^[A-Za-z0-9_@%+=:,./-]*$/u.test(text)) {
    return text;
  }
  return `"${text
    .replace(/\\/gu, "\\\\")
    .replace(/\n/gu, "\\n")
    .replace(/"/gu, '\\"')}"`;
}
