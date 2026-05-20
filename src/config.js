import fs from "node:fs";
import path from "node:path";

import { formatCsvEnv, parseCsvEnv } from "./env.js";

export const CONFIG_VERSION = 1;

export function loadModelClockConfig({ configFile, env }) {
  if (fs.existsSync(configFile)) {
    return normalizeConfig(JSON.parse(fs.readFileSync(configFile, "utf8")));
  }

  return normalizeConfig(configFromEnv(env));
}

export function saveModelClockConfig(configFile, config) {
  fs.mkdirSync(path.dirname(configFile), { recursive: true });
  fs.writeFileSync(configFile, `${JSON.stringify(normalizeConfig(config), null, 2)}\n`);
}

export function clearModelClockConfig({ configFile }) {
  fs.rmSync(configFile, { force: true });
}

export function createEmptyConfig() {
  return {
    version: CONFIG_VERSION,
    providers: [],
    selectedModels: []
  };
}

export function normalizeConfig(config) {
  const { activeProviderId: _ignoredActiveProviderId, ...configWithoutDefaultProvider } =
    config && typeof config === "object" ? config : {};
  const next = {
    ...createEmptyConfig(),
    ...configWithoutDefaultProvider
  };
  next.version = CONFIG_VERSION;
  next.providers = normalizeProviders(next.providers);
  next.selectedModels = normalizeSelectedModels(next.selectedModels, next.providers);
  return next;
}

export function createProvider({ id, name, baseUrl, apiKey = "", models = [], reasoningEffort, enableThinking }) {
  return {
    id: id || uniqueProviderId(name || baseUrl || "provider", []),
    name: name || "OpenAI Compatible",
    baseUrl: baseUrl || "",
    apiKey: apiKey || "",
    models: dedupe(models),
    ...(reasoningEffort ? { reasoningEffort } : {}),
    ...(enableThinking !== undefined ? { enableThinking } : {})
  };
}

export function upsertProvider(config, provider) {
  const normalized = normalizeProvider(provider, config.providers.filter((entry) => entry.id !== provider.id));
  const exists = config.providers.some((entry) => entry.id === normalized.id);
  const providers = exists
    ? config.providers.map((entry) => entry.id === normalized.id ? normalized : entry)
    : [...config.providers, normalized];
  return normalizeConfig({
    ...config,
    providers
  });
}

export function removeProvider(config, providerId) {
  return normalizeConfig({
    ...config,
    providers: config.providers.filter((provider) => provider.id !== providerId),
    selectedModels: config.selectedModels.filter((ref) => ref.providerId !== providerId)
  });
}

export function setSelectedModelsForProvider(config, providerId, models) {
  const retained = config.selectedModels.filter((ref) => ref.providerId !== providerId);
  const selectedModels = [
    ...retained,
    ...dedupe(models).map((model) => ({ providerId, model }))
  ];
  return normalizeConfig({ ...config, selectedModels });
}

export function setSelectedModelRefs(config, refs) {
  return normalizeConfig({
    ...config,
    selectedModels: Array.isArray(refs) ? refs.map((ref) => ({
      providerId: ref.providerId,
      model: ref.model
    })) : []
  });
}

export function providerById(config, providerId) {
  return config.providers.find((provider) => provider.id === providerId) ?? null;
}

export function providerLabel(provider) {
  return provider?.name || provider?.baseUrl || provider?.id || "Provider";
}

export function modelRefKey(ref) {
  return `${ref.providerId}::${ref.model}`;
}

export function formatModelRef(config, ref) {
  const provider = providerById(config, ref.providerId);
  return `${providerLabel(provider)} · ${ref.model}`;
}

function configFromEnv(env) {
  const models = dedupe([
    env.OPENAI_MODEL,
    ...parseCsvEnv(env.OPENAI_MODELS)
  ].filter(Boolean));
  const selected = parseCsvEnv(env.MODEL_CLOCK_SELECTED_MODELS).filter((model) => models.includes(model));
  const hasProvider = env.OPENAI_BASE_URL || env.OPENAI_API_KEY || models.length > 0;
  const config = createEmptyConfig();

  if (!hasProvider) {
    return config;
  }

  const provider = createProvider({
    id: providerIdFromBaseUrl(env.OPENAI_BASE_URL) || "default",
    name: providerNameFromBaseUrl(env.OPENAI_BASE_URL),
    baseUrl: env.OPENAI_BASE_URL || "",
    apiKey: env.OPENAI_API_KEY || "",
    models,
    reasoningEffort: env.MODEL_CLOCK_REASONING_EFFORT,
    enableThinking: parseOptionalBoolean(env.MODEL_CLOCK_ENABLE_THINKING)
  });

  return normalizeConfig({
    ...config,
    providers: [provider],
    selectedModels: (selected.length > 0 ? selected : models).map((model) => ({
      providerId: provider.id,
      model
    }))
  });
}

function normalizeProviders(providers) {
  const used = new Set();
  return Array.isArray(providers)
    ? providers.map((provider) => normalizeProvider(provider, [...used])).map((provider) => {
        used.add(provider.id);
        return provider;
      })
    : [];
}

function normalizeProvider(provider, existingProviders = []) {
  const existingIds = new Set(existingProviders.map((entry) => typeof entry === "string" ? entry : entry.id).filter(Boolean));
  const name = normalizeProviderName(String(provider?.name || providerNameFromBaseUrl(provider?.baseUrl) || "OpenAI Compatible"));
  const requestedId = normalizeProviderId(provider?.id);
  const id = requestedId && !existingIds.has(requestedId)
    ? requestedId
    : uniqueProviderId(requestedId || name, [...existingIds]);

  return {
    id,
    name,
    baseUrl: String(provider?.baseUrl || ""),
    apiKey: String(provider?.apiKey || ""),
    models: dedupe(provider?.models ?? []),
    ...(provider?.reasoningEffort ? { reasoningEffort: String(provider.reasoningEffort) } : {}),
    ...(provider?.enableThinking !== undefined ? { enableThinking: Boolean(provider.enableThinking) } : {})
  };
}

function normalizeProviderName(name) {
  if (name === "Hugging Face Router") {
    return "Hugging Face";
  }
  if (name === "OpenAI") {
    return "OpenAI Compatible";
  }
  return name;
}

function normalizeProviderId(id) {
  if (id === "hugging-face-router") {
    return "hugging-face";
  }
  if (id === "openai") {
    return "openai-compatible";
  }
  return id ? String(id) : "";
}

function normalizeSelectedModels(selectedModels, providers) {
  const providerIdAliases = new Map(providers.map((provider) => [provider.id, provider.id]));
  if (providerIdAliases.has("hugging-face")) {
    providerIdAliases.set("hugging-face-router", "hugging-face");
  }
  if (providerIdAliases.has("openai-compatible")) {
    providerIdAliases.set("openai", "openai-compatible");
  }
  const providerIds = new Set(providers.map((provider) => provider.id));
  const providerModels = new Map(providers.map((provider) => [provider.id, new Set(provider.models)]));
  const seen = new Set();
  const refs = Array.isArray(selectedModels) ? selectedModels : [];

  return refs
    .map((ref) => typeof ref === "string" ? null : {
      providerId: providerIdAliases.get(ref?.providerId) ?? ref?.providerId,
      model: ref?.model
    })
    .filter((ref) => ref?.providerId && ref?.model)
    .filter((ref) => providerIds.has(ref.providerId) && providerModels.get(ref.providerId)?.has(ref.model))
    .filter((ref) => {
      const key = modelRefKey(ref);
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}

function providerNameFromBaseUrl(baseUrl = "") {
  if (baseUrl.includes("api.openai.com")) {
    return "OpenAI Compatible";
  }
  if (baseUrl.includes("router.huggingface.co")) {
    return "Hugging Face";
  }
  if (baseUrl.includes("openrouter.ai")) {
    return "OpenRouter";
  }
  return "OpenAI Compatible";
}

function providerIdFromBaseUrl(baseUrl = "") {
  if (baseUrl.includes("api.openai.com")) {
    return "openai-compatible";
  }
  if (baseUrl.includes("router.huggingface.co")) {
    return "hugging-face";
  }
  if (baseUrl.includes("openrouter.ai")) {
    return "openrouter";
  }
  return "";
}

function uniqueProviderId(value, existingIds) {
  const existing = new Set(existingIds);
  const base = slugify(value || "provider") || "provider";
  let id = base;
  let suffix = 2;
  while (existing.has(id)) {
    id = `${base}-${suffix}`;
    suffix += 1;
  }
  return id;
}

function slugify(value) {
  return String(value)
    .toLowerCase()
    .replace(/https?:\/\//gu, "")
    .replace(/[^a-z0-9]+/gu, "-")
    .replace(/^-|-$/gu, "");
}

function dedupe(values) {
  return parseCsvEnv(formatCsvEnv(Array.isArray(values) ? values : []));
}

function parseOptionalBoolean(value) {
  if (value === undefined || value === "") {
    return undefined;
  }
  if (value === true || value === "true") {
    return true;
  }
  if (value === false || value === "false") {
    return false;
  }
  return undefined;
}
