import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import React, { useMemo, useState } from "react";
import { Box, Text, render, useApp, useInput } from "ink";
import figlet from "figlet";
import ansiShadowFont from "figlet/fonts/ANSI Shadow";
import bigFont from "figlet/fonts/Big";
import miniFont from "figlet/fonts/Mini";
import smallFont from "figlet/fonts/Small";
import standardFont from "figlet/fonts/Standard";

import { createOpenAIClient } from "./openai-compatible.js";
import {
  clearModelClockConfig,
  createEmptyConfig,
  createProvider,
  formatModelRef,
  loadModelClockConfig,
  modelRefKey,
  providerById,
  providerLabel,
  removeProvider,
  saveModelClockConfig,
  setSelectedModelRefs,
  setSelectedModelsForProvider,
  upsertProvider
} from "./config.js";
import { loadDataset } from "./dataset.js";
import { formatCsvEnv, loadDotEnv, parseCsvEnv } from "./env.js";
import { runBoundarySearch } from "./search.js";

const h = React.createElement;

figlet.parseFont("ANSI Shadow", ansiShadowFont);
figlet.parseFont("Big", bigFont);
figlet.parseFont("Mini", miniFont);
figlet.parseFont("Small", smallFont);
figlet.parseFont("Standard", standardFont);

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_DIR = process.env.MODEL_CLOCK_CONFIG_DIR ?? path.join(os.homedir(), ".modelclock");
const CONFIG_FILE = process.env.MODEL_CLOCK_CONFIG_FILE ?? path.join(CONFIG_DIR, "config.json");
const LOCAL_ENV_FILE = ".env";
const PROVIDER_PRESETS = [
  { id: "openrouter", name: "OpenRouter", value: "https://openrouter.ai/api/v1" },
  { id: "hugging-face", name: "Hugging Face", value: "https://router.huggingface.co/v1" },
  { id: "openai-compatible", name: "OpenAI Compatible", value: "custom" }
];
const REMOVED_OPTIONS = new Set([
  "dataset",
  "include-metadata",
  "max-tokens",
  "min-items",
  "probe-mode",
  "sample-size",
  "temperature",
  "tolerance-days",
  "trials",
  "id-match-threshold",
  "cloze-blanks-per-title",
  "cloze-match-threshold",
  "software-match-threshold",
  "software-max-false-positives"
]);
const SAVED_CONFIG_KEYS = [
  "OPENAI_BASE_URL",
  "OPENAI_API_KEY",
  "OPENAI_MODEL",
  "OPENAI_MODELS",
  "MODEL_CLOCK_SELECTED_MODELS",
  "MODEL_CLOCK_REASONING_EFFORT",
  "MODEL_CLOCK_ENABLE_THINKING"
];

const DEFAULTS = {
  dataDir: resolveDefaultDataDir(),
  dataset: "software-releases",
  sampleSize: 10,
  minItems: 5,
  maxSteps: 30,
  trials: 1,
  temperature: 0,
  maxTokens: 2048,
  probeMode: "software-version",
  replicates: 3,
  probeDatesPerRound: 3,
  softwareItemsPerProbeDate: 5,
  softwareDecoysPerProbeDate: 3,
  credibleIntervalDays: 14,
  minRounds: 12,
  noisyProbeRealWeight: 0.25,
  softwareMatchThreshold: 3,
  softwareMaxFalsePositives: 1,
  dryRun: false,
  json: false
};

const color = createColorTheme();
const inkColor = {
  blue: "#2563eb",
  blueBright: "#38bdf8",
  yellow: "#facc15"
};
const TAGLINE = "Probe LLM knowledge cutoffs";

export async function main(argv = process.argv.slice(2), env = process.env) {
  loadDotEnv(LOCAL_ENV_FILE, env);
  const config = loadModelClockConfig({ configFile: CONFIG_FILE, env });

  const args = parseArgs(argv);
  if (argv.length === 0 && process.stdin.isTTY && process.stdout.isTTY) {
    args.command = "interactive";
  }

  if (args.help) {
    console.log(helpText());
    return;
  }

  if (!args.json && args.command !== "reset" && args.command !== "interactive" && args.command !== "settings") {
    printBanner();
  }

  if (args.command === "reset") {
    await removeAllSettings(env, config, { force: args.yes });
    return;
  }

  if (args.command === "settings") {
    clearTerminal();
    const settingsResult = await runInkInteractive(args, env, { config, initialScreen: "settings" });
    Object.assign(args, settingsResult);
    if (args.command === "exit") {
      return;
    }
    return;
  }

  if (args.command === "interactive") {
    clearTerminal();
    Object.assign(args, await readInteractiveOptions(args, env, config));
    if (args.command === "exit") {
      return;
    }
  }

  if (!args.dryRun && args.command !== "inspect" && !args.inspect) {
    await ensureApiConfig({ args, env, config });
  }

  const options = buildOptions(args, env, config);
  const dataset = loadDataset(options);

  if (args.inspect) {
    printInspect(dataset, options);
    return;
  }

  const runResults = await runModelSet({ dataset, options, env });

  if (options.json) {
    const payload = runResults.length === 1 ? runResults[0].result : runResults;
    console.log(JSON.stringify(payload, null, 2));
    return;
  }
}

function parseArgs(argv) {
  const args = {
    command: "run"
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (token === "--help" || token === "-h") {
      args.help = true;
      continue;
    }

    if (token === "inspect" || token === "--inspect") {
      args.inspect = true;
      args.command = "inspect";
      continue;
    }

    if (token === "suite" || token === "modes") {
      throw new Error(`${token} has been removed; modelclock now runs the software release probe only.`);
    }

    if (token === "run" || token === "interactive" || token === "settings" || token === "reset") {
      args.command = token;
      continue;
    }

    if (token === "--yes" || token === "-y") {
      args.yes = true;
      continue;
    }

    if (token === "--dry-run") {
      args.dryRun = true;
      continue;
    }

    if (token === "--json") {
      args.json = true;
      continue;
    }

    if (!token.startsWith("--")) {
      throw new Error(`Unknown argument: ${token}`);
    }

    const [rawKey, inlineValue] = token.slice(2).split("=", 2);
    if (rawKey === "max-steps") {
      throw new Error("--max-steps is no longer configurable; modelclock uses an internal 30-round safety cap.");
    }
    if (REMOVED_OPTIONS.has(rawKey)) {
      throw new Error(`--${rawKey} is no longer configurable; modelclock uses the software release probe defaults.`);
    }
    const key = toCamelCase(rawKey);
    const value = inlineValue ?? argv[index + 1];

    if (value === undefined || value.startsWith("--")) {
      throw new Error(`Missing value for --${rawKey}`);
    }

    if (inlineValue === undefined) {
      index += 1;
    }

    args[key] = value;
  }

  return args;
}

async function runModelSet({ dataset, options, env }) {
  const modelRefs = options.dryRun && !options.modelsExplicit && options.modelRefs.length === 0
    ? [{ providerId: "dry-run", providerName: "Dry run", model: "dry-run" }]
    : options.modelRefs;
  const results = [];

  if (!options.dryRun && modelRefs.length === 0) {
    throw new Error("At least one model is required.");
  }

  for (let index = 0; index < modelRefs.length; index += 1) {
    const ref = modelRefs[index];
    const modelOptions = {
      ...options,
      provider: ref.provider,
      providerName: ref.providerName,
      model: ref.model,
      modelDisplay: ref.displayName ?? ref.model,
      modelRef: ref,
      models: [ref.model],
      modelRefs: [ref]
    };
    const client = createClientForOptions(modelOptions, env);

    if (!modelOptions.json) {
      if (modelRefs.length > 1) {
        console.log("");
        console.log(color.dim(`Model ${index + 1}/${modelRefs.length}`));
      }
      printRunHeader(modelOptions);
    }

    const result = await runBoundarySearch({
      dataset,
      client,
      options: modelOptions,
      onStep: modelOptions.json ? null : printStepProgress
    });
    results.push({ model: modelOptions.modelDisplay, options: publicOptions(modelOptions), result });

    if (!modelOptions.json) {
      printResult(result, modelOptions);
    }
  }

  if (!options.json && results.length > 1) {
    printResultsSummary(results);
  }

  return results;
}

function createClientForOptions(options, env) {
  return options.dryRun
    ? createDryRunClient(options)
    : createOpenAIClient({
        baseUrl: options.provider.baseUrl,
        apiKey: options.provider.apiKey,
        model: options.model,
        temperature: options.temperature,
        maxTokens: options.maxTokens,
        reasoningEffort: options.provider.reasoningEffort,
        enableThinking: options.provider.enableThinking
      });
}

function publicOptions(options) {
  const copy = { ...options };
  delete copy.modelsExplicit;
  if (copy.provider) {
    copy.provider = {
      ...copy.provider,
      apiKey: copy.provider.apiKey ? maskSecret(copy.provider.apiKey) : ""
    };
  }
  return copy;
}

async function readInteractiveOptions(args, env, config) {
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    throw new Error("Interactive mode requires a TTY. Use explicit flags for non-interactive runs.");
  }

  return runInkInteractive(args, env, { config });
}

async function runInkInteractive(args, env, uiOptions = {}) {
  let result = { command: "exit" };
  const app = render(h(InteractiveApp, {
    args,
    env,
    initialConfig: uiOptions.config,
    initialScreen: uiOptions.initialScreen,
    resetOnly: Boolean(uiOptions.resetOnly),
    setupOnly: Boolean(uiOptions.setupOnly),
    onDone(nextResult) {
      result = nextResult;
    }
  }), { exitOnCtrlC: true });

  await app.waitUntilExit();
  return result;
}

function InteractiveApp({ args, env, initialConfig, initialScreen, resetOnly, setupOnly, onDone }) {
  const app = useApp();
  const [configState, setConfigState] = useState(initialConfig);
  const initialState = useMemo(() => {
    const state = defaultInteractiveArgs(args, env, initialConfig);
    state.modelRefs = initialSelectedModelRefs(state, initialConfig);
    state.providerId = state.modelRefs[0]?.providerId ?? initialConfig.providers[0]?.id ?? "";
    return state;
  }, [args, env, initialConfig]);
  const [state, setState] = useState(initialState);
  const [notice, setNotice] = useState("");
  const [stack, setStack] = useState(() => {
    if (initialScreen) {
      return [normalizeScreen(initialScreen)];
    }
    const setupScreen = missingSetupForRefs(initialConfig, initialState.modelRefs ?? []) ?? missingSetupScreen(initialConfig);
    if (!hasCompleteApiConfig(initialState, env, initialConfig)) {
      return [setupScreen ?? { name: "confirmRun" }];
    }
    return [setupOnly ? { name: "confirmRun" } : { name: "main" }];
  });

  const screen = stack.at(-1) ?? { name: "main" };
  const selectedRefs = selectedModelRefsForState(state, configState);
  const currentProvider = providerById(configState, screen.providerId ?? state.providerId ?? selectedRefs[0]?.providerId ?? configState.providers[0]?.id);

  const finish = (nextResult) => {
    onDone(nextResult);
    app.exit();
  };
  const go = (nextScreen) => {
    setNotice("");
    setStack((current) => [...current, normalizeScreen(nextScreen)]);
  };
  const replace = (nextScreen) => {
    setStack((current) => [...current.slice(0, -1), normalizeScreen(nextScreen)]);
  };
  const back = () => {
    if (stack.length <= 1) {
      finish({ command: "exit" });
      return;
    }
    setStack((current) => current.slice(0, -1));
  };
  const returnToProviderList = () => {
    setStack((current) => {
      const providerListIndex = current.findLastIndex((entry) => entry.name === "providers");
      if (providerListIndex >= 0) {
        return current.slice(0, providerListIndex + 1);
      }
      return [{ name: "providers" }];
    });
  };
  const persistConfig = (nextConfig) => {
    saveModelClockConfig(CONFIG_FILE, nextConfig);
    setConfigState(nextConfig);
    return nextConfig;
  };
  const advanceSetup = (nextConfig = configState) => {
    const next = missingSetupScreen(nextConfig);
    replace(next ?? { name: "confirmRun" });
  };
  const saveProvider = (providerPatch, activeScreen = screen) => {
    const providerId = providerPatch.id ?? activeScreen.providerId ?? currentProvider?.id;
    const existing = providerById(configState, providerId) ?? {};
    const nextConfig = persistConfig(upsertProvider(configState, {
      ...existing,
      ...providerPatch,
      id: providerId
    }));
    const savedProvider = providerById(nextConfig, providerId) ?? nextConfig.providers.at(-1);
    setState((current) => normalizeInteractiveState({
      ...current,
      providerId: savedProvider?.id ?? current.providerId
    }, nextConfig));
    return { nextConfig, savedProvider };
  };
  const createProviderFromPreset = (preset, activeScreen = screen) => {
    if (preset.value === "custom") {
      go({ name: "providerCustomEndpoint", setup: activeScreen.setup });
      return;
    }
    const nextConfig = persistConfig(upsertProvider(configState, createProvider({
      id: preset.id,
      name: preset.name,
      baseUrl: preset.value
    })));
    const savedProvider = providerById(nextConfig, preset.id) ?? nextConfig.providers.at(-1);
    setState((current) => normalizeInteractiveState({ ...current, providerId: savedProvider.id }, nextConfig));
    setNotice(`Saved ${savedProvider.name} to ${CONFIG_FILE}.`);
    go({ name: "apiKeyInput", providerId: savedProvider.id, setup: activeScreen.setup });
  };
  const saveEndpoint = (baseUrl, activeScreen = screen) => {
    const { nextConfig, savedProvider } = saveProvider({ baseUrl }, activeScreen);
    setNotice(`Saved provider endpoint to ${CONFIG_FILE}.`);
    if (activeScreen.setup) {
      advanceSetup(nextConfig);
    } else {
      replace({ name: "providerDetail", providerId: savedProvider.id });
    }
  };
  const saveApiKey = (apiKey, activeScreen = screen) => {
    const { nextConfig } = saveProvider({ apiKey }, activeScreen);
    setNotice(`Saved API key to ${CONFIG_FILE}.`);
    if (activeScreen.setup) {
      advanceSetup(nextConfig);
    } else {
      back();
    }
  };
  const saveModelList = (models, activeScreen = screen) => {
    const providerId = activeScreen.providerId ?? currentProvider?.id;
    const provider = providerById(configState, providerId);
    if (!provider) {
      setNotice("Choose a provider before editing models.");
      return;
    }
    const nextProvider = { ...provider, models };
    let nextConfig = upsertProvider(configState, nextProvider);
    const selectedForProvider = selectedRefs
      .filter((ref) => ref.providerId === providerId)
      .map((ref) => ref.model)
      .filter((model) => models.includes(model));
    nextConfig = setSelectedModelsForProvider(nextConfig, providerId, selectedForProvider.length > 0 ? selectedForProvider : models);
    persistConfig(nextConfig);
    const refs = selectedModelRefsForProvider(nextConfig, providerId);
    setState((current) => normalizeInteractiveState({ ...current, providerId, modelRefs: refs }, nextConfig));
    setNotice(`Saved ${models.length} model(s) to ${CONFIG_FILE}.`);
    if (activeScreen.setup) {
      advanceSetup(nextConfig);
    } else {
      back();
    }
  };
  const saveGlobalModelSelection = (refs, activeScreen = screen) => {
    const nextConfig = persistConfig(setSelectedModelRefs(configState, refs));
    const nextState = normalizeInteractiveState({
      ...state,
      providerId: refs[0]?.providerId ?? state.providerId,
      modelRefs: refs
    }, nextConfig);
    setState(nextState);
    setNotice(`Saved ${refs.length} selected model(s).`);
    if (activeScreen.probe) {
      go({ name: "confirmRun" });
    } else {
      back();
    }
  };
  const startProbe = () => {
    const selectedRunRefs = selectedModelRefsForState(state, configState);
    const missing = missingSetupForRefs(configState, selectedRunRefs) ?? missingSetupScreen(configState);
    if (missing) {
      go(missing);
      return;
    }
    const nextState = normalizeInteractiveState({ ...state, modelRefs: selectedRunRefs }, configState);
    if (nextState.modelRefs.length === 0) {
      setNotice("Select at least one model before starting.");
      go({ name: "modelChooseAll", probe: true });
      return;
    }
    finish(materializeInteractiveRun(nextState));
  };

  let content;
  if (screen.name === "main") {
    content = h(MenuScreen, {
      title: "Start",
      caption: "Arrow keys move, Enter selects, Esc goes back.",
      items: [
        { label: "Start", description: "Estimate LLM knowledge cutoffs", value: "start" },
        { label: "Settings", description: "Providers, API keys, model lists, and reset", value: "settings" },
        { label: "Exit", value: "exit" }
      ],
      onBack: () => finish({ command: "exit" }),
      onSelect: (item) => {
        if (item.value === "start") {
          go(configState.providers.length > 0 && allSavedModelRefs(configState).length > 0
            ? { name: "modelChooseAll", probe: true }
            : (missingSetupScreen(configState) ?? { name: "providerAddPreset", setup: true }));
        } else if (item.value === "settings") {
          go({ name: "settings" });
        } else {
          finish({ command: "exit" });
        }
      }
    });
  } else if (screen.name === "settings") {
    content = h(MenuScreen, {
      title: "Settings",
      caption: "Manage saved OpenRouter, Hugging Face, and OpenAI Compatible providers.",
      items: [
        { label: "Providers", description: `${configState.providers.length} saved`, value: "providers" },
        { label: "Remove all settings", description: `Delete ${CONFIG_FILE}`, value: "reset" },
        { label: "Back", value: "back" }
      ],
      onBack: back,
      onSelect: (item) => {
        if (item.value === "providers") {
          go({ name: "providers" });
        } else if (item.value === "reset") {
          go({ name: "resetConfirm" });
        } else {
          back();
        }
      }
    });
  } else if (screen.name === "providers") {
    content = h(MenuScreen, {
      title: "Providers",
      caption: "Choose a provider to edit, or add a supported provider.",
      items: [
        ...configState.providers.map((provider) => ({
          label: providerLabel(provider),
          description: `${provider.models.length} model(s) · ${provider.baseUrl || "no endpoint"}`,
          value: provider.id
        })),
        { label: "Add provider", value: "add" },
        { label: "Back", value: "back" }
      ],
      onBack: back,
      onSelect: (item) => {
        if (item.value === "add") {
          go({ name: "providerAddPreset" });
        } else if (item.value === "back") {
          back();
        } else {
          setState((current) => ({ ...current, providerId: item.value }));
          go({ name: "providerDetail", providerId: item.value });
        }
      }
    });
  } else if (screen.name === "providerDetail") {
    const provider = providerById(configState, screen.providerId);
    content = h(MenuScreen, {
      title: providerLabel(provider),
      caption: provider?.baseUrl || "No endpoint saved.",
      items: [
        { label: "Edit endpoint", description: provider?.baseUrl || "not set", value: "endpoint" },
        { label: "Edit API key", description: provider?.apiKey ? maskSecret(provider.apiKey) : "not set", value: "key" },
        { label: "Edit Models", description: `${provider?.models.length ?? 0} saved`, value: "models" },
        { label: "Delete provider", value: "delete" },
        { label: "Back", value: "back" }
      ],
      onBack: back,
      onSelect: (item) => {
        if (item.value === "endpoint") {
          go({ name: "endpointSelect", providerId: provider.id });
        } else if (item.value === "key") {
          go({ name: "apiKeyInput", providerId: provider.id });
        } else if (item.value === "models") {
          go({ name: "modelReplace", providerId: provider.id });
        } else if (item.value === "delete") {
          go({ name: "providerDeleteConfirm", providerId: provider.id });
        } else {
          back();
        }
      }
    });
  } else if (screen.name === "providerAddPreset") {
    content = h(MenuScreen, {
      title: screen.setup ? "Setup: Provider" : "Add Provider",
      caption: "Choose a provider type.",
      items: [
        ...PROVIDER_PRESETS.map((preset) => ({
          label: preset.name,
          description: preset.value === "custom" ? "Enter a URL manually" : preset.value,
          value: preset.id,
          preset
        })),
        { label: "Back", value: "back" }
      ],
      onBack: back,
      onSelect: (item) => {
        if (item.value === "back") {
          back();
        } else {
          createProviderFromPreset(item.preset, screen);
        }
      }
    });
  } else if (screen.name === "endpointSelect") {
    content = h(MenuScreen, {
      title: screen.setup ? "Setup: Provider Endpoint" : "Provider Endpoint",
      caption: "Choose a provider endpoint.",
      items: PROVIDER_PRESETS.map((preset) => ({
        label: preset.name,
        description: preset.value === "custom" ? "Enter a URL manually" : preset.value,
        value: preset.value
      })),
      onBack: back,
      onSelect: (item) => {
        if (item.value === "custom") {
          go({ name: "endpointCustom", providerId: screen.providerId, setup: screen.setup });
        } else {
          saveEndpoint(item.value, screen);
        }
      }
    });
  } else if (screen.name === "providerCustomEndpoint") {
    content = h(TextInputScreen, {
      title: "OpenAI Compatible",
      caption: "Enter an OpenAI Compatible base URL.",
      initialValue: "https://api.openai.com/v1",
      validate: (value) => value.trim().length > 0 ? null : "Endpoint is required.",
      onBack: back,
      onSubmit: (value) => {
        const nextConfig = persistConfig(upsertProvider(configState, createProvider({
          id: "openai-compatible",
          name: "OpenAI Compatible",
          baseUrl: value.trim()
        })));
        const savedProvider = nextConfig.providers.at(-1);
        setState((current) => normalizeInteractiveState({ ...current, providerId: savedProvider.id }, nextConfig));
        setNotice(`Saved ${savedProvider.name} to ${CONFIG_FILE}.`);
        go({ name: "apiKeyInput", providerId: savedProvider.id, setup: screen.setup });
      }
    });
  } else if (screen.name === "endpointCustom") {
    content = h(TextInputScreen, {
      title: "Custom Endpoint",
      caption: "Example: https://api.openai.com/v1",
      initialValue: currentProvider?.baseUrl && currentProvider.baseUrl !== "custom" ? currentProvider.baseUrl : "https://api.openai.com/v1",
      validate: (value) => value.trim().length > 0 ? null : "Endpoint is required.",
      onBack: back,
      onSubmit: (value) => saveEndpoint(value.trim(), screen)
    });
  } else if (screen.name === "apiKeyInput") {
    content = h(TextInputScreen, {
      title: screen.setup ? "Setup: API Key" : "API Key",
      caption: currentProvider?.apiKey ? `Current key: ${maskSecret(currentProvider.apiKey)}` : "Stored in your global modelclock config.",
      password: true,
      validate: (value) => value.trim().length > 0 ? null : "API key is required.",
      onBack: back,
      onSubmit: (value) => saveApiKey(value.trim(), screen)
    });
  } else if (screen.name === "modelReplace" || screen.name === "modelListInput") {
    const provider = providerById(configState, screen.providerId ?? state.providerId ?? configState.providers[0]?.id);
    content = h(TextInputScreen, {
      title: screen.setup || screen.name === "modelListInput" ? "Setup: Models" : "Edit Models",
      caption: "Enter one or more model names separated by commas.",
      initialValue: formatCsvEnv(provider?.models ?? []),
      validate: (value) => parseCsvEnv(value).length > 0 ? null : "Enter at least one model name.",
      onBack: back,
      onSubmit: (value) => saveModelList(parseCsvEnv(value), { ...screen, providerId: provider.id })
    });
  } else if (screen.name === "modelChooseAll") {
    const modelRefs = allSavedModelRefs(configState);
    const selectedKeys = selectedModelRefsForState(state, configState).map(modelRefKey);
    const refsByKey = new Map(modelRefs.map((ref) => [modelRefKey(ref), ref]));
    content = h(CheckboxScreen, {
      title: "Models To Test",
      caption: "Space toggles a model. Enter saves the selected run set.",
      items: modelRefs.map((ref) => ({
        label: formatModelRef(configState, ref),
        value: modelRefKey(ref)
      })),
      selectedValues: selectedKeys,
      emptyMessage: "No saved models yet. Add a provider and model list first.",
      onBack: back,
      onSubmit: (values) => {
        const refs = values.map((value) => refsByKey.get(value)).filter(Boolean);
        saveGlobalModelSelection(refs, screen);
      }
    });
  } else if (screen.name === "providerDeleteConfirm") {
    const provider = providerById(configState, screen.providerId);
    content = h(ConfirmScreen, {
      title: "Delete Provider",
      caption: `Remove ${providerLabel(provider)} and its saved models.`,
      confirmLabel: "Delete provider",
      destructive: true,
      onBack: back,
      onCancel: back,
      onConfirm: () => {
        const nextConfig = persistConfig(removeProvider(configState, screen.providerId));
        setState((current) => normalizeInteractiveState({ ...current, providerId: nextConfig.providers[0]?.id ?? "" }, nextConfig));
        setNotice(`Deleted ${providerLabel(provider)}.`);
        returnToProviderList();
      }
    });
  } else if (screen.name === "resetConfirm") {
    content = h(ConfirmScreen, {
      title: "Remove All Settings",
      caption: `This deletes ${CONFIG_FILE}, including the saved API key.`,
      confirmLabel: "Remove settings",
      destructive: true,
      onBack: back,
      onCancel: back,
      onConfirm: () => {
        const emptyConfig = clearSavedSettings(env);
        setConfigState(emptyConfig);
        if (resetOnly) {
          finish({ command: "exit", reset: true });
          return;
        }
        setState(normalizeInteractiveState(defaultInteractiveArgs(args, env, emptyConfig), emptyConfig));
        setNotice("Removed saved modelclock settings.");
        replace({ name: "settings" });
      }
    });
  } else if (screen.name === "confirmRun") {
    content = h(ConfirmScreen, {
      title: "Ready To Run",
      caption: formatInkRunSummary(state, configState),
      confirmLabel: "Start Probe",
      onBack: back,
      onCancel: setupOnly ? () => finish({ command: "exit" }) : back,
      onConfirm: startProbe
    });
  } else {
    content = h(Text, null, `Unknown screen: ${screen.name}`);
  }

  return h(InkFrame, { notice },
    h(Box, { key: screenKey(screen, stack.length), flexDirection: "column" }, content)
  );
}

function normalizeScreen(screen) {
  return typeof screen === "string" ? { name: screen } : screen;
}

function screenKey(screen, depth) {
  return [
    depth,
    screen.name,
    screen.providerId ?? "",
    screen.setup ? "setup" : "",
    screen.probe ? "probe" : ""
  ].join(":");
}

function missingSetupScreen(config) {
  if (config.providers.length === 0) {
    return { name: "providerAddPreset", setup: true };
  }
  const provider = config.providers[0];
  if (!provider.baseUrl) {
    return { name: "endpointSelect", providerId: provider.id, setup: true };
  }
  if (!provider.apiKey) {
    return { name: "apiKeyInput", providerId: provider.id, setup: true };
  }
  if (provider.models.length === 0) {
    return { name: "modelListInput", providerId: provider.id, setup: true };
  }
  return null;
}

function selectedModelRefsForState(state, config) {
  const refs = Array.isArray(state.modelRefs) ? state.modelRefs : [];
  const validKeys = new Set(config.providers.flatMap((provider) => provider.models.map((model) => modelRefKey({
    providerId: provider.id,
    model
  }))));
  const selected = refs.filter((ref) => validKeys.has(modelRefKey(ref)));
  return selected.length > 0 ? selected : initialSelectedModelRefs(state, config);
}

function selectedModelRefsForProvider(config, providerId) {
  const provider = providerById(config, providerId);
  if (!provider) {
    return [];
  }
  const selected = config.selectedModels.filter((ref) => ref.providerId === providerId);
  if (selected.length > 0) {
    return selected;
  }
  return provider.models.map((model) => ({ providerId, model }));
}

function allSavedModelRefs(config) {
  return config.providers.flatMap((provider) => provider.models.map((model) => ({
    providerId: provider.id,
    model
  })));
}

function missingSetupForRefs(config, refs) {
  if (refs.length === 0) {
    return allSavedModelRefs(config).length > 0
      ? { name: "modelChooseAll", probe: true }
      : missingSetupScreen(config);
  }
  for (const ref of refs) {
    const provider = providerById(config, ref.providerId);
    if (!provider) {
      continue;
    }
    if (!provider.baseUrl) {
      return { name: "endpointSelect", providerId: provider.id, setup: true };
    }
    if (!provider.apiKey) {
      return { name: "apiKeyInput", providerId: provider.id, setup: true };
    }
  }
  return null;
}

function normalizeInteractiveState(state, config) {
  const modelRefs = selectedModelRefsForState(state, config);
  const providerId = state.providerId || modelRefs[0]?.providerId || config.providers[0]?.id || "";
  return {
    ...state,
    providerId,
    modelRefs,
    model: modelRefs[0]?.model ?? "",
    models: modelRefs.map((ref) => ref.model)
  };
}

function InkFrame({ notice, children }) {
  const maxWidth = Math.max(Math.min(process.stdout.columns ?? 100, 116), 44);
  const title = banner("MODEL CLOCK", Math.max(maxWidth - 8, 24));

  return h(Box, { flexDirection: "column", width: maxWidth },
    h(Box, { flexDirection: "column", alignItems: "center", marginTop: 2, marginBottom: 1, width: maxWidth },
      h(InkBanner, { text: title }),
      h(Text, { color: inkColor.yellow }, TAGLINE)
    ),
    h(Box, { borderStyle: "round", borderColor: inkColor.yellow, paddingX: 1, flexDirection: "column" },
      children
    ),
    notice ? h(Box, { marginTop: 1 },
      h(Text, { color: inkColor.blueBright }, notice)
    ) : null
  );
}

function uniqueProvidersForRefs(config, refs) {
  const seen = new Set();
  return refs
    .map((ref) => providerById(config, ref.providerId))
    .filter(Boolean)
    .filter((provider) => {
      if (seen.has(provider.id)) {
        return false;
      }
      seen.add(provider.id);
      return true;
    });
}

function InkBanner({ text }) {
  const lines = text.split("\n");
  const denominator = Math.max(lines.length - 1, 1);
  return h(Box, { flexDirection: "column", alignItems: "center" },
    ...lines.map((line, index) => h(Text, {
      key: `${index}:${line}`,
      color: mixHexColor(inkColor.blue, inkColor.yellow, index / denominator)
    }, line))
  );
}

function mixHexColor(fromHex, toHex, ratio) {
  const from = hexToRgb(fromHex);
  const to = hexToRgb(toHex);
  const bounded = Math.min(Math.max(ratio, 0), 1);
  const channel = (start, end) => Math.round(start + (end - start) * bounded)
    .toString(16)
    .padStart(2, "0");
  return `#${channel(from.red, to.red)}${channel(from.green, to.green)}${channel(from.blue, to.blue)}`;
}

function MenuScreen({ title, caption, items, onSelect, onBack }) {
  const [index, setIndex] = useState(0);
  useInput((input, key) => {
    if (key.escape) {
      onBack();
      return;
    }
    if (key.upArrow || input === "k") {
      setIndex((current) => (current - 1 + items.length) % items.length);
      return;
    }
    if (key.downArrow || input === "j") {
      setIndex((current) => (current + 1) % items.length);
      return;
    }
    if (key.return) {
      onSelect(items[index]);
    }
  });

  return h(Box, { flexDirection: "column" },
    h(Text, { bold: true, color: inkColor.blueBright }, title),
    caption ? h(Box, { marginBottom: 1 }, h(Text, { color: "gray" }, caption)) : null,
    ...items.map((item, itemIndex) => h(Box, { key: item.value ?? item.label },
      h(Box, { width: 3 }, h(Text, { color: itemIndex === index ? inkColor.yellow : "gray" }, itemIndex === index ? ">" : " ")),
      h(Box, { width: 24 }, h(Text, { color: itemIndex === index ? "white" : undefined }, item.label)),
      item.description ? h(Text, { color: "gray" }, item.description) : null
    )),
    h(Box, { marginTop: 1 }, h(Text, { color: "gray" }, "Esc back | Enter select"))
  );
}

function CheckboxScreen({ title, caption, items, selectedValues, emptyMessage, onSubmit, onBack }) {
  const [index, setIndex] = useState(0);
  const [selected, setSelected] = useState(() => new Set(selectedValues));
  const [error, setError] = useState("");
  useInput((input, key) => {
    if (key.escape) {
      onBack();
      return;
    }
    if (items.length === 0) {
      if (key.return) {
        onBack();
      }
      return;
    }
    if (key.upArrow || input === "k") {
      setIndex((current) => (current - 1 + items.length) % items.length);
      return;
    }
    if (key.downArrow || input === "j") {
      setIndex((current) => (current + 1) % items.length);
      return;
    }
    if (input === " ") {
      const value = items[index].value;
      setSelected((current) => {
        const next = new Set(current);
        if (next.has(value)) {
          next.delete(value);
        } else {
          next.add(value);
        }
        return next;
      });
      setError("");
      return;
    }
    if (key.return) {
      const values = items.map((item) => item.value).filter((value) => selected.has(value));
      if (values.length === 0) {
        setError("Select at least one model.");
        return;
      }
      onSubmit(values);
    }
  });

  return h(Box, { flexDirection: "column" },
    h(Text, { bold: true, color: inkColor.blueBright }, title),
    caption ? h(Box, { marginBottom: 1 }, h(Text, { color: "gray" }, caption)) : null,
    items.length === 0 ? h(Text, { color: inkColor.yellow }, emptyMessage) : null,
    ...items.map((item, itemIndex) => {
      const active = itemIndex === index;
      const checked = selected.has(item.value);
      return h(Box, { key: item.value },
        h(Box, { width: 3 }, h(Text, { color: active ? inkColor.yellow : "gray" }, active ? ">" : " ")),
        h(Box, { width: 4 }, h(Text, { color: checked ? inkColor.blueBright : "gray" }, checked ? "[x]" : "[ ]")),
        h(Text, null, item.label)
      );
    }),
    error ? h(Box, { marginTop: 1 }, h(Text, { color: inkColor.yellow }, error)) : null,
    h(Box, { marginTop: 1 }, h(Text, { color: "gray" }, "Space toggle | Enter save | Esc back"))
  );
}

function TextInputScreen({ title, caption, initialValue = "", password = false, validate, onSubmit, onBack }) {
  const [value, setValue] = useState(initialValue);
  const [pristine, setPristine] = useState(Boolean(initialValue));
  const [error, setError] = useState("");
  useInput((input, key) => {
    if (key.escape) {
      onBack();
      return;
    }
    if (key.return) {
      const message = validate?.(value) ?? null;
      if (message) {
        setError(message);
        return;
      }
      onSubmit(value);
      return;
    }
    if (key.backspace || key.delete) {
      setPristine(false);
      setValue((current) => current.slice(0, -1));
      setError("");
      return;
    }
    if (input && !key.ctrl && !key.meta) {
      setValue((current) => pristine ? input : `${current}${input}`);
      setPristine(false);
      setError("");
    }
  });

  const visibleValue = password ? "*".repeat(value.length) : value;
  const displayValue = visibleValue || (password ? "" : " ");

  return h(Box, { flexDirection: "column" },
    h(Text, { bold: true, color: inkColor.blueBright }, title),
    caption ? h(Box, { marginBottom: 1 }, h(Text, { color: "gray" }, caption)) : null,
    h(Box, { borderStyle: "single", borderColor: error ? inkColor.yellow : inkColor.blue, paddingX: 1 },
      h(Text, null, `${displayValue}█`)
    ),
    error ? h(Box, { marginTop: 1 }, h(Text, { color: inkColor.yellow }, error)) : null,
    h(Box, { marginTop: 1 }, h(Text, { color: "gray" }, "Enter save | Esc back"))
  );
}

function ConfirmScreen({ title, caption, confirmLabel = "Confirm", destructive = false, onConfirm, onCancel, onBack }) {
  const items = destructive
    ? [
        { label: "Cancel", value: "cancel" },
        { label: confirmLabel, value: "confirm" }
      ]
    : [
        { label: confirmLabel, value: "confirm" },
        { label: "Cancel", value: "cancel" }
      ];
  return h(MenuScreen, {
    title,
    caption,
    items,
    onBack,
    onSelect: (item) => {
      if (item.value === "confirm") {
        onConfirm();
      } else {
        onCancel();
      }
    }
  });
}

function formatInkRunSummary(state, config) {
  const refs = selectedModelRefsForState(state, config);
  const providers = uniqueProvidersForRefs(config, refs);
  return [
    `Providers: ${providers.map(providerLabel).join(", ") || "none"}`,
    `Models: ${refs.map((ref) => formatModelRef(config, ref)).join(", ") || "none selected"}`
  ].join("\n");
}

async function removeAllSettings(env, config, { force = false } = {}) {
  if (!force && (!process.stdin.isTTY || !process.stdout.isTTY)) {
    throw new Error("Refusing to remove settings without confirmation. Use `modelclock reset --yes` in non-interactive environments.");
  }
  let confirmed = force;
  if (!force) {
    const result = await runInkInteractive({ command: "settings" }, env, {
      config,
      initialScreen: "resetConfirm",
      resetOnly: true
    });
    confirmed = result.reset === true;
  }
  if (confirmed !== true) {
    return false;
  }

  clearSavedSettings(env);
  console.log(color.ok("Removed saved modelclock settings."));
  return true;
}

function clearSavedSettings(env) {
  clearModelClockConfig({ configFile: CONFIG_FILE });
  try {
    fs.rmdirSync(CONFIG_DIR);
  } catch {
    // Keep the directory if it contains files not owned by this setting.
  }

  for (const key of SAVED_CONFIG_KEYS) {
    delete env[key];
  }

  return createEmptyConfig();
}

async function ensureApiConfig({ args, env, config }) {
  if (hasCompleteApiConfig(args, env, config)) {
    return;
  }

  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    throw new Error(`API config is missing. Run "modelclock" to create ${CONFIG_FILE}, or set OPENAI_BASE_URL, OPENAI_API_KEY, and OPENAI_MODELS.`);
  }

  const result = await runInkInteractive(args, env, { config, setupOnly: true });
  if (result.command === "exit") {
    throw new Error("API configuration was cancelled.");
  }
  Object.assign(args, result);
}

function hasCompleteApiConfig(args, env, config) {
  const refs = resolveModelRefs(args, env, config);
  if (refs.length === 0) {
    return false;
  }
  return refs.every((ref) => ref.provider?.baseUrl && ref.provider?.apiKey && ref.model);
}

function maskSecret(value) {
  if (!value) {
    return "";
  }
  if (value.length <= 8) {
    return "********";
  }
  return `${value.slice(0, 4)}...${value.slice(-4)}`;
}

function defaultInteractiveArgs(args, env, config) {
  const refs = resolveModelRefs(args, env, config);
  return {
    ...args,
    command: "run",
    plan: "primary",
    dataset: DEFAULTS.dataset,
    probeMode: DEFAULTS.probeMode,
    sampleSize: String(DEFAULTS.sampleSize),
    minItems: String(DEFAULTS.minItems),
    maxSteps: String(DEFAULTS.maxSteps),
    trials: String(DEFAULTS.trials),
    temperature: String(DEFAULTS.temperature),
    maxTokens: String(DEFAULTS.maxTokens),
    softwareMatchThreshold: String(DEFAULTS.softwareMatchThreshold),
    softwareMaxFalsePositives: String(DEFAULTS.softwareMaxFalsePositives),
    providerId: refs[0]?.providerId ?? config.providers[0]?.id ?? "",
    model: refs[0]?.model ?? "",
    models: refs.map((ref) => ref.model),
    modelRefs: refs
  };
}

function initialSelectedModelRefs(state, config) {
  const requestedRefs = resolveModelRefs(state, {}, config);
  if (requestedRefs.length > 0) {
    return requestedRefs.map(({ provider, providerName, displayName, ...ref }) => ref);
  }
  if (config.selectedModels.length > 0) {
    return config.selectedModels;
  }
  return allSavedModelRefs(config);
}

function materializeInteractiveRun(state) {
  return {
    ...primaryPreset(state),
    command: "run",
    modelRefs: state.modelRefs,
    models: state.modelRefs.map((ref) => ref.model),
    model: state.modelRefs[0]?.model ?? state.model,
    inspect: false
  };
}

function primaryPreset(state) {
  return {
    ...state,
    plan: "primary",
    command: "run",
    dataset: DEFAULTS.dataset,
    probeMode: DEFAULTS.probeMode,
    minItems: String(DEFAULTS.minItems),
    sampleSize: String(DEFAULTS.sampleSize),
    softwareMatchThreshold: String(DEFAULTS.softwareMatchThreshold),
    softwareMaxFalsePositives: String(DEFAULTS.softwareMaxFalsePositives),
    maxTokens: String(DEFAULTS.maxTokens)
  };
}

function buildOptions(args, env, config) {
  const modelRefs = resolveModelRefs(args, env, config);
  const options = {
    dataDir: args.dataDir ?? env.MODEL_CLOCK_DATA_DIR ?? DEFAULTS.dataDir,
    dataset: DEFAULTS.dataset,
    model: modelRefs[0]?.model ?? "",
    models: modelRefs.map((ref) => ref.model),
    modelRefs,
    modelsExplicit: args.models !== undefined || args.model !== undefined || args.modelRefs !== undefined,
    sampleSize: DEFAULTS.sampleSize,
    minItems: DEFAULTS.minItems,
    maxSteps: DEFAULTS.maxSteps,
    trials: DEFAULTS.trials,
    temperature: DEFAULTS.temperature,
    maxTokens: DEFAULTS.maxTokens,
    probeMode: DEFAULTS.probeMode,
    replicates: DEFAULTS.replicates,
    probeDatesPerRound: DEFAULTS.probeDatesPerRound,
    softwareItemsPerProbeDate: DEFAULTS.softwareItemsPerProbeDate,
    softwareDecoysPerProbeDate: DEFAULTS.softwareDecoysPerProbeDate,
    credibleIntervalDays: DEFAULTS.credibleIntervalDays,
    minRounds: DEFAULTS.minRounds,
    noisyProbeRealWeight: DEFAULTS.noisyProbeRealWeight,
    softwareMatchThreshold: DEFAULTS.softwareMatchThreshold,
    softwareMaxFalsePositives: DEFAULTS.softwareMaxFalsePositives,
    dryRun: Boolean(args.dryRun),
    json: Boolean(args.json)
  };

  return options;
}

function resolveModelRefs(args, env, config) {
  if (Array.isArray(args.modelRefs) && args.modelRefs.length > 0) {
    return enrichModelRefs(config, args.modelRefs);
  }

  const explicitModels = explicitModelArgs(args);
  const explicitProvider = args.providerId ? providerById(config, args.providerId) : null;
  if (args.providerId && !explicitProvider) {
    throw new Error(`Unknown provider id: ${args.providerId}`);
  }
  if (explicitModels.length > 0 && config.providers.length === 0) {
    return explicitModels.map((model) => ({
      providerId: "dry-run",
      provider: { id: "dry-run", name: "Dry run", baseUrl: "", apiKey: "", models: explicitModels },
      providerName: "Dry run",
      model,
      displayName: `Dry run · ${model}`
    }));
  }

  if (explicitModels.length > 0) {
    return explicitProvider
      ? enrichModelRefs(config, explicitModels.map((model) => ({ providerId: explicitProvider.id, model })))
      : resolveExplicitModelNames(config, explicitModels);
  }

  if (config.selectedModels.length > 0) {
    return enrichModelRefs(config, config.selectedModels);
  }

  return enrichModelRefs(config, allSavedModelRefs(config));
}

function explicitModelArgs(args) {
  if (Array.isArray(args.models)) {
    return args.models.filter(Boolean);
  }
  if (typeof args.models === "string") {
    return parseCsvEnv(args.models);
  }
  if (args.model) {
    return [args.model];
  }
  return [];
}

function resolveExplicitModelNames(config, models) {
  const refs = [];
  for (const model of models) {
    const matches = config.providers
      .filter((provider) => provider.models.includes(model))
      .map((provider) => ({ providerId: provider.id, model }));
    if (matches.length === 1) {
      refs.push(matches[0]);
      continue;
    }
    if (matches.length > 1) {
      throw new Error(`Model "${model}" exists under multiple providers. Choose it in the interactive model list.`);
    }
    throw new Error(`Model "${model}" is not saved under any provider. Add it in settings first.`);
  }
  return enrichModelRefs(config, refs);
}

function enrichModelRefs(config, refs) {
  return refs
    .map((ref) => {
      const provider = providerById(config, ref.providerId);
      if (!provider || !ref.model) {
        return null;
      }
      return {
        providerId: provider.id,
        provider,
        providerName: providerLabel(provider),
        model: ref.model,
        displayName: formatModelRef(config, { providerId: provider.id, model: ref.model })
      };
    })
    .filter(Boolean);
}

function resolveDefaultDataDir() {
  const candidates = [
    path.resolve(MODULE_DIR, "..", "data"),
    path.resolve(path.dirname(process.execPath), "data"),
    path.resolve(path.dirname(process.execPath), "..", "data"),
    path.resolve(process.cwd(), "data")
  ];
  return candidates.find((candidate) => fs.existsSync(path.join(candidate, "software-releases.json"))) ?? candidates[0];
}

function toCamelCase(value) {
  return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function createDryRunClient(options) {
  return {
    async ask({ expectedSoftwareQuestions }) {
      return {
        content: JSON.stringify({
          answers: expectedSoftwareQuestions.map((question) => ({
            index: question.index,
            answer: question.expected
          })),
          reason: "dry-run echoes expected software answers"
        }),
        usage: null
      };
    }
  };
}

function createColorTheme() {
  const enabled = isColorEnabled();
  const wrap = (open, close = "\x1b[0m") => (value) => enabled ? `${open}${value}${close}` : value;
  const rgb = (red, green, blue) => wrap(`\x1b[38;2;${red};${green};${blue}m`, "\x1b[39m");
  const blue = rgb(37, 99, 235);
  const blueBright = rgb(56, 189, 248);
  const yellow = rgb(250, 204, 21);
  return {
    cyan: blueBright,
    magenta: yellow,
    blue,
    blueBright,
    green: blueBright,
    yellow,
    bold: wrap("\x1b[1m", "\x1b[22m"),
    dim: wrap("\x1b[2m", "\x1b[22m"),
    accent: blueBright,
    ok: blueBright,
    warn: yellow
  };
}

function printBanner() {
  console.log("");
  const maxWidth = Math.max(Math.min(process.stdout.columns ?? 100, 120), 24);
  const title = centerTextBlock(banner("MODEL CLOCK", maxWidth), maxWidth);
  console.log(isColorEnabled() ? verticalGradientBlock(title, ["#2563eb", "#facc15"]) : title);
  console.log("");
  console.log(color.yellow(centerText(TAGLINE, maxWidth)));
}

function clearTerminal() {
  if (!process.stdout.isTTY) {
    return;
  }
  process.stdout.write("\x1b[2J\x1b[3J\x1b[H");
}

function banner(text, maxWidth) {
  const normalized = text.toUpperCase();
  const fonts = maxWidth >= 36
    ? ["ANSI Shadow", "Big", "Standard", "Small", "Mini"]
    : ["Small", "Mini"];

  for (const font of fonts) {
    const rendered = figlet.textSync(normalized, {
      font,
      horizontalLayout: "fitted",
      verticalLayout: "fitted"
    });

    if (widestLine(rendered) <= maxWidth && rendered.trim().length > 0) {
      return rendered;
    }
  }

  return normalized;
}

function widestLine(text) {
  return Math.max(...text.split("\n").map((line) => line.length));
}

function centerTextBlock(text, width) {
  return text.split("\n").map((line) => centerText(line, width)).join("\n");
}

function centerText(text, width) {
  const padding = Math.max(Math.floor((width - text.length) / 2), 0);
  return `${" ".repeat(padding)}${text}`;
}

function verticalGradientBlock(text, colors) {
  const lines = text.split("\n");
  const from = hexToRgb(colors[0]);
  const to = hexToRgb(colors[1]);
  const denominator = Math.max(lines.length - 1, 1);

  return lines
    .map((line, index) => {
      const ratio = index / denominator;
      const red = Math.round(from.red + (to.red - from.red) * ratio);
      const green = Math.round(from.green + (to.green - from.green) * ratio);
      const blue = Math.round(from.blue + (to.blue - from.blue) * ratio);

      return `\x1b[38;2;${red};${green};${blue}m${line}\x1b[39m`;
    })
    .join("\n");
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return {
    red: Number.parseInt(value.slice(0, 2), 16),
    green: Number.parseInt(value.slice(2, 4), 16),
    blue: Number.parseInt(value.slice(4, 6), 16)
  };
}

function isColorEnabled() {
  return process.stdout.isTTY && process.env.NO_COLOR === undefined;
}

function printRunHeader(options) {
  console.log("");
  console.log(color.accent("Starting boundary search"));
  console.log(formatTable(["Setting", "Value"], [
    ["Provider", options.providerName ?? "Provider"],
    ["Model", options.dryRun ? `${options.model} (dry-run)` : (options.modelDisplay ?? options.model)]
  ], { borderColor: color.blue }));
}

function printStepProgress(step) {
  if (Array.isArray(step.probes)) {
    const prefix = step.replicates > 1
      ? `Pass ${step.replicate}/${step.replicates} · Round ${step.round}`
      : `Round ${step.round}`;
    console.log(color.dim(prefix));
    for (const probe of step.probes) {
      const styledMark = styleProbeStatus(probe.status);
      console.log(`  ${probe.date} -> ${styledMark} (real ${probe.positiveMatches}/${probe.positiveTotal}; decoys ${probe.falsePositives}/${probe.decoyTotal})`);
    }
    if (step.posterior?.credibleInterval95?.label) {
      console.log(color.dim(`  95% range ${step.posterior.credibleInterval95.label}`));
    }
    return;
  }

  const mark = step.correct ? "known" : "unknown";
  const styledMark = step.correct ? color.ok(mark) : color.warn(mark);
  console.log(`${color.dim(`Step ${step.step}`)} ${step.date} -> ${styledMark} (${step.correctTrials}/${step.trials} correct)`);
}

function styleProbeStatus(status) {
  if (status === "known") {
    return color.ok("known");
  }
  if (status === "unknown") {
    return color.warn("unknown");
  }
  return color.yellow(status);
}

function printInspect(dataset, options) {
  console.log(formatTable(["Field", "Value"], [
    ["Dataset", options.dataset],
    ["Usable dates", String(dataset.entries.length)],
    ["Range", `${dataset.entries[0]?.date ?? "n/a"} to ${dataset.entries.at(-1)?.date ?? "n/a"}`],
    ["Sample size", String(options.sampleSize)],
    ["Minimum items per date", String(options.minItems)]
  ], { borderColor: color.blue }));
}

function printResult(result, options) {
  console.log("");
  console.log(color.accent("Result range"));
  console.log(formatTable(["Field", "Value"], [
    ["Provider", options.providerName ?? "Provider"],
    ["Model", options.dryRun ? `${options.model} (dry-run)` : (options.modelDisplay ?? options.model)],
    ["Estimated cutoff", result.estimatedCutoff ?? "none observed"],
    ["80% credible range", result.credibleInterval80?.label ?? "n/a"],
    ["95% credible range", result.credibleInterval95?.label ?? "n/a"],
    ["Decoy false positives", `${result.decoyFalsePositives ?? 0}/${result.decoyTotal ?? 0}`],
    ["Replicates", String(result.replicates ?? 1)],
    ["Rounds", String(result.rounds ?? result.steps?.length ?? 0)],
    ["Status", result.status ?? (result.converged ? "converged" : "partial")]
  ], { borderColor: color.yellow, outerBorder: "double" }));
}

function printResultsSummary(runResults) {
  console.log("");
  console.log(color.accent("Model summary"));
  console.log(formatTable(["Model", "Estimated cutoff", "80% range", "95% range", "Decoys", "Passes", "Rounds", "Status"], runResults.map(({ model, result }) => [
    model,
    result.estimatedCutoff ?? "none observed",
    result.credibleInterval80?.label ?? "n/a",
    result.credibleInterval95?.label ?? "n/a",
    `${result.decoyFalsePositives ?? 0}/${result.decoyTotal ?? 0}`,
    String(result.replicates ?? 1),
    String(result.rounds ?? result.steps?.length ?? 0),
    result.status ?? (result.converged ? "converged" : "partial")
  ]), { borderColor: color.blue, outerBorder: "double" }));
}

function formatTable(headers, rows, options = {}) {
  const normalizedRows = rows.map((row) => row.map(normalizeTableCell));
  const normalizedHeaders = headers.map(normalizeTableCell);
  const paint = options.borderColor ?? ((value) => value);
  const widths = normalizedHeaders.map((header, columnIndex) => Math.max(
    header.length,
    ...normalizedRows.map((row) => row[columnIndex]?.length ?? 0)
  ));
  const outer = options.outerBorder === "double"
    ? {
        horizontal: "═",
        vertical: "║",
        topLeft: "╔",
        topJoin: "╤",
        topRight: "╗",
        headerLeft: "╟",
        headerJoin: "┼",
        headerRight: "╢",
        bottomLeft: "╚",
        bottomJoin: "╧",
        bottomRight: "╝"
      }
    : {
        horizontal: "─",
        vertical: "│",
        topLeft: "┌",
        topJoin: "┬",
        topRight: "┐",
        headerLeft: "├",
        headerJoin: "┼",
        headerRight: "┤",
        bottomLeft: "└",
        bottomJoin: "┴",
        bottomRight: "┘"
      };
  const border = (left, horizontal, join, right) => paint(`${left}${widths.map((width) => horizontal.repeat(width + 2)).join(join)}${right}`);
  const line = (cells) => `${paint(outer.vertical)}${cells.map((cell, index) => ` ${cell.padEnd(widths[index])} `).join(paint("│"))}${paint(outer.vertical)}`;

  return [
    border(outer.topLeft, outer.horizontal, outer.topJoin, outer.topRight),
    line(normalizedHeaders),
    border(outer.headerLeft, "─", outer.headerJoin, outer.headerRight),
    ...normalizedRows.map(line),
    border(outer.bottomLeft, outer.horizontal, outer.bottomJoin, outer.bottomRight)
  ].join("\n");
}

function normalizeTableCell(value) {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function helpText() {
  return `modelclock

Estimate a model's knowledge boundary with dated software release probes.

Usage:
  modelclock
  modelclock run [options]
  modelclock settings
  modelclock reset
  modelclock interactive

Options:
  --model <name>            Run one saved model for this invocation
  --models <a,b>            Run comma-separated saved models for this invocation
  --data-dir <path>         Use a custom data directory containing software-releases.json
  --dry-run                 Do not call an API; echo expected answers for smoke tests
  --json                    Print full result as JSON
  -y, --yes                 Skip confirmation for reset
  -h, --help                Show help

Saved settings:
  ${CONFIG_FILE}

Required values for real runs:
  Configure providers interactively, or provide compatibility env vars:
    OPENAI_BASE_URL=https://api.openai.com/v1
    OPENAI_API_KEY=...
    OPENAI_MODELS=saved-model-name,another-model-name

If config is missing and the terminal is interactive, modelclock will ask for provider, key, and models, then save ${CONFIG_FILE}.

Examples:
  npx modelclock
  modelclock
  modelclock run
  modelclock settings
  modelclock reset
  modelclock reset --yes
  modelclock run --models saved-model-a,saved-model-b
`;
}
