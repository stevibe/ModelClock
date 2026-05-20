# modelclock

Probe LLM knowledge cutoffs with dated software release versions.

`modelclock` is a CLI that asks models whether specific software versions had been released. It uses those answers to estimate a practical knowledge boundary with uncertainty. It is not a claim about an official training cutoff date.

## Why Software Versions?

Software release data is a useful signal for cutoff probing because it is dated, public, high-volume, and concrete. It is also likely to appear in model training data: most AI companies are actively improving coding capability, so training on package registries, release notes, GitHub releases, framework docs, and developer discussions is a reasonable trend.

That makes software versions a practical probe target. A model may know that React `19.0.0`, Node.js `22.x`, or a PyTorch release exists, while being less able to answer about versions released after its knowledge boundary. `modelclock` turns that behavior into a repeatable date-range estimate.

## Quick Start

Run directly:

```sh
npx modelclock
```

On first run, the interactive CLI asks for:

```text
Provider
API key
Model name(s)
```

Supported provider presets:

```text
OpenRouter
Hugging Face
OpenAI Compatible
```

Settings are saved to:

```text
~/.modelclock/config.json
```

The software release dataset is committed in this repo at [data/software-releases.json](data/software-releases.json), so users do not need to fetch data before running the probe.

## Local Development

Clone and run with npm:

```sh
npm install
npm run start
```

Useful scripts:

```sh
npm run start                # open the interactive CLI
npm run smoke                # run a local dry run without API calls
npm run check                # syntax and CLI help checks
npm run build:software-data  # refresh data/software-releases.json
```

The data refresh script collects release metadata from npm, PyPI, Node.js, Python.org, and selected GitHub release feeds. The generated JSON is committed so normal users can run immediately.

## Usage

Interactive:

```sh
modelclock
modelclock settings
modelclock reset
```

Non-interactive:

```sh
modelclock run --models saved-model-a,saved-model-b
modelclock --dry-run
modelclock --json
```

`--models` refers to model names already saved under providers. If the same model name exists under multiple providers, use the interactive model picker.

## Configuration

The saved config is JSON:

```json
{
  "version": 1,
  "providers": [
    {
      "id": "hugging-face",
      "name": "Hugging Face",
      "baseUrl": "https://router.huggingface.co/v1",
      "apiKey": "your-api-key",
      "models": ["zai-org/GLM-5.1:together"]
    }
  ],
  "selectedModels": [
    {
      "providerId": "hugging-face",
      "model": "zai-org/GLM-5.1:together"
    }
  ]
}
```

Environment variables are still supported as a compatibility fallback:

```sh
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key
OPENAI_MODELS=model-a,model-b
```

## Method

`modelclock` uses fixed probe defaults:

```text
dataset: software-releases
probe: software-version
probe dates per round: 3
real versions per probe date: 5
decoy versions per probe date: 3
independent replicate passes: 3
temperature: 0
max response tokens: 2048
credible interval target: 95% range within 14 days
minimum rounds before accepting a tight interval: 12
max rounds: 30 per replicate
```

Each round chooses three dates from the current posterior distribution, asks about real releases plus plausible nonexistent decoy versions, updates a beta-binomial posterior over possible cutoff positions, and reports 80% and 95% credible ranges.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for scoring rules, decoy handling, replicated passes, stopping criteria, and reporting guidance.

## License

MIT © stevibe
