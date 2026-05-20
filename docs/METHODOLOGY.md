# Methodology

`modelclock` estimates an LLM's practical knowledge boundary from dated software releases. It reports a dataset-specific behavioral date range with uncertainty. It does not prove an official training cutoff date.

## Why Software Releases?

Software release versions are a useful cutoff signal because they are public, dated, concrete, and abundant. They also sit close to the data that many model builders care about: coding capability. AI companies have strong incentives to train on software-related material such as package registries, release notes, GitHub releases, framework documentation, issue discussions, and developer Q&A.

That makes software versions a credible probe target. A model that has seen enough post-release software data may know that an exact version exists. A model past its knowledge boundary should become more conservative, uncertain, or wrong on later releases.

## Dataset

The committed dataset is [data/software-releases.json](../data/software-releases.json). Each record includes:

```json
{
  "date": "2024-12-05",
  "project": "React",
  "version": "19.0.0",
  "source": "npm",
  "url": "https://www.npmjs.com/package/react/v/19.0.0",
  "prompt": "Had React released version 19.0.0 yet?"
}
```

The dataset currently combines releases from npm, PyPI, Node.js, Python.org, and selected GitHub release feeds. The refresh script is available as:

```sh
npm run build:software-data
```

The JSON is committed so users can run `modelclock` without fetching data.

## Prompt

The model sees questions like:

```text
Had React released version 19.0.0 yet?
Reply YES, NO, or UNKNOWN.
```

No release dates are provided. The system prompt asks the model to answer from memory, avoid browsing/tools, and avoid inferring from semantic version patterns.

## Sequential Search

The dataset is sorted by date. `modelclock` maintains a posterior distribution over every possible cutoff position in that date-ordered dataset.

Each round chooses three probe dates from the current posterior:

```text
lower probe   = 25% posterior cutoff quantile
center probe  = 50% posterior cutoff quantile
upper probe   = 75% posterior cutoff quantile
```

If multiple quantiles collapse to the same date, the selector fills remaining slots with high-information dates, while penalizing dates already asked in the current pass.

For each probe date, the probe asks:

```text
5 real software versions released on that date
3 plausible nonexistent versions
```

One round therefore contains up to 15 real release questions and 9 decoy questions.

## Item Selection

When a date has more eligible releases than needed, the sampler prefers high-salience projects such as Python, Node.js, React, TypeScript, Next.js, Vite, Vue, Angular, VS Code, Kubernetes, PyTorch, NumPy, and similar widely observed packages before falling back to the remaining projects.

Within a date, the sampler also tries to diversify projects before selecting multiple releases from the same project. Replicate passes rotate item offsets so that repeated runs do not depend on the exact same version subset.

## Scoring

For a candidate cutoff position `C`:

- real releases at or before `C` are expected to receive `YES`
- real releases after `C` are expected to receive `NO` or `UNKNOWN`
- decoy versions are expected to receive `NO` or `UNKNOWN`

For each probe date:

- `known`: at least two thirds of real releases receive `YES`, with no decoy false positives
- `unknown`: at most one third of real releases receive `YES`, with no decoy false positives
- `mixed`: between those thresholds, with no decoy false positives
- `noisy`: at least one decoy receives `YES`

Probe statuses are used for the live transcript. The posterior itself is updated from the underlying per-question observations.

## Statistical Model

The posterior uses beta-binomial marginal likelihoods:

```text
known-region real YES rate:         Beta(8, 2)
unknown-region accidental YES rate: Beta(1, 9)
decoy-version YES rate:             Beta(1, 19)
```

This allows the estimate to tolerate imperfect model behavior instead of assuming a perfectly monotonic cutoff.

If a probe date has any decoy false positives, real `YES` answers from that same probe are downweighted to 25% weight. This prevents overclaiming from looking like strong evidence that the date is known.

## Replicated Passes

By default, `modelclock` runs three independent replicate passes. Each pass:

- starts with the same prior
- uses the same posterior-guided date selection rule
- rotates selected software versions differently
- produces its own estimated cutoff and credible intervals

The final report aggregates replicates by:

- using the median estimated cutoff position
- summing decoy false positives and totals
- summing rounds
- taking the union of replicate 80% intervals
- taking the union of replicate 95% intervals

This is intentionally conservative. It makes instability visible instead of reporting a falsely narrow single-pass range.

## Stopping Rule

Each replicate stops when either:

```text
95% credible range <= 14 days
```

after at least 12 rounds, or when the posterior collapses to a dataset edge. A 30-round per-replicate cap prevents unbounded runs.

Default values:

```text
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

## Reporting

For public probe reports, include:

```text
model:
provider:
modelclock version/commit:
estimated cutoff:
80% credible range:
95% credible range:
decoy false positives:
replicate count:
round transcript:
dataset version/commit:
```

Example:

```text
Estimated cutoff      2024-08-23 to 2024-08-28
80% credible range    2024-08-21 to 2024-08-29
95% credible range    2024-08-18 to 2024-09-02
Decoy false positives 0/60
```

## Threats To Validity

Important limitations:

- Popular software projects are overrepresented.
- Release existence can sometimes be inferred from version conventions.
- Some models overclaim and say `YES` to decoy versions.
- Some models are conservative and answer `UNKNOWN` for known artifacts.
- Provider wrappers can change model behavior through hidden prompts or reasoning modes.
- Dataset records may reflect package registry upload dates, not announcement dates.
- A model can know a project version but not its exact release chronology.
- A measured boundary can differ by provider, prompt wrapper, decoding settings, and model route.

Decoy versions reduce, but do not eliminate, overclaiming. Read results as a dataset-specific behavioral boundary, not a universal cutoff. The measured range means "the model passed this software-release probe around this point with the stated uncertainty," not "the model was trained until this exact day."
