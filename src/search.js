import { buildMessages } from "./prompt.js";

const POSTERIOR_PRIORS = {
  knownRealYes: { alpha: 8, beta: 2 },
  unknownRealYes: { alpha: 1, beta: 9 },
  decoyYes: { alpha: 1, beta: 19 }
};

const PROBE_QUANTILES = [0.25, 0.5, 0.75];

export async function runBoundarySearch({ dataset, client, options, onStep }) {
  const replicates = Math.max(1, options.replicates ?? 1);
  if (replicates === 1) {
    return runSingleBoundarySearch({ dataset, client, options, onStep, replicate: 1, replicates });
  }

  const replicateResults = [];
  for (let replicate = 1; replicate <= replicates; replicate += 1) {
    replicateResults.push(await runSingleBoundarySearch({
      dataset,
      client,
      options,
      onStep,
      replicate,
      replicates
    }));
  }
  return aggregateReplicates({ entries: dataset.entries, replicateResults, options });
}

async function runSingleBoundarySearch({ dataset, client, options, onStep, replicate = 1, replicates = 1 }) {
  const observations = [];
  const steps = [];
  const probedDateCounts = new Map();
  let posterior = computePosterior({ entries: dataset.entries, observations });
  let summary = summarizePosterior({ entries: dataset.entries, posterior, observations, options });

  while (steps.length < options.maxSteps) {
    const round = steps.length + 1;
    const probeIndices = selectProbeIndices({
      entries: dataset.entries,
      posterior,
      count: options.probeDatesPerRound,
      probedDateCounts
    });
    const probes = probeIndices.map((entryIndex, probeIndex) => {
      const seenCount = probedDateCounts.get(entryIndex) ?? 0;
      probedDateCounts.set(entryIndex, seenCount + 1);
      return buildProbe({
        entry: dataset.entries[entryIndex],
        entryIndex,
        round,
        probeIndex,
        replicate,
        seenCount,
        options
      });
    });
    const questions = buildRoundQuestions(probes);
    const messages = buildMessages({ datasetName: dataset.name, softwareQuestions: questions });
    const response = await client.ask({
      messages,
      date: probes.map((probe) => probe.date).join(","),
      expectedSoftwareQuestions: questions
    });
    const parsed = parseSoftwareAnswers(response.content);
    const scored = scoreRound({ probes, questions, answers: parsed.answers, options });

    observations.push(...scored.observations);
    posterior = computePosterior({ entries: dataset.entries, observations });
    summary = summarizePosterior({ entries: dataset.entries, posterior, observations, options });

    const step = {
      step: round,
      round,
      replicate,
      replicates,
      date: probes.map((probe) => probe.date).join(","),
      correct: scored.probes.some((probe) => probe.status === "known"),
      correctTrials: scored.probes.filter((probe) => probe.status === "known").length,
      trials: scored.probes.length,
      probes: scored.probes,
      posterior: publicPosteriorSummary(summary),
      trialResults: [{
        trial: 1,
        correct: scored.probes.some((probe) => probe.status === "known"),
        mode: "software-version",
        positiveMatches: scored.probes.reduce((total, probe) => total + probe.positiveMatches, 0),
        positiveTotal: scored.probes.reduce((total, probe) => total + probe.positiveTotal, 0),
        falsePositives: scored.probes.reduce((total, probe) => total + probe.falsePositives, 0),
        decoyTotal: scored.probes.reduce((total, probe) => total + probe.decoyTotal, 0),
        questions,
        answers: parsed.answers,
        raw: response.content,
        usage: response.usage
      }]
    };

    steps.push(step);
    if (onStep) {
      onStep(step);
    }

    if (summary.endpointCollapsed || (summary.precise && steps.length >= options.minRounds)) {
      break;
    }
  }

  return {
    method: "bayesian-tri-probe",
    replicate,
    replicates,
    converged: summary.converged,
    status: summary.status,
    estimatedCutoff: summary.estimatedCutoff,
    estimatedCutoffPosition: summary.estimatedCutoffPosition,
    estimatedKnownDate: summary.latestKnownDate,
    firstUnknownDate: summary.firstUnknownDate,
    credibleInterval80: summary.credibleInterval80,
    credibleInterval95: summary.credibleInterval95,
    decoyFalsePositives: summary.decoyFalsePositives,
    decoyTotal: summary.decoyTotal,
    rounds: steps.length,
    searchWindow: summary.credibleInterval95,
    datasetRange: {
      start: dataset.entries[0].date,
      end: dataset.entries.at(-1).date,
      usableDates: dataset.entries.length
    },
    steps
  };
}

function aggregateReplicates({ entries, replicateResults, options }) {
  const estimatedPositions = replicateResults
    .map((result) => result.estimatedCutoffPosition)
    .filter((position) => Number.isInteger(position))
    .sort((a, b) => a - b);
  const estimatedPosition = estimatedPositions[Math.floor(estimatedPositions.length / 2)] ?? entries.length;
  const credibleInterval80 = aggregateInterval({ entries, replicateResults, key: "credibleInterval80", level: 0.8 });
  const credibleInterval95 = aggregateInterval({ entries, replicateResults, key: "credibleInterval95", level: 0.95 });
  const decoyFalsePositives = replicateResults.reduce((total, result) => total + result.decoyFalsePositives, 0);
  const decoyTotal = replicateResults.reduce((total, result) => total + result.decoyTotal, 0);
  const rounds = replicateResults.reduce((total, result) => total + result.rounds, 0);
  const stable = credibleInterval95.widthDays !== null && credibleInterval95.widthDays <= options.credibleIntervalDays;
  const statuses = [...new Set(replicateResults.map((result) => result.status))];
  const sharedCensoredStatus = statuses.length === 1 && /censored/u.test(statuses[0]) ? statuses[0] : null;

  return {
    method: "replicated-bayesian-tri-probe",
    converged: stable || replicateResults.every((result) => result.converged),
    status: sharedCensoredStatus ?? (stable
      ? `replicated 95% range <= ${options.credibleIntervalDays} days`
      : `replicate-adjusted range from ${replicateResults.length} passes`),
    estimatedCutoff: boundaryLabel(entries, estimatedPosition),
    estimatedCutoffPosition: estimatedPosition,
    estimatedKnownDate: latestKnownDate(entries, estimatedPosition),
    firstUnknownDate: firstUnknownDate(entries, estimatedPosition),
    credibleInterval80,
    credibleInterval95,
    decoyFalsePositives,
    decoyTotal,
    rounds,
    replicates: replicateResults.length,
    replicateResults: replicateResults.map(compactReplicateResult),
    searchWindow: credibleInterval95,
    datasetRange: {
      start: entries[0].date,
      end: entries.at(-1).date,
      usableDates: entries.length
    },
    steps: replicateResults.flatMap((result) => result.steps)
  };
}

function aggregateInterval({ entries, replicateResults, key, level }) {
  const intervals = replicateResults.map((result) => result[key]).filter(Boolean);
  const startPosition = Math.min(...intervals.map((interval) => interval.startPosition));
  const endPosition = Math.max(...intervals.map((interval) => interval.endPosition));

  return {
    level,
    startPosition,
    endPosition,
    label: positionRangeLabel(entries, startPosition, endPosition),
    startDate: intervalStartDate(entries, startPosition),
    endDate: intervalEndDate(entries, endPosition),
    widthDays: intervalWidthDays(entries, startPosition, endPosition)
  };
}

function compactReplicateResult(result) {
  return {
    replicate: result.replicate,
    estimatedCutoff: result.estimatedCutoff,
    credibleInterval80: result.credibleInterval80,
    credibleInterval95: result.credibleInterval95,
    decoyFalsePositives: result.decoyFalsePositives,
    decoyTotal: result.decoyTotal,
    rounds: result.rounds,
    status: result.status
  };
}

function buildProbe({ entry, entryIndex, round, probeIndex, replicate, seenCount, options }) {
  const realCount = options.softwareItemsPerProbeDate;
  const decoyCount = options.softwareDecoysPerProbeDate;
  const items = selectDiverseItems(entry.items, realCount, ((replicate - 1) * 97) + ((round + probeIndex + seenCount) * realCount));

  return {
    id: `r${round}p${probeIndex + 1}`,
    date: entry.date,
    entryIndex,
    items,
    decoyItems: items.slice(0, Math.min(decoyCount, items.length))
  };
}

function selectDiverseItems(items, count, offset) {
  const rotated = rotate(items, items.length === 0 ? 0 : offset % items.length);
  const selected = [];
  const seenProjects = new Set();

  for (const item of rotated) {
    const key = item.project ?? item.title;
    if (seenProjects.has(key)) {
      continue;
    }
    selected.push(item);
    seenProjects.add(key);
    if (selected.length >= count) {
      return selected;
    }
  }

  for (const item of rotated) {
    if (selected.includes(item)) {
      continue;
    }
    selected.push(item);
    if (selected.length >= count) {
      return selected;
    }
  }

  return selected;
}

function rotate(items, offset) {
  if (items.length === 0) {
    return [];
  }
  return [...items.slice(offset), ...items.slice(0, offset)];
}

function buildRoundQuestions(probes) {
  const realQuestions = [];
  const decoyQuestions = [];

  for (const probe of probes) {
    for (const item of probe.items) {
      realQuestions.push({
        expected: "YES",
        kind: "real",
        probeId: probe.id,
        probeDate: probe.date,
        releaseIndex: probe.entryIndex,
        releaseDate: probe.date,
        project: item.project,
        version: item.version,
        prompt: releaseExistencePrompt(item.project, item.version)
      });
    }

    for (const item of probe.decoyItems) {
      decoyQuestions.push({
        expected: "NO",
        kind: "decoy",
        probeId: probe.id,
        probeDate: probe.date,
        releaseIndex: probe.entryIndex,
        releaseDate: probe.date,
        project: item.project,
        version: item.decoyVersion,
        prompt: releaseExistencePrompt(item.project, item.decoyVersion)
      });
    }
  }

  return deterministicInterleave(realQuestions, decoyQuestions).map((question, index) => ({
    ...question,
    index: index + 1
  }));
}

function releaseExistencePrompt(project, version) {
  return `Had ${project} released version ${version} yet?`;
}

function deterministicInterleave(positives, decoys) {
  const result = [];
  const maxLength = Math.max(positives.length, decoys.length);
  for (let index = 0; index < maxLength; index += 1) {
    if (positives[index]) {
      result.push(positives[index]);
    }
    if (decoys[index]) {
      result.push(decoys[index]);
    }
  }
  return result;
}

function scoreRound({ probes, questions, answers, options }) {
  const byIndex = new Map(answers.map((answer) => [answer.index, answer.answer]));
  const probeScores = new Map(probes.map((probe) => [probe.id, {
    id: probe.id,
    date: probe.date,
    entryIndex: probe.entryIndex,
    positiveMatches: 0,
    positiveTotal: 0,
    falsePositives: 0,
    decoyTotal: 0,
    status: "unknown"
  }]));
  const observations = [];

  for (const question of questions) {
    const answer = byIndex.get(question.index) ?? "UNKNOWN";
    const answeredYes = answer === "YES";
    const probe = probeScores.get(question.probeId);

    if (question.kind === "real") {
      probe.positiveTotal += 1;
      if (answeredYes) {
        probe.positiveMatches += 1;
      }
      observations.push({
        kind: "real",
        probeId: question.probeId,
        releaseIndex: question.releaseIndex,
        releaseDate: question.releaseDate,
        answer,
        weight: 1
      });
    } else {
      probe.decoyTotal += 1;
      if (answeredYes) {
        probe.falsePositives += 1;
      }
      observations.push({
        kind: "decoy",
        probeId: question.probeId,
        releaseIndex: question.releaseIndex,
        releaseDate: question.releaseDate,
        answer,
        weight: 1
      });
    }
  }

  const scoredProbes = [...probeScores.values()].map((probe) => ({
    ...probe,
    status: classifyProbe(probe)
  }));
  const noisyProbeIds = new Set(scoredProbes
    .filter((probe) => probe.falsePositives > 0)
    .map((probe) => probe.id));
  const weightedObservations = observations.map((observation) => {
    if (observation.kind === "real" && noisyProbeIds.has(observation.probeId)) {
      return {
        ...observation,
        weight: options.noisyProbeRealWeight ?? 0.25
      };
    }
    return observation;
  });

  return { probes: scoredProbes, observations: weightedObservations };
}

function classifyProbe(probe) {
  if (probe.falsePositives > 0) {
    return "noisy";
  }
  const yesRate = probe.positiveTotal === 0 ? 0 : probe.positiveMatches / probe.positiveTotal;
  if (yesRate >= 2 / 3) {
    return "known";
  }
  if (yesRate <= 1 / 3) {
    return "unknown";
  }
  return "mixed";
}

function computePosterior({ entries, observations }) {
  const logWeights = [];

  for (let position = 0; position <= entries.length; position += 1) {
    const counts = countObservationsForCutoff({ position, observations });
    logWeights.push(
      betaBinomialLogMarginal(POSTERIOR_PRIORS.knownRealYes, counts.knownYes, counts.knownNo)
      + betaBinomialLogMarginal(POSTERIOR_PRIORS.unknownRealYes, counts.unknownYes, counts.unknownNo)
      + betaBinomialLogMarginal(POSTERIOR_PRIORS.decoyYes, counts.decoyYes, counts.decoyNo)
    );
  }

  const maxLogWeight = Math.max(...logWeights);
  const weights = logWeights.map((weight) => Math.exp(weight - maxLogWeight));
  const totalWeight = weights.reduce((total, weight) => total + weight, 0);
  const probabilities = weights.map((weight) => weight / totalWeight);
  let cumulative = 0;
  const cdf = probabilities.map((probability) => {
    cumulative += probability;
    return cumulative;
  });

  return { probabilities, cdf };
}

function countObservationsForCutoff({ position, observations }) {
  const counts = {
    knownYes: 0,
    knownNo: 0,
    unknownYes: 0,
    unknownNo: 0,
    decoyYes: 0,
    decoyNo: 0
  };

  for (const observation of observations) {
    const answeredYes = observation.answer === "YES";
    const weight = observation.weight ?? 1;
    if (observation.kind === "decoy") {
      if (answeredYes) {
        counts.decoyYes += weight;
      } else {
        counts.decoyNo += weight;
      }
      continue;
    }

    if (observation.releaseIndex < position) {
      if (answeredYes) {
        counts.knownYes += weight;
      } else {
        counts.knownNo += weight;
      }
    } else if (answeredYes) {
      counts.unknownYes += weight;
    } else {
      counts.unknownNo += weight;
    }
  }

  return counts;
}

function selectProbeIndices({ entries, posterior, count, probedDateCounts }) {
  const selected = new Set();
  const desiredCount = Math.min(count, entries.length);

  for (const quantile of PROBE_QUANTILES.slice(0, desiredCount)) {
    selected.add(positionToProbeIndex(findQuantilePosition(posterior, quantile), entries.length));
  }

  if (selected.size < desiredCount) {
    const ranked = entries
      .map((_entry, index) => ({
        index,
        score: informationScoreForProbe({ posterior, probeIndex: index }) / (1 + (probedDateCounts.get(index) ?? 0))
      }))
      .sort((a, b) => b.score - a.score || a.index - b.index);

    for (const candidate of ranked) {
      selected.add(candidate.index);
      if (selected.size >= desiredCount) {
        break;
      }
    }
  }

  return [...selected].sort((a, b) => a - b);
}

function informationScoreForProbe({ posterior, probeIndex }) {
  const knownProbability = posterior.probabilities
    .slice(probeIndex + 1)
    .reduce((total, probability) => total + probability, 0);
  return knownProbability * (1 - knownProbability);
}

function positionToProbeIndex(position, entryCount) {
  return Math.min(Math.max(position, 0), entryCount - 1);
}

function summarizePosterior({ entries, posterior, observations, options }) {
  const estimatedPosition = findQuantilePosition(posterior, 0.5);
  const credibleInterval80 = credibleInterval({ entries, posterior, level: 0.8 });
  const credibleInterval95 = credibleInterval({ entries, posterior, level: 0.95 });
  const decoyStats = summarizeDecoys(observations);
  const precise = credibleInterval95.widthDays !== null
    && credibleInterval95.widthDays <= options.credibleIntervalDays;
  const endpointCollapsed = credibleInterval95.startPosition === credibleInterval95.endPosition
    && (credibleInterval95.startPosition === 0 || credibleInterval95.startPosition === entries.length);
  const converged = precise || endpointCollapsed;

  return {
    precise,
    endpointCollapsed,
    estimatedCutoffPosition: estimatedPosition,
    estimatedCutoff: boundaryLabel(entries, estimatedPosition),
    latestKnownDate: latestKnownDate(entries, estimatedPosition),
    firstUnknownDate: firstUnknownDate(entries, estimatedPosition),
    credibleInterval80,
    credibleInterval95,
    decoyFalsePositives: decoyStats.falsePositives,
    decoyTotal: decoyStats.total,
    converged,
    status: posteriorStatus({ entries, credibleInterval95, precise, options })
  };
}

function publicPosteriorSummary(summary) {
  return {
    estimatedCutoff: summary.estimatedCutoff,
    credibleInterval80: summary.credibleInterval80,
    credibleInterval95: summary.credibleInterval95,
    decoyFalsePositives: summary.decoyFalsePositives,
    decoyTotal: summary.decoyTotal,
    status: summary.status
  };
}

function credibleInterval({ entries, posterior, level }) {
  const tail = (1 - level) / 2;
  const startPosition = findQuantilePosition(posterior, tail);
  const endPosition = findQuantilePosition(posterior, 1 - tail);

  return {
    level,
    startPosition,
    endPosition,
    label: positionRangeLabel(entries, startPosition, endPosition),
    startDate: intervalStartDate(entries, startPosition),
    endDate: intervalEndDate(entries, endPosition),
    widthDays: intervalWidthDays(entries, startPosition, endPosition)
  };
}

function findQuantilePosition(posterior, quantile) {
  const target = Math.min(Math.max(quantile, 0), 1);
  const position = posterior.cdf.findIndex((value) => value >= target);
  return position === -1 ? posterior.cdf.length - 1 : position;
}

function summarizeDecoys(observations) {
  let falsePositives = 0;
  let total = 0;
  for (const observation of observations) {
    if (observation.kind !== "decoy") {
      continue;
    }
    total += 1;
    if (observation.answer === "YES") {
      falsePositives += 1;
    }
  }
  return { falsePositives, total };
}

function posteriorStatus({ entries, credibleInterval95, precise, options }) {
  if (precise) {
    return `95% credible range <= ${options.credibleIntervalDays} days`;
  }
  if (credibleInterval95.startPosition <= 0) {
    return `left-censored before ${entries[0].date}`;
  }
  if (credibleInterval95.endPosition >= entries.length) {
    return `right-censored at ${entries.at(-1).date}`;
  }
  return "max rounds reached";
}

function boundaryLabel(entries, position) {
  if (position <= 0) {
    return `before ${entries[0].date}`;
  }
  if (position >= entries.length) {
    return `${entries.at(-1).date} or later`;
  }
  return `${entries[position - 1].date} to ${entries[position].date}`;
}

function positionRangeLabel(entries, startPosition, endPosition) {
  if (startPosition === endPosition) {
    return boundaryLabel(entries, startPosition);
  }
  return `${intervalStartLabel(entries, startPosition)} to ${intervalEndLabel(entries, endPosition)}`;
}

function intervalStartLabel(entries, position) {
  if (position <= 0) {
    return `before ${entries[0].date}`;
  }
  return entries[position - 1].date;
}

function intervalEndLabel(entries, position) {
  if (position >= entries.length) {
    return `${entries.at(-1).date} or later`;
  }
  return entries[position].date;
}

function intervalStartDate(entries, position) {
  if (position <= 0) {
    return null;
  }
  return entries[position - 1].date;
}

function intervalEndDate(entries, position) {
  if (position >= entries.length) {
    return null;
  }
  return entries[position].date;
}

function latestKnownDate(entries, position) {
  return position > 0 ? entries[position - 1].date : null;
}

function firstUnknownDate(entries, position) {
  return position < entries.length ? entries[position].date : null;
}

function intervalWidthDays(entries, startPosition, endPosition) {
  if (startPosition <= 0 || endPosition >= entries.length) {
    return null;
  }
  return daysBetween(entries[startPosition - 1].date, entries[endPosition].date);
}

function daysBetween(startDate, endDate) {
  const start = new Date(`${startDate}T00:00:00Z`).getTime();
  const end = new Date(`${endDate}T00:00:00Z`).getTime();
  return Math.max(0, Math.round((end - start) / 86_400_000));
}

function betaBinomialLogMarginal(prior, success, failure) {
  return logBeta(prior.alpha + success, prior.beta + failure) - logBeta(prior.alpha, prior.beta);
}

function logBeta(alpha, beta) {
  return logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
}

function logGamma(value) {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7
  ];

  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }

  let x = 0.9999999999998099;
  const shifted = value - 1;
  for (let index = 0; index < coefficients.length; index += 1) {
    x += coefficients[index] / (shifted + index + 1);
  }
  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(x);
}

function parseSoftwareAnswers(content) {
  let parsed = null;
  try {
    parsed = JSON.parse(content);
  } catch {
    parsed = null;
  }

  if (parsed && Array.isArray(parsed.answers)) {
    return {
      answers: parsed.answers.map((answer) => ({
        index: Number.isInteger(answer.index) ? answer.index : null,
        answer: normalizeSoftwareAnswer(answer.answer)
      })).filter((answer) => answer.index !== null)
    };
  }

  return {
    answers: [...String(content).matchAll(/\b(YES|NO|UNKNOWN)\b/giu)].map((match, index) => ({
      index: index + 1,
      answer: normalizeSoftwareAnswer(match[1])
    }))
  };
}

function normalizeSoftwareAnswer(value) {
  const normalized = String(value ?? "").trim().toUpperCase();
  if (normalized === "YES" || normalized === "NO" || normalized === "UNKNOWN") {
    return normalized;
  }
  return "UNKNOWN";
}
