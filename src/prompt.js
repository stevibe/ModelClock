export function buildMessages({ softwareQuestions = [] }) {
  return [
    {
      role: "system",
      content: [
        "You are a strict software release knowledge evaluator.",
        "Answer from memory only.",
        "Do not browse, use tools, infer from semantic versioning, or assume every plausible version exists.",
        "Some items are plausible but nonexistent decoy versions.",
        "For each item, answer YES only if you know that exact version was actually released.",
        "Answer NO if you know it was not released.",
        "Answer UNKNOWN if you are not sure.",
        "Return only valid JSON."
      ].join(" ")
    },
    {
      role: "user",
      content: [
        "For each software/version pair, had that exact version been released?",
        "No dates are provided. Use release knowledge only.",
        "Treat plausible-looking versions carefully; decoy versions may be mixed in.",
        "",
        softwareQuestions.map((question) => `${question.index}. ${question.prompt}`).join("\n"),
        "",
        "Return JSON with this shape:",
        '{"answers": [{"index": number, "answer": "YES or NO or UNKNOWN"}], "reason": "short explanation"}'
      ].join("\n")
    }
  ];
}
