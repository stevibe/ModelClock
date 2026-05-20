#!/usr/bin/env node

import { main } from "../src/cli.js";

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  if (process.env.MODEL_CLOCK_DEBUG === "1") {
    console.error(error);
  }
  process.exitCode = 1;
});
