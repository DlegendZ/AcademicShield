// Confirms the transpiled JS models reproduce the Python predictions.
// Run from repo root:  node scripts/validate_parity.mjs
import { readFileSync } from "node:fs";
import { scoreModelA } from "../web/static/models/model_a.js";
import { scoreModelB } from "../web/static/models/model_b.js";

const fx = JSON.parse(readFileSync(new URL("./parity_fixture.json", import.meta.url)));

function maxAbs(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// ---- Model A: probabilities ----
let aMax = 0, aArgmaxMismatch = 0;
const A = fx.model_a;
for (let i = 0; i < A.inputs.length; i++) {
  const got = scoreModelA(A.inputs[i]);
  const exp = A.expected_proba[i];
  aMax = Math.max(aMax, maxAbs(got, exp));
  const ai = got.indexOf(Math.max(...got));
  const ei = exp.indexOf(Math.max(...exp));
  if (ai !== ei) aArgmaxMismatch++;
}

// ---- Model B: GPA ----
let bMax = 0;
const B = fx.model_b;
for (let i = 0; i < B.inputs.length; i++) {
  const got = scoreModelB(B.inputs[i]);
  bMax = Math.max(bMax, Math.abs(got - B.expected_gpa[i]));
}

console.log(`Model A  rows=${A.inputs.length}  max|Δproba|=${aMax.toExponential(3)}  argmax mismatches=${aArgmaxMismatch}`);
console.log(`Model B  rows=${B.inputs.length}  max|Δgpa|=${bMax.toExponential(3)}`);

const okA = aMax < 1e-4 && aArgmaxMismatch === 0;
const okB = bMax < 1e-4;
if (okA && okB) {
  console.log("PARITY OK ✓  JS == Python within 1e-4");
  process.exit(0);
} else {
  console.error("PARITY FAILED ✗");
  process.exit(1);
}
