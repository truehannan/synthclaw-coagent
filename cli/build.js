import { build } from "esbuild";

await build({
  entryPoints: ["src/index.js"],
  bundle: true,
  platform: "node",
  target: "node18",
  format: "esm",
  outfile: "dist/index.js",
  banner: { js: "#!/usr/bin/env node\n" },
  external: ["shelljs"],
});

console.log("✓ Built dist/index.js");
