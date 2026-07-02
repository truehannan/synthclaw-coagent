// SYNTHCLAW block-character wordmark (proper W letter)
export const SYNTHCLAW_BLOCK = [
  "\u2588\u2580\u2580 \u2588\u2584\u2588 \u2588\u2584 \u2588 \u2580\u2588\u2580 \u2588 \u2588 \u2588\u2580\u2580 \u2588   \u2588\u2580\u2588 \u2588 \u2588 \u2588 \u2588",
  "\u2580\u2580\u2588  \u2588  \u2588 \u2580\u2588  \u2588  \u2588\u2580\u2588 \u2588   \u2588   \u2588\u2580\u2588 \u2588\u2584\u2588\u2584\u2588 \u2588",
  "\u2580\u2580\u2580  \u2580  \u2580  \u2580  \u2580  \u2580 \u2580 \u2580\u2580\u2580 \u2580\u2580\u2580 \u2580 \u2580 \u2580 \u2580 \u2580 \u2580",
];

// Scorpion/claw icon — 2-frame animation (alternates every 1s)
// Frame 1: tail slightly left
export const ICON_FRAME_1 = [
  "   \u2588\u2580\u2588",
  "     \u2588",
  "  \u2588\u2588\u2588\u2588\u2588\u2588\u2588",
  "\u2588\u2580\u2588\u2584\u2588\u2588\u2588\u2584\u2588\u2580\u2588",
  "\u2588\u2584       \u2584\u2588",
  "\u2588\u2584\u2584     \u2584\u2584\u2588",
];

// Frame 2: tail shifted right
export const ICON_FRAME_2 = [
  "     \u2588\u2580\u2588",
  "     \u2588",
  "  \u2588\u2588\u2588\u2588\u2588\u2588\u2588",
  "\u2588\u2580\u2588\u2584\u2588\u2588\u2588\u2584\u2588\u2580\u2588",
  "\u2588\u2584       \u2584\u2588",
  "\u2588\u2584\u2584     \u2584\u2584\u2588",
];

// Legacy export
export const SYNTHCLAW_ASCII = "\n  " + SYNTHCLAW_BLOCK.join("\n  ") + "\n";
