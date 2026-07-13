/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "#0a0a0a",
        foreground: "#fafafa",
        card: "#111111",
        "card-hover": "#1a1a1a",
        border: "#222222",
        muted: "#888888",
        primary: "#ef4444",
        "primary-hover": "#f87171",
        "primary-dim": "rgba(239, 68, 68, 0.12)",
        danger: "#ff3b30",
        success: "#30d158",
        warning: "#ff9f0a",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Geist Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      borderRadius: {
        sm: "4px",
      },
    },
  },
  plugins: [],
};
