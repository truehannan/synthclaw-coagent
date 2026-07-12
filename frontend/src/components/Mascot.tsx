import { useEffect, useState } from "react";

const EYES_OPEN = [
  " █▀▌     ▐▀█",
  "█▄ ▄     ▄ ▄█",
  "  █▄█████▄█",
  "█▀ ▀▀▀▀▀▀▀ ▀█",
  " █▄▌     ▐▄█",
];

const EYES_CLOSED = [
  " █▀▌     ▐▀█",
  "█▄         ▄█",
  "  █████████",
  "█▀ ▀▀▀▀▀▀▀ ▀█",
  " █▄▌     ▐▄█",
];

export default function Mascot({ className = "" }: { className?: string }) {
  const [blink, setBlink] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setBlink(true);
      setTimeout(() => setBlink(false), 150);
    }, 3000 + Math.random() * 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <pre className={`font-mono text-xs leading-tight text-primary select-none ${className}`} aria-hidden="true">
      {(blink ? EYES_CLOSED : EYES_OPEN).join("\n")}
    </pre>
  );
}
