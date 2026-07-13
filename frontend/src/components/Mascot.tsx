const EYES_OPEN = [
  " █▀▌     ▐▀█",
  "█▄ ▄     ▄ ▄█",
  "  █▄█████▄█",
  "█▀ ▀▀▀▀▀▀▀ ▀█",
  " █▄▌     ▐▄█",
];

export default function Mascot({ className = "" }: { className?: string }) {
  return (
    <pre className={`font-mono text-[10px] leading-none text-primary select-none whitespace-pre ${className}`} aria-hidden="true">
      {EYES_OPEN.join("\n")}
    </pre>
  );
}
