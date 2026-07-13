const FACE = [
  "  /\\_/\\  ",
  " ( o.o ) ",
  "  > ^ <  ",
];

export default function Mascot({ className = "", size = "md" }: { className?: string; size?: "sm" | "md" }) {
  const textSize = size === "sm" ? "text-[8px]" : "text-xs";
  return (
    <pre className={`font-mono ${textSize} leading-tight text-primary select-none ${className}`} aria-hidden="true">
      {FACE.join("\n")}
    </pre>
  );
}
