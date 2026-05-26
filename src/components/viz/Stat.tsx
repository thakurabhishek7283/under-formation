export function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="viz-stat">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
    </div>
  );
}
