import { ZeusWxBrand } from "./ZeusLogo";

export function Header() {
  return (
    <header className="flex h-14 items-center justify-between border-b border-slate-800/80 bg-slate-950/85 px-4 backdrop-blur-md">
      <h1 className="flex items-center" aria-label="Zeus Wx">
        <ZeusWxBrand />
      </h1>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-950/40 px-3 py-1 text-xs font-semibold text-emerald-400">
          <span className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]" />
          <span>GFS 0.25° Active</span>
        </div>
      </div>
    </header>
  );
}
