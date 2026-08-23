"""guestshots — pick flattering, active guest screenshots from podcast YouTube videos."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from . import pipeline as P

app = typer.Typer(add_completion=False, help=__doc__)
con = Console()
ROOT = Path(__file__).resolve().parent.parent
HOSTS_DIR = ROOT / "hosts"
OUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"


def _progress(stage: str, done: int = 0, total: int = 0, msg: str = "") -> None:
    if msg:
        con.print(f"  [{stage}] {msg}")
    elif total and (done == total or done % 500 == 0):
        con.print(f"  [{stage}] {done}/{total}")


@app.command()
def main(
    urls: list[str] = typer.Argument(..., help="YouTube URLs"),
    n: int = typer.Option(5, "-n", "--shots", help="Screenshots per video"),
    fps: float = typer.Option(1.0, help="Analysis frames per second"),
    min_gap: float = typer.Option(20.0, help="Min seconds between picked shots"),
    host_dir: Path = typer.Option(HOSTS_DIR, help="Folder with reference photos of the host"),
    host_sim: float = typer.Option(0.45, help="Cosine similarity to call an identity 'host'"),
    guest_id: int | None = typer.Option(None, help="Force identity id as guest (see identities/ crops)"),
    solo_only: bool = typer.Option(False, help="Only frames where the guest is alone in the picture"),
    llm: bool = typer.Option(True, help="Re-rank top candidates with a vision LLM via OpenRouter (needs OPENROUTER_API_KEY)"),
    llm_model: str = typer.Option("openai/gpt-5.4-mini", help="OpenRouter model id for re-ranking"),
    llm_pool: int | None = typer.Option(None, help="Candidates per final shot sent to the LLM (n × llm_pool, default 4)"),
    profile: str | None = typer.Option(None, help="Option preset: 'portrait' (quote-graphic backgrounds)"),
    criteria: str = typer.Option("", help="Extra free-text requirements for the LLM stage, e.g. 'no glasses reflections'"),
    max_face_bottom: float | None = typer.Option(None, help="Drop shots whose face box bottom is below this fraction of frame height (e.g. 0.55)"),
    require_gaze_camera: bool | None = typer.Option(None, "--require-gaze-camera", help="LLM gate: guest must look into the camera"),
):
    """Download each video, find the guest (not the host), and save N flattering + active screenshots."""
    opts = P.Options.build(profile, n=n, fps=fps, min_gap=min_gap, host_sim=host_sim, guest_id=guest_id,
                           solo_only=solo_only or None, llm=llm, llm_model=llm_model, llm_pool=llm_pool,
                           criteria=criteria, max_face_bottom=max_face_bottom, require_gaze_camera=require_gaze_camera)
    refs = [p for p in sorted(host_dir.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    host_embs = P.host_embeddings(refs)
    if host_embs:
        con.print(f"host refs: {len(host_embs)} photo(s) from {host_dir}")
    else:
        con.print(f"[yellow]no host reference photos in {host_dir}[/] — "
                  "will infer host as the identity shared across videos (needs ≥2 URLs).")

    analyses = []
    for u in urls:
        a = P.analyze(u, CACHE_DIR, fps, _progress)
        con.print(f"[bold]{a.vid.title}[/]: {len(a.obs)} faces → {len(a.idents)} identities: "
                  + ", ".join(f"#{i.id}×{i.count}" for i in a.idents))
        analyses.append(a)
    P.mark_hosts(analyses, host_embs, host_sim)

    for a in analyses:
        out = OUT_DIR / f"{a.vid.id}_{P.slug(a.vid.title)}"
        try:
            res = P.produce(a, opts, out, _progress)
        except P.PipelineError as e:
            con.print(f"[red]{a.vid.title}: {e}[/] (crops in {out / 'identities'})")
            continue
        if res.report.get("llm_usage"):
            u = res.report["llm_usage"]
            con.print(f"  LLM usage: {u['prompt_tokens']} in / {u['completion_tokens']} out"
                      + (f" → ${u['cost']:.4f}" if u.get("cost") is not None else ""))
        con.print(f"  [green]✓ {len(res.shots)} shots → {res.out_dir / 'shots'}[/]")
        for r in res.report["shots"]:
            p = r["parts"]
            note = (f"[{p['llm_eyes']}/{p.get('llm_gaze', '?')}] {p['llm_note']}" if p.get("llm_note")
                    else f"eyes {p['eyes']:.2f} smile {p['smile']:.2f} speak {p['speaking']:.2f} motion {p['motion']:.2f}")
            con.print(f"    {r['file']}  {'solo' if r['solo'] else 'two-shot'}  {note}")


if __name__ == "__main__":
    app()
