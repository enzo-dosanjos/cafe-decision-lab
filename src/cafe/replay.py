"""Capture and render synchronized end-of-minute café rollouts."""

import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

from .contracts import build_catalogue
from .simulation.simulator import run_day


NAMES = {"fixed": "Fixed", "adaptive": "Adaptive", "mlp": "MLP", "rnn": "Vanilla RNN", "lstm": "LSTM"}
COLORS = ["#345D79", "#8A5632", "#5B4B85", "#23765B"]


def clock(value: str) -> int:
    """Convert a 24-hour clock label to minutes since midnight."""
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError(f"expected HH:MM, got {value!r}")

    return int(value[:2]) * 60 + int(value[3:])


def capture(day, managers, policy, config, opening: str, seed: int, output: Path) -> None:
    """Run four selected managers against one schedule and save their minute traces."""
    from .models.lstm import LSTMPredictor

    catalogue = build_catalogue(config)
    schedule = hashlib.sha256(json.dumps(asdict(day), sort_keys=True).encode()).hexdigest()
    competitor = asdict(catalogue[config.simulation["competitor_action_id"]])
    recurrent = managers[-1].predictor
    kind = "lstm" if isinstance(recurrent, LSTMPredictor) else recurrent.kind
    bundle = dict(version=1, mode="evaluation", scenario=day.scenario_id, opening=opening,
                  length=config.simulation["day_minutes"], period=config.simulation["period_minutes"],
                  products=[dict(id=i, label=p["name"].title(), base_price=p["base_price"])
                            for i, p in enumerate(config.products)],
                  scale=max(max(a.stock) for a in catalogue), max_staff=max(a.staff for a in catalogue),
                  schedule=schedule, competitor=competitor, seed=seed, runs=[])

    for name, manager in zip(("fixed", "adaptive", "mlp", kind), managers):
        rows = []
        result = run_day(day, policy, manager, config, trajectory_id=name, snapshots=rows)

        for record in result.periods:
            last = rows[min((record.period_index + 1) * bundle["period"], len(rows)) - 1]

            if last["left"] != record.abandonment or last["stock"] != record.waste.tolist():
                raise ValueError(f"{name}: inconsistent period {record.period_index}")

        bundle["runs"].append(dict(policy=name, scenario=day.scenario_id, schedule=schedule,
                                   competitor=competitor, minutes=rows))

    validate(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle))


def demo() -> dict:
    """Build the synthetic reference and a short transition around minute ten."""
    b = dict(version=1, mode="demo", scenario="demo", schedule="demo", competitor=13,
             opening="12:00", length=13, period=11, max_staff=3, scale=16, runs=[],
             products=[dict(id=i, label=n, base_price=p) for i, (n, p) in
                       enumerate(zip(("Espresso", "Latte", "Pastry"), (22, 30, 18)))])
    values = [(0.9,7,2,[4,1,6],2),(1.1,4,3,[6,3,8],1),
              (0.9,3,3,[4,3,6],0),(0.9,5,2,[2,0,4],1)]

    for name, (price, queue, staff, stock, left) in zip(NAMES, values):
        rows = [dict(minute=m, period=m//11, price=price, queue=queue if m < 12 else 0,
                     staff=staff if m < 11 else 1, stock=stock if m < 11 else [10]*3,
                     left=left if m < 11 else (0 if m == 11 else queue)) for m in range(13)]
        b["runs"].append(dict(policy=name, schedule="demo", scenario="demo", competitor=13, minutes=rows))

    return b


def validate(b: dict) -> None:
    """Reject malformed bundles before producing any frames."""
    def count(value, minimum=0, maximum=math.inf):
        """Check that a JSON count is an integer within its configured bounds."""

        return type(value) is int and minimum <= value <= maximum

    clock(b["opening"])

    if b["version"] != 1 or b["mode"] not in ("demo", "evaluation"):
        raise ValueError("unsupported replay version or mode")

    if not all(count(b[k], 1) for k in ("length", "period", "max_staff", "scale")):
        raise ValueError("invalid replay dimensions")

    if [p["id"] for p in b["products"]] != [0,1,2] or any(not p["label"] or p["base_price"] <= 0 for p in b["products"]):
        raise ValueError("invalid ordered product mapping")

    names = [r["policy"] for r in b["runs"]]

    if names not in (["fixed","adaptive","mlp","rnn"], ["fixed","adaptive","mlp","lstm"]):
        raise ValueError("expected fixed, adaptive, mlp and recurrent policy traces")

    for r in b["runs"]:
        name = r["policy"]

        if any(r[k] != b[k] for k in ("scenario","schedule","competitor")):
            raise ValueError(f"{name}: mismatched scenario, schedule or competitor")

        if len(r["minutes"]) != b["length"]:
            raise ValueError(f"{name}: incomplete or duplicate minute coverage")

        for m, row in enumerate(r["minutes"]):
            if not (count(row["minute"],m,m) and count(row["period"],m//b["period"],m//b["period"])
                    and count(row["staff"],0,b["max_staff"]) and count(row["queue"]) and count(row["left"])
                    and len(row["stock"]) == 3 and all(count(v,0,b["scale"]) for v in row["stock"])
                    and math.isfinite(row["price"]) and row["price"] > 0
                    and (b["mode"] == "demo" or count(row["action"]))):
                raise ValueError(f"{name}: invalid snapshot at minute {m}")


def frame(b: dict, minute: int):
    """Draw one full-HD frame with fixed pixel coordinates and a shared clock."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, FancyBboxPatch

    fig = Figure(figsize=(19.2,10.8), dpi=100, facecolor="#F6F2E9")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0,0,1,1), xlim=(0,1920), ylim=(1080,0))
    ax.axis("off")
    ink, muted, alert = "#172D27", "#51635B", "#AA422B"

    def box(x,y,w,h,r,color,edge="none"):
        """Draw a rounded rectangle in frame coordinates."""
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=color, ec=edge, linewidth=1))

    def label(x,y,value,size,color=ink,bold=False,align="left"):
        """Place text by its top edge with pixel-sized lettering."""
        ax.text(x,y,str(value),va="top",ha=align,fontsize=size*.72,color=color,
                fontfamily="DejaVu Sans",weight="bold" if bold else "normal")

    def dot(x,y,r,color):
        """Place a filled circular icon."""
        ax.add_patch(Circle((x,y),r,color=color,lw=0))

    label(64,43,"CAFÉ DECISION LAB",18,muted,True)
    label(64,75,"Four policies, one lunchtime rush",57,ink,True)
    label(66,163,"Same arrival schedule  •  Competitor settings fixed across runs",23,muted)
    box(1612,55,243,122,16,"#18382E")
    label(1734,80,"SIMULATION CLOCK",17,"#C6D1CB",True,"center")
    time = (clock(b["opening"]) + minute) % 1440
    label(1734,106,f"{time//60:02}:{time%60:02}",48,"white",True,"center")

    for i, run in enumerate(b["runs"]):
        x,y = 64+912*(i%2),215+389*(i//2)
        c, row = COLORS[i], run["minutes"][minute]
        box(x,y,879,365,20,"#FFFDF8","#D8DED4")
        box(x+22,y+29,7,38,3,c)
        label(x+47,y+25,NAMES[run["policy"]],36,ink,True)
        label(x+842,y+36,f"Price  {row['price']:.2f}×",22,c,True,"right")
        box(x+29,y+85,820,1,0,"#D8DED4")
        box(x+581,y+107,1,220,0,"#D8DED4")

        for offset, title in [(33,"QUEUE"),(398,"STAFF"),(611,"STOCK LEFT")]:
            label(x+offset,y+109,title,18,muted,True)

        box(x+29,y+146,336,71,12,"#F1F5F0")
        box(x+397,y+146,161,71,12,"#EAF0E9")

        for j in range(min(8,row["queue"])):
            dot(x+55+34*j,y+181,12,c)

        if row["queue"] > 8:
            label(x+359,y+170,f"+{row['queue']-8}",16,c,True,"right")

        label(x+371,y+158,"›",38,muted)
        slots = min(5,b["max_staff"])

        for j in range(slots):
            cx = x+397+161*(j+.5)/slots
            fill = c if j < row["staff"] else "#C6D1CB"
            dot(cx,y+169,7,fill)
            box(cx-12,y+180,24,18,5,fill)

        label(x+31,y+233,f"{row['queue']} waiting",22,ink,True)
        label(x+399,y+233,f"{row['staff']} on duty",22,ink,True)
        label(x+31,y+291,f"{row['left']} left this period",20,alert if row["left"] else muted)

        for j,(product,n) in enumerate(zip(b["products"],row["stock"])):
            yy = y+151+65*j
            label(x+611,yy,product["label"],20)
            label(x+845,yy,n,20,ink if n else alert,True,"right")
            box(x+611,yy+31,234,10,3,"#E7ECE4")

            if n:
                width = 234*n/b["scale"]
                box(x+611,yy+31,width,10,min(3,width/2),c)

    dot(76,1034,10,COLORS[0])
    label(98,1022,"Each dot is one waiting customer",18,muted)
    label(1855,1022,"Illustrative frame with synthetic values" if b["mode"]=="demo" else b["scenario"],
          18,muted,align="right")

    return canvas


def render(args) -> None:
    """Render a PNG or stream inclusive minute frames to a 30 fps H.264 encoder."""
    try:
        b = demo() if args.demo else json.loads(args.input.read_text())
        validate(b)
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{args.input or 'demo'}: {error}") from error

    still = args.output.suffix.lower() == ".png"

    if still and (not args.at or args.start or args.end):
        raise ValueError("PNG requires --at HH:MM")
    elif not still and (args.output.suffix.lower() != ".mp4" or args.at or not args.start or not args.end):
        raise ValueError("MP4 requires --start and --end HH:MM")

    start, end = [(clock(t)-clock(b["opening"])) % 1440 for t in
                  ((args.at,args.at) if still else (args.start,args.end))]

    if not 0 <= start <= end < b["length"] or not 0 < args.speed <= 30:
        raise ValueError("interval outside replay coverage or speed outside (0, 30]")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if still:
        frame(b,start).print_png(args.output)

        return

    command = ["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgba","-s","1920x1080",
               "-r","30","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(args.output)]

    with subprocess.Popen(command,stdin=subprocess.PIPE) as proc:
        for i in range(end-start+1):
            canvas = frame(b,start+i)
            canvas.draw()
            repeats = math.ceil((i+1)*30/args.speed)-math.ceil(i*30/args.speed)

            for _ in range(repeats):
                proc.stdin.write(canvas.buffer_rgba())

        proc.stdin.close()

        if proc.wait():
            raise RuntimeError("FFmpeg replay encoding failed")
