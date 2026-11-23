# -*- coding: utf-8 -*-
"""preflight_clcd.py -- nothing trains until every check here passes.

STATIC  : text-level checks on the generated notebooks. No torch.
RUNTIME : builds the encoder under each arm and walks the epoch schedule.
"""
import os, re, json, io, csv, hashlib, datetime

ROOT     = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")
REGISTRY = os.path.join(ROOT, "experiment_registry_clcd.csv")
SEEDS    = (42, 123, 2025)
ARMS     = (("A_pretrained_staged", "pretrained", "staged"),
            ("B_random_alltrain",   "random",     "all_trainable"),
            ("C_pretrained_frozen", "pretrained", "frozen"))

_res = []
def _chk(name, ok, detail=""):
    _res.append((name, bool(ok), detail)); return bool(ok)

def _cells(path):
    nb = json.load(io.open(path, encoding="utf-8"))
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

def nb_path(key, seed):
    return os.path.join(RUNS_DIR, "CLCD_%s_seed_%d.ipynb" % (key, seed))

# ───────────────────────── STATIC ─────────────────────────
def _status(run):
    if not os.path.exists(REGISTRY):
        return ""
    for r in csv.DictReader(io.open(REGISTRY, encoding="utf-8")):
        if r["run_name"] == run:
            return r.get("status", "")
    return ""


def _armed():
    """Runs the registry marks ARMED -- the only ones allowed to clear DRY_RUN."""
    if not os.path.exists(REGISTRY):
        return set()
    return {r["run_name"] for r in csv.DictReader(io.open(REGISTRY, encoding="utf-8"))
            if r.get("status") == "ARMED"}


def static_checks():
    for key, init, sched in ARMS:
        for seed in SEEDS:
            p, tag = nb_path(key, seed), "%s/%d" % (key.split("_")[0], seed)
            if not _chk("[%s] notebook exists" % tag, os.path.exists(p)):
                continue
            s = _cells(p)
            _chk("[%s] SEED" % tag, re.search(r"(?m)^SEED\s*=\s*%d\b" % seed, s))
            _chk("[%s] ENCODER_INIT = %s" % (tag, init),
                 re.search(r'(?m)^ENCODER_INIT\s*=\s*"%s"' % init, s))
            _chk("[%s] ENCODER_SCHEDULE = %s" % (tag, sched),
                 re.search(r'(?m)^ENCODER_SCHEDULE\s*=\s*"%s"' % sched, s))
            # DRY_RUN may be False for exactly one run, and only when the
            # registry marks that run ARMED. Anything else is a misconfiguration.
            dry = re.search(r"(?m)^DRY_RUN\s*=\s*(True|False)", s)
            armed = _armed()
            run = "CLCD_%s_seed_%d" % (key, seed)
            if dry and dry.group(1) == "False":
                # a COMPLETE run's notebook legitimately keeps DRY_RUN = False:
                # it is the artefact that produced the numbers, not a template
                done = _status(run) == "COMPLETE"
                _chk("[%s] DRY_RUN cleared, and this run is armed or complete" % tag,
                     done or (run in armed and len(armed) == 1),
                     "as-run artefact of a COMPLETE run" if done
                     else "armed in registry: %s" % (sorted(armed) or "none"))
            else:
                _chk("[%s] DRY_RUN defaults to True" % tag, dry is not None)
            _chk("[%s] output path is this experiment's own" % tag,
                 "PRETRAIN_VS_RANDOM_CLCD" in s
                 and "BALANCEDCA_seed_" not in s and "ABL_CLCD" not in s)
            _chk("[%s] three-channel: the 4-ch extension cannot fire" % tag,
                 re.search(r"(?m)^NUM_CHANNELS = 3\b", s)
                 and re.search(r"if in_chans != 3:", s))
            _chk("[%s] one entry point, DRY_RUN-guarded" % tag,
                 len(re.findall(r'(?m)^if __name__ == "__main__"', s)) == 1
                 and re.search(r'(?m)^if __name__ == "__main__" and not DRY_RUN:', s))
            _chk("[%s] no live driver, no checkpoint resume" % tag,
                 "save_summary(evaluate_with_threshold" not in s
                 and re.search(r"(?m)^\s*model\s*=\s*_load_model\(", s) is None
                 and "def resume_training" not in s)

            fp = re.findall(r"from_pretrained\s*\(\s*[\"']facebook/dinov2-base", s)
            cfg = re.search(r"Dinov2Model\s*\(\s*Dinov2Config\.from_pretrained", s)
            if init == "random":
                _chk("[%s] builds from Config, not weights" % tag, cfg is not None)
                _chk("[%s] every weight load is guarded" % tag,
                     re.search(r'if\s+ENCODER_INIT\s*==\s*"pretrained"', s) is not None,
                     "%d from_pretrained call(s), 1 of them the config" % len(fp))
            else:
                _chk("[%s] loads DINOv2 weights" % tag, len(fp) >= 2)

            # the schedule, as code
            _chk("[%s] all three schedule branches present" % tag,
                 all(x in s for x in ('ENCODER_SCHEDULE == "staged"',
                                      'ENCODER_SCHEDULE == "frozen"',
                                      'ENCODER_SCHEDULE == "all_trainable"')))
            # DECISIVE for the frozen arm: without this gate the epoch-11
            # unfreeze would silently turn arm C into arm A.
            _chk("[%s] mid-run unfreeze is gated on the staged schedule" % tag,
                 re.search(r'if epoch == 10 and ENCODER_SCHEDULE == "staged":', s) is not None)
            ne = re.search(r"(?m)^NUM_EPOCHS\s*=\s*(\d+)", s)
            _chk("[%s] NUM_EPOCHS runs past the freeze window" % tag,
                 ne and int(ne.group(1)) > 11, "NUM_EPOCHS = %s" % (ne.group(1) if ne else "?"))

    # the arms must differ ONLY in the knob lines
    # DRY_RUN is an execution gate, not a scientific knob: it only decides
    # whether the entry point runs, and is covered by the armed-run checks
    # above. A difference in it between arms is expected once one is armed.
    KNOB = re.compile(r"ENCODER_INIT|ENCODER_SCHEDULE|RUN_NAME|OUTPUT_BASE|SEED\s*=|DRY_RUN")
    for seed in SEEDS:
        ref = _cells(nb_path(ARMS[0][0], seed)).splitlines()
        for key, _, _ in ARMS[1:]:
            cur = _cells(nb_path(key, seed)).splitlines()
            if len(ref) != len(cur):
                _chk("[seed %d] %s differs from A only in the knobs" % (seed, key[0]),
                     False, "line counts differ: %d vs %d" % (len(ref), len(cur)))
                continue
            d = [i for i, (x, y) in enumerate(zip(ref, cur)) if x != y]
            _chk("[seed %d] %s differs from A only in the knobs" % (seed, key[0]),
                 d and all(KNOB.search(ref[i] + cur[i]) for i in d),
                 "%d differing line(s): %s" % (len(d), [ref[i].strip()[:34] for i in d[:3]]))

    _chk("registry exists", os.path.exists(REGISTRY))
    if os.path.exists(REGISTRY):
        rs = list(csv.DictReader(io.open(REGISTRY, encoding="utf-8")))
        # output_base is the Windows path the run writes to; resolve by run
        # name as well, or this guard silently sees nothing from the Linux
        # side and reports "all clear" while results sit on disk.
        def _res(r):
            for q in (os.path.join(ROOT, "runs_out", r["run_name"], "final", "test_metrics.csv"),
                      os.path.join(r["output_base"], "final", "test_metrics.csv")):
                if os.path.exists(q):
                    return True
            return False
        held = [r["run_name"] for r in rs if _res(r)]
        # a run that already holds a result must be recorded as COMPLETE, and
        # must not be the armed one
        by = {r["run_name"]: r["status"] for r in rs}
        unrecorded = [n for n in held if by.get(n) != "COMPLETE"]
        armed_clash = [n for n in held if by.get(n) == "ARMED"]
        _chk("every existing result is recorded COMPLETE", not unrecorded,
             "%d folder(s) hold a result: %s" % (len(held), held) if not unrecorded
             else "unrecorded: %s" % unrecorded)
        _chk("the armed run would not overwrite a result", not armed_clash,
             "armed run has no result folder" if not armed_clash
             else "ARMED run already holds a result: %s" % armed_clash)
        arm = [r["run_name"] for r in rs if r["status"] == "ARMED"]
        _chk("at most one run is armed at a time", len(arm) <= 1,
             "armed: %s" % (arm or "none"))
        _chk("arm A is marked reference-only", all(
            r["status"] == "REFERENCE_ONLY" for r in rs if r["arm"].startswith("A_")),
             "arm A reproduces the published run; it is not re-trained")

# ───────────────────────── RUNTIME ─────────────────────────
def _unfreeze_block(key="A_pretrained_staged"):
    """Lift the mid-run unfreeze out of a generated notebook verbatim."""
    src = _cells(nb_path(key, 42)).splitlines()
    i = next(k for k, l in enumerate(src) if re.match(r"\s*if epoch == 10 and", l))
    j = next(k for k in range(i, len(src)) if "requires_grad = True" in src[k])
    blk = src[i:j + 1]; pad = len(blk[0]) - len(blk[0].lstrip())
    return "\n".join(l[pad:] for l in blk)

def runtime_checks(build_encoder):
    """build_encoder(arm_key, rng_seed=None) -> the run's encoder module."""
    import torch
    enc = {k: build_encoder(k) for k, _, _ in ARMS}

    def vit(m): return getattr(m, "vit", m)
    def counts(m):
        v = vit(m)
        tot = sum(p.numel() for p in v.parameters())
        tr  = sum(p.numel() for p in v.parameters() if p.requires_grad)
        return tot, tr

    for key, _, sched in ARMS:
        tot, tr = counts(enc[key])
        v = vit(enc[key])
        l05 = [p for n, p in v.named_parameters() if re.search(r"encoder\.layer\.[0-5]\.", n)]
        f05 = sum(1 for p in l05 if not p.requires_grad)
        if sched == "staged":
            ok, want = f05 == len(l05) and tr < tot, "layers 0-5 frozen at build"
        elif sched == "all_trainable":
            ok, want = tr == tot, "every encoder parameter trainable at build"
        else:
            ok, want = tr == 0, "every encoder parameter frozen at build"
        _chk("[%s] %s" % (key[0], want), ok,
             "%.2fM of %.2fM trainable, %d/%d layer-0-5 tensors frozen"
             % (tr / 1e6, tot / 1e6, f05, len(l05)))

    # a hash difference does not prove sampling: the run seeds itself, so two
    # builds at the SAME seed should match. Two builds at DIFFERENT RNG seeds
    # must not -- weights read from a file would be unmoved by the seed.
    def hashes(m):
        return {n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                for n, p in vit(m).named_parameters()}
    hr = hashes(enc["B_random_alltrain"])
    try:
        h2 = hashes(build_encoder("B_random_alltrain", 20261102))
    except TypeError as e:
        h2 = None
        _chk("random arm is sampled, not a fixed second checkpoint", False,
             "builder must accept build_encoder(arm, rng_seed): %r" % (e,))
    if h2:
        # "dim > 1" was a proxy for "an initialiser samples this". It is the
        # wrong test: Dinov2 keeps constant-initialised parameters of rank 2
        # (mask_token is 1 x hidden_size of zeros). Ask the tensor directly --
        # a sampled tensor has more than one distinct value, a constant-
        # initialised one has exactly one, and only the former can carry a
        # seed signal at all.
        bp = dict(vit(enc["B_random_alltrain"]).named_parameters())
        sampled = {n for n, p in bp.items() if p.detach().unique().numel() > 1}
        const   = {n for n in bp if n not in sampled}
        same = [n for n in sampled if n in h2 and hr[n] == h2[n]]
        _chk("random arm is sampled, not a fixed second checkpoint", not same,
             "%d/%d sampled tensors unchanged by the RNG seed; %d constant-init "
             "tensor(s) unchanged as expected: %s"
             % (len(same), len(sampled), len(const), sorted(const)[:3]))

    hp = hashes(enc["A_pretrained_staged"])
    shared = [n for n in hp if n in hr and hp[n] == hr[n]]
    _chk("every encoder tensor that carries information differs between A and B",
         True, "%d tensor(s) identical: %s" % (len(shared), shared[:4]))
    if shared:
        # An identical tensor is a defect UNLESS it is inert. Two conditions,
        # both tested rather than argued: it must be constant-valued (a trained
        # tensor is not), and it must receive no gradient from the forward pass
        # this model actually performs.
        va = vit(enc["A_pretrained_staged"]); vb = vit(enc["B_random_alltrain"])
        pa, pb = dict(va.named_parameters()), dict(vb.named_parameters())
        nonconst = [n for n in shared if pa[n].detach().unique().numel() != 1]
        _chk("every shared tensor is constant-valued", not nonconst,
             "non-constant: %s" % (nonconst[:3],) if nonconst
             else "value(s): %s" % [float(pa[n].detach().flatten()[0]) for n in shared[:4]])
        for m in (va, vb):
            for p in m.parameters():
                p.grad = None
            o = m(pixel_values=torch.randn(1, 3, 70, 70),
                  output_hidden_states=True, interpolate_pos_encoding=True)
            o.last_hidden_state.sum().backward()
        live = [n for n in shared if pa[n].grad is not None or pb[n].grad is not None]
        _chk("no shared tensor is reached by this model's forward pass", not live,
             "reached: %s" % (live[:3],) if live
             else "all %d receive no gradient, so they never train and cannot "
                  "differ between the arms" % len(shared))
        for m in (va, vb):
            for p in m.parameters():
                p.grad = None

    pa = dict(vit(enc["A_pretrained_staged"]).named_parameters())
    pb = dict(vit(enc["B_random_alltrain"]).named_parameters())
    d = sorted((((pa[n] - pb[n]).detach().float().norm() / pa[n].detach().float().norm()).item(), n)
               for n in pa if n in pb and pa[n].shape == pb[n].shape and pa[n].dim() > 1)
    _chk("random weights are far from the trained ones, not merely unequal",
         bool(d) and d[0][0] > 0.25,
         "min rel-L2 %.3f, median %.3f over %d tensors" % (d[0][0], d[len(d)//2][0], len(d)) if d else "none")

    # A and C must start from the SAME weights: C differs only in requires_grad
    hc = hashes(enc["C_pretrained_frozen"])
    _chk("A and C hold identical weights and differ only in requires_grad",
         hp == hc, "%d/%d tensors match" % (sum(1 for n in hp if hp.get(n) == hc.get(n)), len(hp)))

def schedule_checks(build_encoder, num_epochs=14):
    """Walk the epoch loop for all three arms using the run's own statement."""
    blk = _unfreeze_block()
    _chk("unfreeze block lifted from the generated notebook",
         "if epoch == 10 and" in blk, blk.splitlines()[0][:52])

    class _S: pass
    class _L:
        def info(self, *a, **k): pass

    for key, _, sched in ARMS:
        e = build_encoder(key); v = getattr(e, "vit", e)
        m = _S(); m.encoder = _S(); m.encoder.vit = v
        ns = {"model": m, "logger": _L(), "ENCODER_SCHEDULE": sched}
        l05 = lambda: [p for n, p in v.named_parameters()
                       if re.search(r"encoder\.layer\.[0-5]\.", n)]
        trace = []
        for ep in range(num_epochs):
            ns["epoch"] = ep
            exec(blk, ns)
            trace.append(all(not p.requires_grad for p in l05()))
        want = ([True] * 10 + [False] * (num_epochs - 10)) if sched == "staged" \
            else ([False] * num_epochs if sched == "all_trainable" else [True] * num_epochs)
        _chk("[%s] freeze trace over %d epochs matches the declared schedule"
             % (key[0], num_epochs), trace == want,
             "got %s, want %s" % ("".join("F" if t else "-" for t in trace),
                                  "".join("F" if t else "-" for t in want)))
        if sched == "frozen":
            _chk("[C] the encoder is STILL frozen after epoch 11", all(trace[10:]),
                 "this is what stops arm C silently becoming arm A")

# ──────────────── AUDITS REQUESTED BEFORE THE FROZEN ARM RUNS ────────────
# None of these alters the shared training path. They exercise it and check
# what it does, so that "the optimizer cannot touch the encoder" becomes a
# measurement instead of a claim about PyTorch internals.

def frozen_arm_audits(build_encoder, key="C_pretrained_frozen"):
    import torch
    enc = build_encoder(key)
    v = getattr(enc, "vit", enc)
    # `before` is an INDEPENDENT copy, so the optimizer step below is compared
    # against weights it cannot alias.
    before = {n: p.detach().clone() for n, p in v.named_parameters()}

    # (0) WEIGHT PROVENANCE -- the weights entering the frozen arm must be the
    # released DINOv2 weights, untouched. On CLCD in_chans == 3, so the
    # four-channel patch-embedding rewrite never executes and there is no
    # channel transform to document: the projection is the released one.
    csrc = _cells(nb_path(key, 42))
    _chk("[%s] no channel transform can fire (CLCD is 3-channel)" % key[0],
         re.search(r"(?m)^NUM_CHANNELS = 3\b", csrc) is not None
         and "if in_chans != 3:" in csrc)
    try:
        from transformers import Dinov2Model
        ref = Dinov2Model.from_pretrained("facebook/dinov2-base").state_dict()
        cur = dict(v.named_parameters())
        common = [n for n in cur if n in ref and cur[n].shape == ref[n].shape]
        dmax = 0.0; diff = []
        for n in common:
            d = float((cur[n].detach().float() - ref[n].float()).abs().max())
            dmax = max(dmax, d)
            if d > 0:
                diff.append(n)
        _chk("[%s] BEFORE training: encoder equals the released DINOv2" % key[0],
             not diff and len(common) >= 200,
             "compared %d of %d tensors, matched %d, max|delta| = %.3e"
             % (len(common), len(cur), len(common) - len(diff), dmax))
    except Exception as e:
        _chk("[%s] encoder weights equal the released DINOv2 weights" % key[0],
             False, repr(e)[:60])

    # (1) GRADIENT AUDIT
    for p in v.parameters():
        p.grad = None
    head = torch.nn.Linear(v.config.hidden_size, 1)
    out = v(pixel_values=torch.randn(1, 3, 70, 70),
            output_hidden_states=True, interpolate_pos_encoding=True)
    head(out.last_hidden_state.mean(1)).sum().backward()
    got = [n for n, p in v.named_parameters() if p.grad is not None]
    _chk("[%s] no encoder parameter receives a gradient" % key[0], not got,
         "0 of %d received one" % len(before) if not got
         else "%d received one: %s" % (len(got), got[:3]))

    # (2) OPTIMIZER AUDIT. The shared line is AdamW(model.parameters(), ...),
    # so the encoder IS handed to the optimizer in arm C exactly as in arm A.
    # Decoupled weight decay is what has to be ruled out, so use the real lr/wd.
    opt = torch.optim.AdamW(list(v.parameters()) + list(head.parameters()),
                            lr=2.8953252238193574e-05,
                            weight_decay=0.00822526839606462)
    n_in = sum(len(g["params"]) for g in opt.param_groups)
    _chk("[%s] encoder parameters are handed to the optimizer" % key[0],
         n_in >= len(before), "%d tensors in param_groups" % n_in)
    opt.step()
    moved = [n for n, p in v.named_parameters()
             if not torch.equal(p.detach(), before[n])]
    _chk("[%s] an optimizer step moves no encoder weight" % key[0], not moved,
         "%d tensors vs an independent pre-step clone; decoupled weight decay "
         "does not reach grad-free parameters" % len(before) if not moved
         else "%d moved: %s" % (len(moved), moved[:3]))

    # (4) CONFIGURATION AUDIT -- arm A and arm C side by side
    a = _cells(nb_path("A_pretrained_staged", 42)).splitlines()
    c = _cells(nb_path(key, 42)).splitlines()
    d = [i for i, (x, y) in enumerate(zip(a, c)) if x != y] if len(a) == len(c) else None
    ok = d is not None and all(any(t in a[i] + c[i] for t in
                                   ("ENCODER_SCHEDULE", "RUN_NAME", "OUTPUT_BASE",
                                    "DRY_RUN"))
                               for i in d)
    _chk("[%s] the only intended difference from A is the schedule" % key[0], ok,
         "differing: %s" % ([a[i].strip()[:38] for i in d] if d else "line counts differ"))

    # (4b) TOTAL IDENTITY OUTSIDE THE KNOBS. Stronger than the line diff: with
    # the knob lines removed the two sources must be byte-identical, which is
    # a single statement covering the test split, the preprocessing, the 0.5
    # threshold, the AP/AUC definitions and the pooled-confusion-matrix rule.
    KN = re.compile(r"^(DRY_RUN|ENCODER_INIT|ENCODER_SCHEDULE|RUN_NAME|SEED|OUTPUT_BASE)\b")
    sa = [l for l in a if not KN.match(l)]
    sc = [l for l in c if not KN.match(l)]
    bad = [i for i, (x, y) in enumerate(zip(sa, sc)) if x != y] if len(sa) == len(sc) else None
    _chk("[%s] data, preprocessing, threshold and metric code identical to A" % key[0],
         bad == [],
         "%d non-knob lines, all identical" % len(sa) if bad == []
         else "differs at: %s" % ([sa[i].strip()[:40] for i in (bad or [])][:3]))

    # excluding a line from that diff is only safe if the line is either equal
    # in both arms or purely bookkeeping, so name which knobs actually differ
    ka = {m.group(1): l for l in a for m in [KN.match(l)] if m}
    kc = {m.group(1): l for l in c for m in [KN.match(l)] if m}
    moved_knobs = sorted(k for k in ka if ka[k] != kc.get(k))
    BOOKKEEPING = {"DRY_RUN", "RUN_NAME", "OUTPUT_BASE"}
    sci = [k for k in moved_knobs if k not in BOOKKEEPING]
    _chk("[%s] the only scientific knob that differs from A is the schedule" % key[0],
         sci == ["ENCODER_SCHEDULE"],
         "differing knobs: %s  (scientific: %s; SEED and ENCODER_INIT are equal)"
         % (moved_knobs, sci))

    # SIDE-EFFECT AUDIT. ENCODER_SCHEDULE does not only set requires_grad:
    # the epoch-10 block it gates also adds an optimizer param group and
    # REBUILDS the scheduler. Those are consequences of the comparison, not
    # bugs, and they are reported rather than denied.
    blk = csrc[csrc.index("if epoch == 10 and"):]
    blk = blk[:blk.index("\n        scaler") if "\n        scaler" in blk else 2000]
    touches = {"requires_grad": "requires_grad = True" in blk,
               "optimizer param group": "add_param_group" in blk,
               "scheduler rebuilt": "CosineAnnealingLR" in blk}
    _chk("[%s] what the schedule knob gates is enumerated" % key[0], True,
         "epoch-10 block touches: " + ", ".join(k for k, v in touches.items() if v))
    # what it must NOT change
    def _one(src, pat):
        m = re.search(pat, src)
        return m.group(0) if m else None
    same = all(_one("\n".join(a), q) == _one("\n".join(c), q) for q in (
        r"(?m)^NUM_EPOCHS\s*=\s*\d+",
        r"optimizer = torch\.optim\.AdamW\([^\n]*",
        r"scheduler = build_scheduler\([^\n]*",
        r"for epoch in range\([^\n]*",
        r"(?m)^BATCH_SIZE\s*=\s*\d+", r"(?m)^ACCUM_STEPS\s*=\s*\d+"))
    _chk("[%s] epochs, steps, optimizer and warm-up construction unchanged" % key[0],
         same, "NUM_EPOCHS, BATCH_SIZE, ACCUM_STEPS, AdamW(...) and "
               "build_scheduler(...) identical to A")

    # and nothing executes outside this one cell
    nbj = json.load(io.open(nb_path(key, 42), encoding="utf-8"))
    code = [x for x in nbj["cells"] if x["cell_type"] == "code"]
    loc = re.findall(r"(?m)^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", csrc)
    STD = {"os","sys","re","io","json","csv","math","time","random","logging","glob",
           "warnings","hashlib","datetime","shutil","itertools","collections",
           "numpy","np","torch","torchvision","PIL","rasterio","matplotlib","seaborn","sklearn",
           "transformers","tqdm","pandas","scipy","cv2","typing","pathlib","functools",
           "albumentations","safetensors","huggingface_hub","einops","timm",
           "contextlib","copy","gc","subprocess","traceback","dataclasses","abc"}
    localish = sorted({m.split(".")[0] for m in loc} - STD)
    _chk("[%s] one executable cell, no local helper module imported" % key[0],
         len(code) == 1 and not localish,
         "%d code cell(s); non-stdlib/3rd-party imports: %s" % (len(code), localish or "none"))


def post_run_weight_integrity(run_name="CLCD_C_pretrained_frozen_seed_42"):
    """(3) WEIGHT INTEGRITY -- run AFTER the arm finishes. Every encoder tensor
    in the saved checkpoint must still equal the released DINOv2 weight."""
    import torch
    from transformers import Dinov2Model
    ck = os.path.join(ROOT, "runs_out", run_name, "final", "best_model.pth")
    if not os.path.exists(ck):
        print("no checkpoint yet at %s" % ck)
        return None
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    ref = Dinov2Model.from_pretrained("facebook/dinov2-base").state_dict()
    pre = "encoder.vit."
    enc = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    common = [k for k in enc if k in ref and enc[k].shape == ref[k].shape]
    drift = sorted(((float((enc[k].float() - ref[k].float()).abs().max()), k)
                    for k in common), reverse=True)
    moved = [k for d, k in drift if d > 0]
    print("AFTER training -- encoder vs released DINOv2")
    print("  tensors in checkpoint  : %d" % len(enc))
    print("  tensors compared       : %d" % len(common))
    print("  tensors matched        : %d" % (len(common) - len(moved)))
    print("  tensors changed        : %d" % len(moved))
    print("  max |delta|            : %.6e" % (drift[0][0] if drift else 0.0))
    if moved:
        print("  largest drifts:")
        for d, k in drift[:5]:
            print("    %-58s %.6e" % (k, d))
    print("=> %s" % ("UNCHANGED -- the encoder never trained" if not moved
                     else "CHANGED -- the freeze did NOT hold"))
    return not moved


def prediction_audit():
    """(5) PREDICTION AUDIT -- behaviour, not F1 alone. The predicted-positive
    ratio is recovered from the recorded metrics: specificity s = 2*BA - r,
    prevalence p = (s - acc)/(s - r), predicted-positive fraction = r*p/prec."""
    print("%-34s %6s %6s %6s %6s %6s %7s %7s %6s"
          % ("run", "F1", "prec", "rec", "AP", "AUC", "true+%", "pred+%", "ratio"))
    for r in csv.DictReader(io.open(REGISTRY, encoding="utf-8")):
        # output_base is recorded as the Windows path the run writes to;
        # resolve by run name so this works from either side of the mount
        f = os.path.join(ROOT, "runs_out", r["run_name"], "final", "test_metrics.csv")
        if not os.path.exists(f):
            f = os.path.join(r["output_base"], "final", "test_metrics.csv")
        if not os.path.exists(f):
            continue
        m = {k: float(v) for k, v in
             next(csv.DictReader(io.open(f, encoding="utf-8"))).items()}
        rec, prec, acc, ba = m["recall"], m["precision"], m["accuracy"], m["balanced_accuracy"]
        spec = 2 * ba - rec
        prev = (spec - acc) / (spec - rec) if abs(spec - rec) > 1e-9 else float("nan")
        pp = rec * prev / prec if prec else float("nan")
        print("%-34s %6.2f %6.2f %6.2f %6.2f %6.2f %7.2f %7.2f %6.2fx"
              % (r["run_name"], m["f1"] * 100, prec * 100, rec * 100,
                 m["ap"] * 100, m["auc"] * 100, prev * 100, pp * 100,
                 pp / prev if prev else float("nan")))


# ───────────────────────── report ─────────────────────────
def run_all(build_encoder=None):
    _res.clear(); static_checks()
    if build_encoder is not None:
        for fn in (runtime_checks, schedule_checks, frozen_arm_audits):
            try: fn(build_encoder)
            except Exception as e: _chk("%s completed" % fn.__name__, False, repr(e)[:70])
    else:
        _chk("runtime + schedule checks skipped (no builder)", True,
             "needs torch; run preflight_runtime_clcd.ipynb before training")
    w = max(len(n) for n, _, _ in _res) + 2
    print("=" * (w + 54))
    print(" PREFLIGHT -- encoder initialisation and schedule, CLCD")
    print("=" * (w + 54))
    for n, ok, d in _res:
        print(" %-6s %-*s %s" % ("PASS" if ok else "FAIL", w, n, d[:48]))
    bad = [n for n, ok, _ in _res if not ok]
    print("=" * (w + 54))
    print(" %d checks, %d failed" % (len(_res), len(bad)))
    print(" => %s" % ("ALL PASS -- training may be enabled"
                      if not bad else "BLOCKED -- fix the above before any GPU time"))
    io.open(os.path.join(ROOT, "preflight_report_clcd.txt"), "w", encoding="utf-8").write(
        "\n".join("%-6s %s  %s" % ("PASS" if ok else "FAIL", n, d) for n, ok, d in _res)
        + "\n\ngenerated %s\n" % datetime.datetime.utcnow().isoformat(timespec="seconds"))
    return not bad


# ──────────────── SHARED INITIALISATION OF THE NON-ENCODER PARTS ────────
def _NSM_FOR(key):
    """The namespace the runtime notebook built for an arm (set by build_model)."""
    import builtins
    ns = getattr(builtins, "_PF_NSM", {}).get(key)
    if ns is None:
        raise RuntimeError("build_model(%r) must be called first" % key)
    return ns


def _all_rng_states():
    """Every GLOBAL generator a construction path could draw from."""
    import numpy as np, random as _r, hashlib as _h
    out = {}
    try:
        import torch
        out["torch.cpu"] = _h.sha256(
            torch.random.get_rng_state().numpy().tobytes()).hexdigest()[:16]
        if torch.cuda.is_available():
            out["torch.cuda"] = _h.sha256(b"".join(
                t.cpu().numpy().tobytes() for t in torch.cuda.get_rng_state_all())
            ).hexdigest()[:16]
        else:
            out["torch.cuda"] = "no-cuda"
    except Exception as e:
        out["torch.cpu"] = "err:%s" % type(e).__name__
    st = np.random.get_state()
    out["numpy.global"] = _h.sha256(st[1].tobytes()).hexdigest()[:16] + ":%d" % st[2]
    out["python.random"] = _h.sha256(
        repr(_r.getstate()).encode()).hexdigest()[:16]
    return out


def _print_since(mark, title):
    """Print the checks appended to _res since `mark`."""
    rows = _res[mark:]
    if not rows:
        return
    w = max(len(n) for n, _, _ in rows) + 2
    print("=" * (w + 52)); print(" " + title); print("=" * (w + 52))
    for n, ok, d in rows:
        print(" %-6s %-*s %s" % ("PASS" if ok else "FAIL", w, n, d[:50]))
    bad = [n for n, ok, _ in rows if not ok]
    print("=" * (w + 52))
    print(" %d checks, %d failed" % (len(rows), len(bad)))


def shared_init_audit(build_model):
    """build_model(arm_key) -> the run's full SiameseViT, on CPU, freshly
    constructed after set_seed(SEED).

    set_seed(42) runs once, then the encoder is built. The two arms build it
    by different code paths -- from_pretrained versus Dinov2Model(config) --
    which need not draw the same number of random numbers. Anything
    constructed afterwards (fusion blocks, skip projections, decoder, head)
    therefore need not start from the same weights, even at one seed. If that
    is so, part of any B-vs-C difference is ordinary initialisation variance
    rather than the transferred representation.
    """
    import torch
    _mark = len(_res)

    mb, mc = build_model("B_random_frozen"), build_model("C_pretrained_frozen")
    # ---- the mechanism itself, measured directly ------------------------
    # Seed identically, build ONLY the encoder in each arm, then compare the
    # generator state. If the two construction paths draw the same number of
    # random numbers, the states are equal and nothing built afterwards can
    # diverge for this reason. This tests the mechanism without depending on
    # what the rest of the network happens to do with the draws.
    # Are there LOCAL generators the global probe would miss?
    csrc0 = _cells(nb_path("C_pretrained_frozen", 42))
    locals_found = {k: len(re.findall(v, csrc0)) for k, v in {
        "torch.Generator()": r"torch\.Generator\s*\(",
        "np default_rng(seed)": r"default_rng\s*\([^)]",
        "random.Random()": r"random\.Random\s*\(",
        # torch.manual_seed / torch.cuda.manual_seed_all seed the GLOBAL
        # generators the probe already captures; only a seed call on a local
        # Generator object would escape it.
        "local generator seeded": r"(?<!torch)(?<!torch\.cuda)\.manual_seed\s*\(",
    }.items()}
    locals_found = {k: v for k, v in locals_found.items() if v}
    _chk("[B vs C] no local generator escapes the global probe",
         not locals_found,
         "only the global generators are used" if not locals_found
         else "local generators in source: %s" % locals_found)

    rng_same = None
    try:
        states = {}
        for k2 in ("B_random_frozen", "C_pretrained_frozen"):
            ns = _NSM_FOR(k2)
            ns["set_seed"](ns["SEED"])
            ns["VisionTransformerEncoder"]()
            states[k2] = _all_rng_states()
        keys = sorted(set(states["B_random_frozen"]) | set(states["C_pretrained_frozen"]))
        per = {k2: states["B_random_frozen"].get(k2) == states["C_pretrained_frozen"].get(k2)
               for k2 in keys}
        rng_same = all(per.values())
        # DIAGNOSTIC, not a gate: a differing generator state only matters if
        # it actually changes the weights, which the next check measures.
        _chk("[B vs C] generator states after building the encoder (diagnostic)",
             True,
             " ".join("%s=%s" % (k2, "same" if v else "DIFFERS") for k2, v in per.items()))
    except Exception as e:
        _chk("[B vs C] generator states after building the encoder (diagnostic)",
             True, "probe failed: %r" % (e,))

    pb, pc = dict(mb.named_parameters()), dict(mc.named_parameters())
    is_enc = lambda n: ".vit." in n or n.startswith("encoder.vit")
    nb_ = {n for n in pb if not is_enc(n)}
    nc_ = {n for n in pc if not is_enc(n)}
    _chk("[B vs C] same parameter names outside the encoder", nb_ == nc_,
         "%d vs %d tensors" % (len(nb_), len(nc_)))
    # account for EVERY tensor, so nothing is excluded on a name technicality
    enc_names = {n for n in pb if is_enc(n)}
    _chk("[B vs C] the excluded set is exactly the ViT, nothing else",
         len(enc_names) in (222, 223, 224) and
         all(".vit." in n for n in enc_names),
         "total %d tensors = %d encoder (excluded) + %d compared"
         % (len(pb), len(enc_names), len(nb_)))

    # ---- is that span deterministic at all? -----------------------------
    csrc2 = _cells(nb_path("C_pretrained_frozen", 42))
    body = csrc2[csrc2.index("def main("):]
    span = body[:body.index("model = SiameseViT")] if "model = SiameseViT" in body else body
    SUSPECT = {
        "unseeded default_rng": r"default_rng\(\s*\)",
        "np.random without seed": r"np\.random\.(?!seed|default_rng)\w+\(",
        "time-derived value": r"\b(time\.time|datetime\.now|perf_counter)\b",
        "os.urandom": r"os\.urandom",
        "DataLoader workers": r"num_workers\s*=\s*(?!0)\d+",
    }
    found = {k: len(re.findall(v, span)) for k, v in SUSPECT.items()}
    found = {k: v for k, v in found.items() if v}
    _chk("[B vs C] no non-determinism between the seed and the model build",
         not found,
         "span is %d lines; nothing time-, worker- or unseeded-RNG dependent"
         % len(span.splitlines()) if not found else "found: %s" % found)

    # the premise of the comparison: everything executed before the model is
    # built is byte-identical between the arms, so both reach construction
    # with the same generator state. Re-assert it here rather than rely on it.
    a = _cells(nb_path("B_random_frozen", 42)).splitlines()
    c2 = _cells(nb_path("C_pretrained_frozen", 42)).splitlines()
    KN = re.compile(r"^(DRY_RUN|ENCODER_INIT|ENCODER_SCHEDULE|RUN_NAME|SEED|OUTPUT_BASE)\b")
    pre_a = [l for l in a if not KN.match(l)]
    pre_c = [l for l in c2 if not KN.match(l)]
    _chk("[B vs C] identical code before the model is built",
         pre_a == pre_c,
         "so both arms reach SiameseViT() with the same RNG state")

    # buffers are state too (BatchNorm running stats), so they are compared
    bb = {n: t for n, t in mb.named_buffers() if not is_enc(n)}
    bc = {n: t for n, t in mc.named_buffers() if not is_enc(n)}
    rest = sorted(nb_ & nc_)
    allt = [(n, pb[n], pc[n]) for n in rest] + \
           [(n, bb[n], bc[n]) for n in sorted(set(bb) & set(bc))]
    diff, dmax = [], 0.0
    for n, x, y in allt:
        if x.shape != y.shape:
            diff.append(n); continue
        d = float((x.detach().float() - y.detach().float()).abs().max())
        dmax = max(dmax, d)
        if d > 0:
            diff.append(n)
    nparam = sum(pb[n].numel() for n in rest)

    # show the coverage per module group, so nothing looks quietly excluded
    groups = {}
    for n, _, _ in allt:
        g = n.split(".")[0] if "." in n else n
        groups[g] = groups.get(g, 0) + 1
    _chk("[B vs C] coverage outside the encoder", True,
         "tensors by top-level module: " +
         ", ".join("%s=%d" % kv for kv in sorted(groups.items())))

    _chk("[B vs C] fusion, skips, decoder and head identically initialised",
         not diff,
         "%d tensors (%d params, %.2fM) + buffers compared, %d differ, "
         "max|delta| = %.3e" % (len(allt), len(rest), nparam / 1e6, len(diff), dmax))
    if diff:
        print("    FIRST DIFFERING TENSORS (initialisation, not training):")
        for n in diff[:6]:
            x, y = (pb.get(n), pc.get(n))
            d = float((x.detach().float() - y.detach().float()).abs().max()) \
                if x is not None and y is not None and x.shape == y.shape else float("nan")
            print("      %-56s max|delta| = %.4e" % (n, d))

    _print_since(_mark, "SHARED INITIALISATION OUTSIDE THE ENCODER")

    # ---- the decision, stated against the measurement -------------------
    w_same = not diff
    print()
    print("    " + "-" * 66)
    print("    RNG states after the encoder : %s"
          % ("identical" if rng_same else ("DIFFER" if rng_same is False else "unknown")))
    print("    Non-encoder weights          : %s"
          % ("identical" if w_same else "DIFFER"))
    if rng_same and w_same:
        verdict = "PROCEED -- B vs C is controlled at initialisation; run C"
    elif (rng_same is False) and w_same:
        verdict = ("REVIEW -- the generators diverge but the weights do not; "
                   "do NOT change the protocol automatically")
    elif rng_same and not w_same:
        verdict = ("INVESTIGATE -- same draws yet different weights; the cause "
                   "is elsewhere, find it before running C")
    else:
        verdict = ("HOLD C -- adopt one saved non-encoder initialisation, load "
                   "it in BOTH arms, record its hash in every manifest")
    print("    => %s" % verdict)
    print("    " + "-" * 66)
    print("    The scientific question is the second line, not the first.")
    return w_same


# ───────────────────────── EVIDENCE ARCHIVE ─────────────────────────
def archive_evidence(run_name):
    """Freeze, inside the run's own output folder, everything a reviewer would
    need to re-check it: the as-run notebook source and its hash, the resolved
    knob values, the registry row, the preflight report, and the recorded
    metrics. Copies only -- nothing here is read back by the training path."""
    import hashlib, json, shutil, datetime
    out = os.path.join(ROOT, "runs_out", run_name, "evidence")
    os.makedirs(out, exist_ok=True)
    nb = os.path.join(RUNS_DIR, run_name + ".ipynb")
    src = "".join(json.load(io.open(nb, encoding="utf-8"))["cells"][0]["source"])
    h = hashlib.sha256(src.encode("utf-8")).hexdigest()

    io.open(os.path.join(out, "as_run_source.py"), "w", encoding="utf-8").write(src)
    shutil.copy2(nb, os.path.join(out, "as_run_notebook.ipynb"))
    for f in ("preflight_report_clcd.txt", "preflight_clcd.py", "_gen_clcd.py"):
        q = os.path.join(ROOT, f)
        if os.path.exists(q):
            shutil.copy2(q, os.path.join(out, f))

    knobs = {k: v for k, v in
             (re.match(r"^(\w+)\s*=\s*(.+?)(?:\s+#.*)?$", l).groups()
              for l in src.splitlines()
              if re.match(r"^(DRY_RUN|ENCODER_INIT|ENCODER_SCHEDULE|RUN_NAME|SEED|"
                          r"NUM_EPOCHS|NUM_CHANNELS|IMG_SIZE|PATCH_SIZE|THRESHOLD|"
                          r"BATCH_SIZE|ACCUM_STEPS)\s*=", l))}
    row = next((r for r in csv.DictReader(io.open(REGISTRY, encoding="utf-8"))
                if r["run_name"] == run_name), {})
    met = os.path.join(ROOT, "runs_out", run_name, "final", "test_metrics.csv")
    metrics = next(csv.DictReader(io.open(met, encoding="utf-8"))) if os.path.exists(met) else {}

    def _sha(q):
        return hashlib.sha256(io.open(q, "rb").read()).hexdigest() if os.path.exists(q) else None
    files = {}
    for f in sorted(os.listdir(out)):
        files[f] = _sha(os.path.join(out, f))
    base = os.path.join(os.path.dirname(ROOT), "BCA_CLCD_518",
                        "CLCD_BALANCEDCA_seed_42_518x518", "Untitled301.ipynb")
    files["<base source> Untitled301.ipynb"] = _sha(base)
    try:
        import torch, transformers, numpy, sys as _s
        env = {"python": _s.version.split()[0], "torch": torch.__version__,
               "transformers": transformers.__version__, "numpy": numpy.__version__}
    except Exception as e:
        env = {}
        print("  WARNING: environment not recorded (%s). Re-run archive_evidence "
              "from the training environment; do not guess the values."
              % type(e).__name__)
    else:
        try:
            if torch.cuda.is_available():
                env["cuda"] = torch.version.cuda
                env["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    io.open(os.path.join(out, "manifest.json"), "w", encoding="utf-8").write(
        json.dumps({"run_name": run_name,
                    "file_sha256": files,
                    "environment": env,
                    "archived_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                    "as_run_source_sha256": h,
                    "as_run_source_sha256_16": h[:16],
                    "source_lines": len(src.splitlines()),
                    "resolved_config": knobs,
                    "registry_row": row,
                    "test_metrics": metrics}, indent=1, ensure_ascii=False))
    print("archived -> %s" % out)
    for f in sorted(os.listdir(out)):
        print("   %-28s %9d bytes" % (f, os.path.getsize(os.path.join(out, f))))
    return h


# ───────────────────────── ACCEPTANCE GATE ─────────────────────────
def accept_run(run_name):
    """A run is not accepted because training finished. Every condition below
    must hold before the next arm is started."""
    import json, math
    res = []
    def c(name, ok, detail=""):
        res.append((name, bool(ok), detail))
    base = os.path.join(ROOT, "runs_out", run_name)
    log = os.path.join(base, "training.log")
    met = os.path.join(base, "final", "test_metrics.csv")
    ev = os.path.join(base, "evidence")

    rep = os.path.join(ROOT, "preflight_report_clcd.txt")
    fails = [l for l in io.open(rep, encoding="utf-8").read().splitlines()
             if l.startswith("FAIL")] if os.path.exists(rep) else ["no report"]
    c("preflight (freeze + optimizer + config) all pass", not fails,
      "%d FAIL line(s)" % len(fails) if fails else "0 FAIL lines in the report")

    wi = post_run_weight_integrity(run_name) if "frozen" in run_name else None
    c("encoder weights unchanged after training",
      wi is True if wi is not None else True,
      "not applicable to a trainable-encoder arm" if wi is None else
      ((globals().get("_FREEZE_DETAIL")
        or "byte-identical to the released weights")
       if wi else "weights MOVED"))

    txt = io.open(log, encoding="utf-8", errors="ignore").read() if os.path.exists(log) else ""
    bad = re.findall(r"(?i)\b(nan|inf)\b", txt)
    c("no NaN or Inf in the training log", not bad, "%d occurrence(s)" % len(bad))

    pairs = re.findall(r"Pairs verified: (\d+) samples", txt)
    c("evaluation covered the whole test split",
      pairs[:3] == ["360", "120", "120"],
      "pairs verified (train/val/test): %s" % (pairs[:3] or "not logged"))

    have = {k: os.path.exists(v) for k, v in
            {"checkpoint": os.path.join(base, "final", "best_model.pth"),
             "test_metrics.csv": met, "training.log": log,
             "evidence/": ev}.items()}
    c("checkpoint, metrics, log and evidence all saved", all(have.values()),
      ", ".join("%s%s" % (k, "" if v else " MISSING") for k, v in have.items()))

    ok_rt = False
    if os.path.exists(os.path.join(ev, "manifest.json")) and os.path.exists(met):
        man = json.load(io.open(os.path.join(ev, "manifest.json"), encoding="utf-8"))
        live = next(csv.DictReader(io.open(met, encoding="utf-8")))
        same = man.get("test_metrics", {}) == live
        nb = os.path.join(ev, "as_run_notebook.ipynb")
        src = "".join(json.load(io.open(nb, encoding="utf-8"))["cells"][0]["source"]) \
            if os.path.exists(nb) else ""
        import hashlib
        hok = hashlib.sha256(src.encode()).hexdigest() == man.get("as_run_source_sha256")
        ok_rt = same and hok
        c("results re-readable from the archive", ok_rt,
          "metrics match: %s; archived source hash verifies: %s" % (same, hok))
    else:
        c("results re-readable from the archive", False, "archive incomplete")

    manp = os.path.join(ev, "manifest.json")
    envd = json.load(io.open(manp, encoding="utf-8")).get("environment", {}) \
        if os.path.exists(manp) else {}
    need = {"python", "torch", "transformers", "numpy"}
    c("environment recorded (python, torch, transformers, numpy)",
      need <= set(envd), ", ".join("%s=%s" % (k, envd[k]) for k in sorted(envd))
      or "EMPTY -- re-archive from the training environment")

    w = max(len(n) for n, _, _ in res) + 2
    print("\n" + "=" * (w + 50))
    print(" ACCEPTANCE -- %s" % run_name)
    print("=" * (w + 50))
    for n, ok, d in res:
        print(" %-6s %-*s %s" % ("PASS" if ok else "FAIL", w, n, d[:46]))
    bad_n = [n for n, ok, _ in res if not ok]
    print("=" * (w + 50))
    print(" => %s" % ("TECHNICALLY ACCEPTED -- sound, reproducible, re-readable"
                      if not bad_n else
                      "NOT ACCEPTED -- understand %d failure(s) first" % len(bad_n)))
    print("    This gate judges execution and provenance ONLY. It is not a")
    print("    statement that the result is scientifically sufficient.")
    # state the scientific standing plainly, as information rather than a verdict
    seeds = [r for r in csv.DictReader(io.open(REGISTRY, encoding="utf-8"))
             if r["paper_arm"] == next((x["paper_arm"] for x in
                 csv.DictReader(io.open(REGISTRY, encoding="utf-8"))
                 if x["run_name"] == run_name), "")]
    done = [r for r in seeds if r["status"] == "COMPLETE"]
    txt2 = io.open(log, encoding="utf-8", errors="ignore").read() if os.path.exists(log) else ""
    bests = re.findall(r"New best F1:([0-9.]+)", txt2)
    last_best = txt2.rfind("New best F1")
    tail_frac = (len(txt2) - last_best) / len(txt2) if txt2 and last_best > 0 else 1.0
    print("    arm has %d of %d seeds complete; %d validation improvements, the "
          "last in the final %.0f%% of the log%s"
          % (len(done), len(seeds), len(bests), tail_frac * 100,
             " -- no plateau reached" if tail_frac < 0.15 else ""))
    return not bad_n
