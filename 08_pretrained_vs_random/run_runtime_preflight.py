# -*- coding: utf-8 -*-
"""Runtime preflight, as a single script. Same cells as
preflight_runtime_clcd.ipynb, with every line also written to
runtime_preflight_output.txt so the result can be read back from the file
instead of copied by hand.

NOTHING TRAINS HERE. Only the prefix of each generated run is executed, and
the cut is asserted to exclude main() and the entry point.

Run from an Anaconda Prompt:
    python "D:\\Master Thesis Codes\\attention\\PRETRAIN_VS_RANDOM_CLCD\\run_runtime_preflight.py"
or from a Jupyter cell:
    %run "D:\\Master Thesis Codes\\attention\\PRETRAIN_VS_RANDOM_CLCD\\run_runtime_preflight.py"
"""
import sys, os, io as _io, datetime

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "runtime_preflight_output.txt")

class _Tee(object):
    def __init__(self, path):
        self.f = _io.open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.stdout.flush(); self.f.flush()

_tee = _Tee(_OUT)
sys.stdout = _tee
sys.stderr = _tee
print("runtime preflight started %s UTC" % datetime.datetime.utcnow().isoformat(timespec="seconds"))
print("output file: %s" % _OUT)
print("=" * 78)

try:
    # ===== cell 1 of 9 : write everything this notebook prints to a file =====
    # so the result can be read back from disk instead of copied by hand.
    import sys, os, io as _io, datetime, builtins

    _OUT = r"D:\Master Thesis Codes\attention\PRETRAIN_VS_RANDOM_CLCD\runtime_preflight_output.txt"

    class _PF_TEE(object):
        def __init__(self, path, stream):
            self.f = _io.open(path, "w", encoding="utf-8")
            self.stream = stream
        def write(self, s):
            self.stream.write(s)
            try:
                self.f.write(s); self.f.flush()
            except Exception:
                pass
        def flush(self):
            self.stream.flush()
            try: self.f.flush()
            except Exception: pass

    if not isinstance(sys.stdout, _PF_TEE):
        builtins._PF_REAL_STDOUT = sys.stdout
        sys.stdout = _PF_TEE(_OUT, sys.stdout)
        print("logging to %s" % _OUT)
    print("runtime preflight  %s UTC" % datetime.datetime.utcnow().isoformat(timespec="seconds"))
    print("=" * 74)

    # ---- refuse to run in a kernel that already tried and failed -----------
    # A half-imported torch cannot be repaired in-process: its C++ operators are
    # registered once per process, so a second import raises
    # "Only a single TORCH_LIBRARY ... prims". Likewise a stub left in
    # sys.modules by an earlier attempt will poison inspect.getmodule(). Say so
    # in one line instead of producing a sixty-line cascade.
    _poison = []
    if "torch" in sys.modules and not hasattr(sys.modules["torch"], "nn"):
        _poison.append("torch is half-imported from an earlier failed attempt")
    _alb0 = sys.modules.get("albumentations")
    if _alb0 is not None and type(_alb0).__name__ in ("_Forbidden", "_StubModule"):
        _poison.append("a stub from an earlier run is still installed")
    if _poison:
        raise SystemExit(
            "\n\n*** RESTART THE KERNEL BEFORE RUNNING THIS ***\n    "
            + "\n    ".join(_poison)
            + "\n\n    Kernel -> Restart Kernel, then Run All.\n"
              "    Nothing here can fix it from inside the same process.\n")
    print("kernel is clean (torch not yet imported, no stale stub)")


    # ======================================================================
    #  PREFLIGHT, RUNTIME TIER -- encoder initialisation and schedule, CLCD
    #
    #  Nothing trains here. It executes only the PREFIX of each generated run,
    #  from the top of the file through the end of VisionTransformerEncoder.
    #  main() lives below that line and is never reached, so no dataset
    #  directory is opened and no checkpoint is read. CPU only, a few minutes.
    # ======================================================================
    import io, json, os, re, sys, gc, tempfile, importlib
    ROOT = r"D:\Master Thesis Codes\attention\PRETRAIN_VS_RANDOM_CLCD"
    RUNS = os.path.join(ROOT, "runs")
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    import preflight_clcd as pf
    importlib.reload(pf)
    print("loaded", pf.__file__)


    # ---- heavy imports FIRST, before anything touches sys.modules ----------
    # torch's own import calls inspect.getmodule(), which does
    # hasattr(m, "__file__") over every entry in sys.modules. A module-like
    # object that objects to that breaks torch's import and IPython's traceback
    # printer with it. So: import torch before installing anything, and make the
    # fallback below behave like a real module for dunder lookups.
    import torch, numpy as np
    print("torch %s | cuda %s" % (torch.__version__, torch.cuda.is_available()))

    # ---- albumentations: try it for real, stub only if it will not load ----
    # The run imports it at the top, but every use sits inside main(), below the
    # cut this preflight executes. A previous attempt died importing it
    # (albumentations -> huggingface_hub.hf_api -> httpcore -> h2 -> hpack,
    # MemoryError). Prefer the real module; fall back loudly.
    import types as _t
    class _StubModule(_t.ModuleType):
        """Behaves like an empty module for dunders; objects only to real use."""
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            raise RuntimeError(
                "the preflight used albumentations.%s -- it was stubbed because "
                "nothing above main() should need it. Investigate rather than "
                "widen the stub." % name)

    try:
        import albumentations as _alb
        print("albumentations %s imported normally (not stubbed)"
              % getattr(_alb, "__version__", "?"))
    except BaseException as _e:
        sys.modules["albumentations"] = _StubModule("albumentations")
        print("NOTE: albumentations failed to import (%s: %s)"
              % (type(_e).__name__, str(_e)[:70]))
        print("      stubbed for this preflight; it is only used inside main(),")
        print("      which is below the cut and never executed here.")


    def _is_top_def(l):
        return bool(l) and not l[0].isspace() and (l.startswith("class ") or l.startswith("def "))

    def _prefix(key, seed=42, upto="class VisionTransformerEncoder"):
        p  = os.path.join(RUNS, "CLCD_%s_seed_%d.ipynb" % (key, seed))
        nb = json.load(io.open(p, encoding="utf-8"))
        L  = "".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code").split("\n")
        if upto == "def main(":
            # everything through the full model definition; main() and the entry
            # point stay out, so no data directory is opened and nothing trains
            j = next(k for k, l in enumerate(L) if l.startswith("def main("))
        else:
            i = next(k for k, l in enumerate(L) if l.startswith(upto))
            j = next((k for k in range(i + 1, len(L)) if _is_top_def(L[k])), len(L))
        head = "\n".join(L[:j])
        # the cut is asserted, not assumed: this is what makes execution safe
        assert "def main(" not in head, "the cut let main() through"
        assert "if __name__" not in head, "the cut let the entry point through"
        assert "Dinov2Config" in head and "class VisionTransformerEncoder" in head
        tmp = tempfile.mkdtemp(prefix="pf_clcd_")
        head, n = re.subn(r'(?m)^OUTPUT_BASE = .*$',
                          'OUTPUT_BASE = r"%s"' % tmp.replace("\\", "\\\\"), head, count=1)
        assert n == 1
        return head, p

    _NS = {}
    def build_encoder(key, rng_seed=None):
        """key is an arm key such as 'B_random_alltrain'. rng_seed, when given,
        reseeds immediately before the build: the run seeds itself, so two builds
        at the same seed SHOULD match, and a different seed is how we tell a
        sampled initialisation from weights read off disk."""
        if key not in _NS:
            head, p = _prefix(key)
            ns = {"__name__": "pf_%s" % key}
            exec(compile(head, p, "exec"), ns)
            # DRY_RUN is NOT what makes this safe, and it is legitimately False
            # in two notebooks: the completed arm D (kept as its as-run artefact)
            # and whichever arm is currently armed. Safety comes from the cut:
            # _prefix() asserts that neither main() nor the entry point is in the
            # text being executed, so no training path exists here to gate.
            assert ns["DRY_RUN"] in (True, False), "DRY_RUN missing from the source"
            assert ns["NUM_CHANNELS"] == 3 and ns["IMG_SIZE"] == 518
            _NS[key] = ns
            print("  prefix ok  %-24s init=%-11s sched=%-14s DRY_RUN=%s"
                  % (key, ns["ENCODER_INIT"], ns["ENCODER_SCHEDULE"], ns["DRY_RUN"]))
        ns = _NS[key]
        if rng_seed is not None:
            import torch, numpy as np, random
            torch.manual_seed(rng_seed); np.random.seed(rng_seed % (2**32)); random.seed(rng_seed)
        return ns["VisionTransformerEncoder"]()


    _NSM = {}
    def build_model(key, seed=42):
        """The run's full SiameseViT, built exactly as main() would build it:
        set_seed(SEED) runs inside the prefix, then the model is constructed."""
        if key not in _NSM:
            head, p = _prefix(key, seed, upto="def main(")
            ns = {"__name__": "pfm_%s" % key}
            exec(compile(head, p, "exec"), ns)
            assert ns["DRY_RUN"] in (True, False)
            assert "SiameseViT" in ns, "SiameseViT not defined in the prefix"
            _NSM[key] = (ns, head)
            # publish for preflight_clcd's RNG probe, which needs to construct
            # the encoder alone in each arm
            import builtins
            if not hasattr(builtins, "_PF_NSM"):
                builtins._PF_NSM = {}
            builtins._PF_NSM[key] = ns
            print("  model prefix ok  %-22s init=%-11s sched=%s"
                  % (key, ns["ENCODER_INIT"], ns["ENCODER_SCHEDULE"]))
        ns, _ = _NSM[key]
        # re-seed exactly as the run does before building the model, so the
        # comparison reproduces the run's own initialisation order
        ns["set_seed"](ns["SEED"])
        return ns["SiameseViT"]()


    # smoke build first, so a failure is legible rather than buried in the report
    import torch
    for k, _, _ in pf.ARMS:
        m = build_encoder(k)
        v = getattr(m, "vit", m)
        tot = sum(p.numel() for p in v.parameters())
        tr  = sum(p.numel() for p in v.parameters() if p.requires_grad)
        pe  = v.embeddings.patch_embeddings.projection.weight
        print("%-24s encoder %.2fM total | %.2fM trainable | patch-embed %s"
              % (k, tot/1e6, tr/1e6, tuple(pe.shape)))
        del m
    gc.collect()

    ok = pf.run_all(build_encoder=build_encoder)
    print()
    print("RUNTIME TIER:", "ALL PASS" if ok else "BLOCKED")
    print("report:", os.path.join(ROOT, "preflight_report_clcd.txt"))

    # ---- does the encoder build leave the REST of the network identical? ----
    # set_seed(42) runs once; the two arms then build the encoder by different
    # code paths, which need not consume the same number of random draws.
    ok_init = pf.shared_init_audit(build_model)
    print()
    print("shared non-encoder initialisation:", "IDENTICAL" if ok_init else "DIFFERENT")


    # Training stays off. DRY_RUN is cleared by hand, in ONE notebook at a time,
    # and only after the line above reads ALL PASS.
    print("DRY_RUN is still True in all nine notebooks. Nothing has trained.")

    # ===== last cell : close the log =====
    import sys, builtins
    if isinstance(sys.stdout, _PF_TEE):
        print("=" * 74)
        print("end of runtime preflight")
        sys.stdout.f.close()
        sys.stdout = builtins._PF_REAL_STDOUT
        print("written to %s" % _OUT)
    else:
        print("log was not open")

except SystemExit as _e:
    print(_e)
except Exception:
    import traceback
    print()
    print("!" * 78)
    print("RUNTIME PREFLIGHT RAISED -- nothing was trained, nothing was changed")
    traceback.print_exc()
    print("!" * 78)
finally:
    print()
    print("=" * 78)
    print("finished %s UTC" % datetime.datetime.utcnow().isoformat(timespec="seconds"))
    try:
        sys.stdout = _tee.stdout; sys.stderr = sys.__stderr__; _tee.f.close()
    except Exception:
        pass
    print("written to %s" % _OUT)
