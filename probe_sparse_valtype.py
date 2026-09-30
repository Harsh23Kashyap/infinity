import sys, time, traceback
sys.path.insert(0, "python/infinity_sdk")
try:
    import infinity
    from infinity.common import ConflictType, NetworkAddress
except Exception:
    for line in traceback.format_exc().splitlines()[-6:]:
        print(f"::error title=IMPORT::{line[:280]}", flush=True)
    sys.exit(1)

def note(m): print(f"::notice title=PROBE::{m}", flush=True)
def err(m): print(f"::error title=PROBE::{m}", flush=True)
def short(s, n=350): return " ".join(str(s).split())[:n]

SPECS = {
    "t_f32": "sparse,30000,float,int32",
    "t_f16": "sparse,30000,float16,int32",
    "t_bf16": "sparse,30000,bfloat16,int32",
}

def connect_with_retry(tries=30):
    for _ in range(tries):
        try:
            c = infinity.connect(NetworkAddress("127.0.0.1", 23817))
            c.show_databases()
            return c
        except Exception:
            time.sleep(2)
    return None

phase = sys.argv[1] if len(sys.argv) > 1 else "define"

conn = connect_with_retry(15 if phase == "define" else 3)
if conn is None:
    err(f"phase={phase}: cannot connect to server")
    sys.exit(1)

db = conn.get_database("probe_db") if phase != "define" else None

if phase == "define":
    db = conn.create_database("probe_db", ConflictType.Ignore)
    db = conn.get_database("probe_db")
    for name, ctype in SPECS.items():
        try:
            db.drop_table(name, ConflictType.Ignore)
            t = db.create_table(name, {"id": {"type": "int"}, "s": {"type": ctype}})
            note(f"create_table {name} [{ctype}]: OK")
        except Exception as e:
            err(f"create_table {name} [{ctype}]: {type(e).__name__}: {short(e)}")
    for name in SPECS:
        try:
            t = db.get_table(name)
            t.insert([{"id": 1, "s": {0: 1.5, 7: 2.5}}, {"id": 2, "s": {3: 0.5}}])
            note(f"insert {name}: OK")
        except Exception as e:
            err(f"insert {name}: {type(e).__name__}: {short(e)}")
elif phase == "match":
    name = sys.argv[2]
    t = db.get_table(name)
    try:
        res, extra, _ = t.output(["id", "s"]).to_result()
        note(f"plain select {name}: OK rows={len(res.get('id', []))}")
    except Exception as e:
        err(f"plain select {name}: {type(e).__name__}: {short(e)}")
    try:
        res, extra, _ = t.output(["id"]).match_sparse("s", {0: 1.0, 3: 0.5}, "ip", 3).to_result()
        note(f"match_sparse {name}: OK result={short(res)}")
    except Exception as e:
        err(f"match_sparse {name}: {type(e).__name__}: {short(e)}")
    time.sleep(3)
    try:
        conn.show_tables()
        note(f"server alive after match on {name}: YES")
    except Exception as e:
        err(f"server alive after match on {name}: NO ({type(e).__name__}: {short(e, 150)})")
