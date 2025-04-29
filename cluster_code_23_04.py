import os, json, time, warnings
from collections import defaultdict
from typing import List, Dict

import numpy as np, pandas as pd, psycopg2
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import __version__ as skl_ver
from tqdm import tqdm

TEST_TABLE    = "test_table"
WIPE_TASKS    = True             
DOC2VEC_DIM   = 50              
DOC2VEC_EPOCH = 20

load_dotenv()
UC  = dict(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
           dbname=os.getenv("TM_DB_NAME"), user=os.getenv("TM_DB_USER"),
           password=os.getenv("TM_DB_PASSWORD"))
OUT = dict(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
           dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
           password=os.getenv("DB_PASSWORD"))

warnings.filterwarnings("ignore", category=UserWarning, module="gensim")

REQ_NUM = ["cpu_cores_count", "gpu_count", "nodes_count", "time_limit"]
REQ_CAT = ["partition", "constraints"]

RUN_NUM = ["cpu_utilization_avg", "gpu_utilization_avg", "memory_usage_avg",
           "filesystem_read_sum", "filesystem_write_sum",
           "fs_disk_read_sum", "fs_disk_write_sum",
           "fs_lustre_read_sum", "fs_lustre_write_sum",
           "ib_send_sum", "ib_receive_sum"]

strip_comments = lambda s: "\n".join(
    l for l in s.splitlines() if not l.lstrip().startswith("#")
)

def fetch_jobs() -> pd.DataFrame:
    sql = f"""
    SELECT job_id,
           {', '.join(REQ_NUM + REQ_CAT + RUN_NUM)},
           batch_script, time_start, time_end
    FROM cabinet_job
    WHERE batch_script IS NOT NULL;
    """
    with psycopg2.connect(**UC) as conn:
        print("→ Загружаем задачи (без LIMIT) …")
        return pd.read_sql_query(sql, conn)


def build_req(df: pd.DataFrame) -> np.ndarray:
    print("→ Embedding A (requested resources) …")
    X_num = StandardScaler().fit_transform(df[REQ_NUM].fillna(0).values)
    enc   = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = enc.fit_transform(df[REQ_CAT].fillna("nan")).astype(float)
    X     = np.hstack([X_num, X_cat])
    return PCA(n_components=1, random_state=42).fit_transform(X).ravel()


def build_run(df: pd.DataFrame) -> np.ndarray:
    print("→ Embedding B (runtime usage) …")
    dfc = df.copy()
    dfc["runtime_sec"] = (
        dfc["time_end"] - dfc["time_start"]
    ).dt.total_seconds().fillna(0)
    X = StandardScaler().fit_transform(dfc[RUN_NUM + ["runtime_sec"]].fillna(0).values)
    return PCA(n_components=1, random_state=42).fit_transform(X).ravel()


def build_sw(df: pd.DataFrame) -> np.ndarray:
    print("→ Embedding C (software signature) …")
    docs: List[TaggedDocument] = []
    for i, txt in enumerate(df["batch_script"].astype(str)):
        docs.append(TaggedDocument(strip_comments(txt).split(), [i]))

    model = Doc2Vec(
        vector_size=DOC2VEC_DIM, min_count=2, workers=4, epochs=DOC2VEC_EPOCH
    )
    model.build_vocab(docs, update=False)
    model.train(docs, total_examples=len(docs), epochs=model.epochs)
    vecs = np.vstack([model.dv[i] for i in range(len(docs))])
    return PCA(n_components=1, random_state=42).fit_transform(vecs).ravel()


def wipe_old_tasks(cur):
    if WIPE_TASKS:
        cur.execute(
            f"DELETE FROM {TEST_TABLE} WHERE (name::json)->>'type' = 'task';"
        )


def main():
    start = time.time()
    df = fetch_jobs()
    total_tasks = len(df)
    if total_tasks == 0:
        print("Нет задач — выход."); return
    print(f"→ Всего задач: {total_tasks:,}")

    emb_req = build_req(df)
    emb_run = build_run(df)
    emb_sw  = build_sw(df)

    rows = [
        (
            json.dumps(
                {
                    "type": "task",
                    "job_id": int(jid),
                    "req": float(e1),
                    "run": float(e2),
                    "sw":  float(e3),
                }
            ),
        )
        for jid, e1, e2, e3 in zip(df["job_id"], emb_req, emb_run, emb_sw)
    ]
    with psycopg2.connect(**OUT) as conn, conn.cursor() as cur:
        wipe_old_tasks(cur)
        cur.executemany(
            f"INSERT INTO {TEST_TABLE} (name) VALUES (%s)",
            rows,
        )
        conn.commit()

        cur.execute(
            f"SELECT COUNT(*) FROM {TEST_TABLE} "
            f"WHERE (name::json)->>'type' = 'task';"
        )
        inserted = cur.fetchone()[0]

    assert (
        inserted == total_tasks
    ), f"Assert failed: задач {total_tasks}, а в test_table {inserted}"

    print(
        f"{inserted} задач записано в {TEST_TABLE} "
        f"за {time.time()-start:0.1f} с  (обработаны все задачи)"
    )


if __name__ == "__main__":
    main()