# ============================================================
# 🤖 MACHINE LEARNING  - Streamlit App (Layout Dashboard)
# Regresi, Klasifikasi, Clustering + MongoDB (Retrain & Komentar)
# ============================================================

import re
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

# Model regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# Model klasifikasi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Model clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

# Metrik evaluasi
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)

# MongoDB
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId

# ------------------------------------------------------------
# ⚙️ Konfigurasi halaman
# ------------------------------------------------------------
st.set_page_config(
    page_title="Machine Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# 🎨 CSS: layout dashboard (sidebar menu, kartu, panel skor)
# Catatan: container ber-key "card_*" / "panel_*" butuh Streamlit >= 1.39
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Latar & kanvas utama ---------- */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #EBD3E0 0%, #CFA9C3 100%);
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1280px;
        background: #FFF5F9;
        border-radius: 28px;
        padding: 2rem 2.2rem 2.5rem 2.2rem;
        margin: 1.2rem auto;
        box-shadow: 0 10px 40px rgba(120, 50, 90, 0.18);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { background: #F7DCEA; }
    section[data-testid="stSidebar"] .block-container {
        background: transparent; box-shadow: none; padding: 1.2rem 1rem; margin: 0;
    }
    .brand { font-size: 1.6rem; font-weight: 800; color: #7A2A57; letter-spacing: -0.5px; }
    .brand span { color: #EC4899; }
    .profile {
        display: flex; align-items: center; gap: 12px;
        margin: 14px 0 18px 0; padding: 4px 2px;
    }
    .avatar {
        width: 52px; height: 52px; border-radius: 50%;
        background: linear-gradient(135deg, #F9A8D4, #BE185D);
        display: flex; align-items: center; justify-content: center; font-size: 1.6rem;
        box-shadow: 0 2px 8px rgba(190,24,93,.3);
    }
    .profile-nama { font-weight: 700; color: #3B0A24; line-height: 1.2; }
    .profile-role { font-size: .8rem; color: #9D5C7C; }

    /* Menu navigasi (st.radio) tampil seperti daftar menu */
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 10px 14px; border-radius: 14px; width: 100%; cursor: pointer;
        transition: background .15s;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.55); }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #FFF5F9; font-weight: 700;
        box-shadow: 0 2px 8px rgba(190,24,93,.12);
    }

    /* ---------- Header ---------- */
    .hello { font-size: 1.9rem; font-weight: 800; color: #5B1F42; letter-spacing: .5px; margin: 0; }
    .hello-sub { color: #9D5C7C; margin: 2px 0 0 0; font-size: .98rem; }
    .chip {
        display: inline-block; background: #F7DCEA; color: #7A2A57; font-weight: 600;
        border-radius: 999px; padding: 6px 14px; margin: 0 6px 6px 0; font-size: .85rem;
        border: 1px solid #F3C4DC;
    }

    /* ---------- Kartu ---------- */
    [class*="st-key-card"] {
        background: #FCE7F3; border: 1px solid #F5CDE2; border-radius: 22px;
        padding: 1.1rem 1.3rem; box-shadow: 0 2px 10px rgba(190,24,93,.08);
    }
    .card-title { font-weight: 700; color: #5B1F42; font-size: 1.02rem; margin-bottom: .4rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #FCE7F3 100%);
        border: 1px solid #FBCFE8; border-radius: 16px;
        padding: 14px 18px; box-shadow: 0 2px 8px rgba(190,24,93,0.08);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div { color: #3B0A24 !important; }

    /* ---------- Panel skor (kanan) ---------- */
    .st-key-panel_kanan {
        border: 1px solid #D9A9C4; border-radius: 26px; padding: 1rem;
        background: #FBE3EF;
    }
    .gauge-card {
        background: linear-gradient(160deg, #A8688B 0%, #8E4F73 100%);
        border-radius: 22px; padding: 16px 12px 14px 12px; margin-bottom: 14px;
        text-align: center; color: #fff;
        box-shadow: 0 3px 10px rgba(110, 40, 80, .25);
    }
    .gauge-title { font-weight: 600; font-size: .95rem; opacity: .95; }
    .ring {
        width: 112px; height: 112px; border-radius: 50%; margin: 10px auto 4px auto;
        display: flex; align-items: center; justify-content: center; position: relative;
        background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.22) 0);
    }
    .ring::before {
        content: ""; position: absolute; inset: 11px; border-radius: 50%; background: #9A5A7D;
    }
    .ring span { position: relative; font-weight: 800; font-size: 1.2rem; color: #fff; }
    .stat-num { font-size: 2rem; font-weight: 800; margin: 18px 0 14px 0; }
    .panel-link { text-align: center; color: #7A2A57; font-weight: 600; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 📦 Konstanta dataset, model, dan MongoDB
# ------------------------------------------------------------
DATASET_REGRESI = {
    "🏠 Diabetes": datasets.load_diabetes,
    "🌸 Linnerud (target 1)": datasets.load_linnerud,
}
DATASET_KLASIFIKASI = {
    "🌺 Iris": datasets.load_iris,
    "🍷 Wine": datasets.load_wine,
    "🎗️ Breast Cancer": datasets.load_breast_cancer,
    "🔢 Digits": datasets.load_digits,
}
DATASET_CLUSTERING = {
    "🌺 Iris": datasets.load_iris,
    "🍷 Wine": datasets.load_wine,
    "🎗️ Breast Cancer": datasets.load_breast_cancer,
}

MODEL_REGRESI = ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest"]
MODEL_KLASIFIKASI = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]
MODEL_CLUSTERING = ["KMeans", "Agglomerative", "DBSCAN", "Gaussian Mixture"]

NAMA_DB = "test_db"
NAMA_COLLECTION = "iris_db"
NAMA_COLLECTION_KOMENTAR = "komentar"
FITUR_IRIS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES_MAP = {"setosa": 0, "versicolor": 1, "virginica": 2}  # string -> integer
SPECIES_LIST = list(SPECIES_MAP.keys())
TEMA_GRAFIK = {"figure.facecolor": "#FFF5F9", "axes.facecolor": "#FFFFFF", "grid.color": "#FBCFE8"}
WIB = timedelta(hours=7)  # selisih UTC -> WIB untuk tampilan komentar

MENU = ["🏠 Dashboard", "📈 Visualisasi", "📋 Evaluasi", "🗄️ Database & Retrain", "💬 Komentar"]


def buat_model(tugas: str, nama: str, random_state: int):
    """Membuat model sesuai pilihan; model yang sensitif skala dibungkus scaler."""
    if tugas == "Regresi":
        peta = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=random_state),
            "Lasso": Lasso(alpha=0.1, random_state=random_state),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        }
    else:
        peta = {
            "Logistic Regression": make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, random_state=random_state)
            ),
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "SVM": make_pipeline(StandardScaler(), SVC(random_state=random_state)),
        }
    return peta[nama]


def buat_model_cluster(nama: str, k: int, eps: float, min_samples: int, random_state: int):
    """Membuat model clustering sesuai pilihan dan parameternya."""
    if nama == "KMeans":
        return KMeans(n_clusters=k, n_init=10, random_state=random_state)
    if nama == "Agglomerative":
        return AgglomerativeClustering(n_clusters=k)
    if nama == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples)
    return GaussianMixture(n_components=k, random_state=random_state)


# ------------------------------------------------------------
# 💾 Cache pemuatan dataset
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def muat_dataset(tugas: str, nama: str):
    """Memuat dataset toy dan mengembalikan X (DataFrame), y (Series), nama kelas."""
    if tugas == "Regresi":
        loader = DATASET_REGRESI[nama]
    elif tugas == "Klasifikasi":
        loader = DATASET_KLASIFIKASI[nama]
    else:
        loader = DATASET_CLUSTERING[nama]
    data = loader()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    if nama.startswith("🌸 Linnerud"):
        # Linnerud punya 3 target; ambil target pertama (Weight) saja
        y = pd.Series(data.target[:, 0], name=data.target_names[0])
    else:
        y = pd.Series(data.target, name="target")

    punya_kelas = tugas in ("Klasifikasi", "Clustering") and hasattr(data, "target_names")
    kelas = list(data.target_names) if punya_kelas else None
    return X, y, kelas


# ------------------------------------------------------------
# 🍃 Fungsi MongoDB
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_mongo_client():
    """Membuat koneksi MongoDB satu kali saja (di-cache sebagai resource)."""
    uri = st.secrets["MONGODB_URI"]
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # uji koneksi; error jika gagal
    return client


def cek_koneksi() -> bool:
    """Memeriksa koneksi MongoDB; menampilkan pesan error yang jelas bila gagal."""
    try:
        get_mongo_client()
        return True
    except KeyError:
        st.error(
            "🔑 `MONGODB_URI` tidak ditemukan di `st.secrets`. "
            "Buat file `.streamlit/secrets.toml` (lokal) atau isi menu Secrets (Streamlit Cloud):\n\n"
            '```toml\nMONGODB_URI = "mongodb+srv://user:password@cluster.mongodb.net/"\n```'
        )
    except PyMongoError as e:
        st.error(
            "🔌 Gagal terhubung ke MongoDB. Periksa URI, username/password, "
            "dan whitelist IP di MongoDB Atlas."
        )
        st.caption(f"Detail error: {e}")
        get_mongo_client.clear()  # hapus cache koneksi gagal agar bisa dicoba lagi
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan tak terduga saat koneksi: {e}")
    return False


def get_collection():
    """Mengambil collection iris_db dari database test_db."""
    return get_mongo_client()[NAMA_DB][NAMA_COLLECTION]


def get_collection_komentar():
    """Mengambil collection komentar dari database test_db."""
    return get_mongo_client()[NAMA_DB][NAMA_COLLECTION_KOMENTAR]


@st.cache_data(ttl=5, show_spinner=False)
def ambil_data_mongo() -> pd.DataFrame:
    """Mengambil seluruh dokumen Iris dari MongoDB (cache 5 detik)."""
    dokumen = list(get_collection().find().sort("timestamp", -1))
    if not dokumen:
        return pd.DataFrame(columns=["_id"] + FITUR_IRIS + ["species", "timestamp"])
    df = pd.DataFrame(dokumen)
    df["_id"] = df["_id"].astype(str)  # ObjectId -> string agar bisa ditampilkan
    return df


@st.cache_data(ttl=5, show_spinner=False)
def ambil_komentar(batas: int = 100) -> list:
    """Mengambil komentar terbaru dari MongoDB (cache 5 detik)."""
    dokumen = get_collection_komentar().find().sort("timestamp", -1).limit(batas)
    return [
        {"nama": d.get("nama", "Anonim"), "komentar": d.get("komentar", ""), "timestamp": d.get("timestamp")}
        for d in dokumen
    ]


def bersihkan_cache_data():
    """Hapus cache query supaya data terbaru langsung tampil setelah insert/delete."""
    ambil_data_mongo.clear()
    ambil_komentar.clear()


def escape_md(teks: str) -> str:
    """Escape karakter markdown pada teks dari publik supaya tidak dirender sebagai markdown/link."""
    teks = re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$])", r"\\\1", teks)
    return teks.replace("\n", "  \n")


# ------------------------------------------------------------
# 🧱 Helper tampilan (kartu, header, gauge)
# ------------------------------------------------------------
def kartu(nama: str):
    """Container bergaya kartu (di-style lewat CSS dengan key 'card_*')."""
    return st.container(key=f"card_{nama}")


def judul_kartu(teks: str):
    st.markdown(f'<div class="card-title">{teks}</div>', unsafe_allow_html=True)


def header_halaman(judul: str, sub: str, chips: list | None = None):
    st.markdown(f'<p class="hello">{judul}</p><p class="hello-sub">{sub}</p>', unsafe_allow_html=True)
    if chips:
        st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)


def gauge_html(judul: str, pct, teks: str, warna: str) -> str:
    """Kartu skor: cincin persentase bila pct (0-1) tersedia, selain itu angka besar."""
    if pct is None:
        isi = f'<div class="stat-num">{teks}</div>'
    else:
        p = max(0.0, min(1.0, float(pct))) * 100
        isi = f'<div class="ring" style="--p:{p:.0f};--c:{warna}"><span>{teks}</span></div>'
    return f'<div class="gauge-card"><div class="gauge-title">{judul}</div>{isi}</div>'


def skor_ringkas(hasil):
    """Mengembalikan 3 skor utama (judul, pct 0-1 / None, teks) untuk panel kanan."""
    if hasil is None:
        return [("Skor 1", 0, "–"), ("Skor 2", 0, "–"), ("Skor 3", 0, "–")]
    if hasil["tugas"] == "Clustering":
        sil, dbi, ari = hasil["silhouette"], hasil["davies_bouldin"], hasil["ari"]
        return [
            ("Silhouette", sil if sil is not None else 0, f"{sil:.2f}" if sil is not None else "–"),
            ("Adjusted Rand Index", max(ari, 0), f"{ari:.2f}"),
            ("Davies-Bouldin", None, f"{dbi:.2f}" if dbi is not None else "–"),
        ]
    y_test, y_pred = hasil["y_test"], hasil["y_pred"]
    if hasil["tugas"] == "Regresi":
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return [("R²", max(r2, 0), f"{r2:.2f}"), ("MAE", None, f"{mae:.2f}"), ("RMSE", None, f"{rmse:.2f}")]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    return [
        ("Accuracy", acc, f"{acc:.0%}"),
        ("F1-Score", f1, f"{f1:.0%}"),
        ("Recall", rec, f"{rec:.0%}"),
    ]


# ------------------------------------------------------------
# 🧠 Session state: simpan hasil agar tidak hilang saat rerun
# ------------------------------------------------------------
if "hasil" not in st.session_state:
    st.session_state.hasil = None
if "hasil_retrain" not in st.session_state:
    st.session_state.hasil_retrain = None

# ------------------------------------------------------------
# 🎛️ Sidebar: brand, profil, menu, pengaturan, tombol train
# ------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">ML<span>·</span>Playground</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="profile"><div class="avatar">🤖</div>'
        '<div><div class="profile-nama">Machine Learning</div>'
        '<div class="profile-role">562024006</div></div></div>',
        unsafe_allow_html=True,
    )

    halaman = st.radio("Menu", MENU, label_visibility="collapsed", key="halaman")

    st.divider()
    with st.expander("⚙️ Pengaturan Model", expanded=True):
        tugas = st.radio("🎯 Jenis Tugas", ["Regresi", "Klasifikasi", "Clustering"])

        if tugas == "Regresi":
            daftar_data, daftar_model = DATASET_REGRESI, MODEL_REGRESI
        elif tugas == "Klasifikasi":
            daftar_data, daftar_model = DATASET_KLASIFIKASI, MODEL_KLASIFIKASI
        else:
            daftar_data, daftar_model = DATASET_CLUSTERING, MODEL_CLUSTERING

        nama_dataset = st.selectbox("📚 Dataset", list(daftar_data.keys()))
        nama_model = st.selectbox("🧩 Model", daftar_model)

        # Parameter khusus clustering (nilai default agar variabel selalu terdefinisi)
        k_klaster, eps, min_samples = 3, 1.5, 5
        if tugas == "Clustering":
            if nama_model == "DBSCAN":
                eps = st.slider("📡 eps", 0.1, 10.0, 1.5, 0.1,
                                help="Jarak maksimum antar titik tetangga. Dataset berfitur banyak (mis. Wine) butuh eps lebih besar.")
                min_samples = st.slider("👥 min_samples", 2, 30, 5, 1)
            else:
                k_klaster = st.slider("🔢 Jumlah Klaster (k)", 2, 10, 3, 1)

        test_size = st.slider(
            "✂️ Test Size", 0.10, 0.50, 0.20, 0.05,
            disabled=(tugas == "Clustering"),
            help="Tidak dipakai pada clustering, tetapi tetap dipakai untuk Retrain di menu Database.",
        )
        # 🎲 Random state agar hasil bisa direproduksi
        random_state = st.number_input("🎲 Random State", min_value=0, max_value=9999, value=42, step=1)

    tombol_latih = st.button("🚀 Train Model", use_container_width=True, type="primary")
    st.caption("Dibuat dengan ❤️ menggunakan Streamlit & scikit-learn")

# Muat data (di-cache)
X, y, nama_kelas = muat_dataset(tugas, nama_dataset)

# ------------------------------------------------------------
# 🏋️ Proses training saat tombol ditekan
# ------------------------------------------------------------
if tombol_latih:
    with st.spinner("⏳ Sedang memproses model..."):
        rs = int(random_state)

        if tugas == "Clustering":
            # Standarisasi fitur agar semua fitur setara pengaruhnya
            Xs = StandardScaler().fit_transform(X)
            model = buat_model_cluster(nama_model, k_klaster, eps, min_samples, rs)
            labels = np.asarray(model.fit_predict(Xs))

            # Proyeksi PCA 2D untuk visualisasi
            pca = PCA(n_components=2, random_state=rs)
            Z = pca.fit_transform(Xs)
            pusat = pca.transform(model.cluster_centers_) if nama_model == "KMeans" else None

            # Metrik internal (noise DBSCAN, label -1, dikeluarkan dari perhitungan)
            mask = labels != -1
            n_klaster = len(set(labels[mask]))
            sil = dbi = chi = None
            if n_klaster >= 2 and mask.sum() > n_klaster:
                sil = silhouette_score(Xs[mask], labels[mask])
                dbi = davies_bouldin_score(Xs[mask], labels[mask])
                chi = calinski_harabasz_score(Xs[mask], labels[mask])

            # Elbow plot (hanya untuk KMeans)
            elbow = None
            if nama_model == "KMeans":
                rentang = list(range(2, 11))
                inersia = [
                    KMeans(n_clusters=kk, n_init=10, random_state=rs).fit(Xs).inertia_
                    for kk in rentang
                ]
                elbow = (rentang, inersia)

            # Tabel perbandingan klaster vs label asli
            label_asli = [nama_kelas[i] for i in y] if nama_kelas else list(y)
            tabel_silang = pd.crosstab(
                pd.Series(label_asli, name="Label asli"),
                pd.Series(["Noise" if l == -1 else f"Klaster {l}" for l in labels], name="Klaster"),
            )

            st.session_state.hasil = {
                "tugas": "Clustering",
                "dataset": nama_dataset,
                "model": nama_model,
                "random_state": rs,
                "k": k_klaster, "eps": eps, "min_samples": min_samples,
                "labels": labels,
                "Z": Z,
                "pusat": pusat,
                "var_pca": pca.explained_variance_ratio_,
                "n_klaster": n_klaster,
                "n_noise": int((~mask).sum()),
                "silhouette": sil, "davies_bouldin": dbi, "calinski": chi,
                "ari": adjusted_rand_score(y, labels),
                "elbow": elbow,
                "tabel_silang": tabel_silang,
            }
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=rs,
                stratify=y if tugas == "Klasifikasi" else None,
            )
            model = buat_model(tugas, nama_model, rs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Simpan semua hasil ke session_state
            st.session_state.hasil = {
                "tugas": tugas,
                "dataset": nama_dataset,
                "model": nama_model,
                "test_size": test_size,
                "random_state": rs,
                "y_test": np.array(y_test),
                "y_pred": np.array(y_pred),
                "kelas": nama_kelas,
            }
    st.toast("Model berhasil dilatih! 🎉", icon="✅")

hasil = st.session_state.hasil


# ------------------------------------------------------------
# 📈 Fungsi render per halaman
# ------------------------------------------------------------
def render_visualisasi(hasil, ringkas: bool = False):
    """Grafik hasil model. Mode ringkas dipakai di Dashboard (satu grafik, ukuran kecil)."""
    if hasil is None:
        st.info("👈 Pilih pengaturan di sidebar, lalu klik **🚀 Train Model** untuk memulai.")
        return

    sk = 0.7 if ringkas else 1.0  # faktor skala ukuran gambar
    if not ringkas:
        st.success(
            f"✅ Hasil terakhir: **{hasil['model']}** pada dataset **{hasil['dataset']}** "
            + (
                f"(Clustering, random state {hasil['random_state']})"
                if hasil["tugas"] == "Clustering"
                else f"({hasil['tugas']}, test size {hasil['test_size']:.0%}, random state {hasil['random_state']})"
            )
        )
    sns.set_theme(style="whitegrid", rc=TEMA_GRAFIK)

    if hasil["tugas"] == "Clustering":
        # Siapkan data plot dengan nama klaster; noise diberi warna abu-abu
        nama_label = ["Noise" if l == -1 else f"Klaster {l}" for l in hasil["labels"]]
        df_plot = pd.DataFrame({"PC1": hasil["Z"][:, 0], "PC2": hasil["Z"][:, 1], "Klaster": nama_label})
        urutan = sorted(set(nama_label), key=lambda s: (s == "Noise", s))
        warna = sns.color_palette("tab10", n_colors=max(len(urutan), 3))
        palet = {nm: ("#9CA3AF" if nm == "Noise" else warna[i]) for i, nm in enumerate(urutan)}

        def gambar_scatter():
            st.subheader("🔵 Sebaran Klaster (PCA 2D)")
            fig, ax = plt.subplots(figsize=(7 * sk, 5.5 * sk))
            sns.scatterplot(
                data=df_plot, x="PC1", y="PC2", hue="Klaster", hue_order=urutan,
                palette=palet, s=60, alpha=0.8, edgecolor="white", ax=ax,
            )
            if hasil["pusat"] is not None:
                ax.scatter(hasil["pusat"][:, 0], hasil["pusat"][:, 1], c="black", marker="X",
                           s=220, edgecolor="white", linewidths=1.5, label="Pusat klaster")
            ax.set_xlabel(f"PC1 ({hasil['var_pca'][0]:.0%} varians)")
            ax.set_ylabel(f"PC2 ({hasil['var_pca'][1]:.0%} varians)")
            ax.legend(loc="best", fontsize=8)
            sns.despine()
            st.pyplot(fig)
            plt.close(fig)

        if ringkas:
            gambar_scatter()
            return

        kiri, kanan = st.columns(2)
        with kiri:
            gambar_scatter()
        with kanan:
            if hasil["elbow"] is not None:
                st.subheader("📉 Elbow Plot (KMeans)")
                rentang, inersia = hasil["elbow"]
                fig2, ax2 = plt.subplots(figsize=(7, 5.5))
                ax2.plot(rentang, inersia, "o-", color="#EC4899", lw=2)
                ax2.axvline(hasil["k"], ls="--", color="#BE185D", label=f"k dipilih = {hasil['k']}")
                ax2.set_xlabel("Jumlah klaster (k)")
                ax2.set_ylabel("Inertia")
                ax2.legend()
                sns.despine()
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.subheader("📊 Ukuran Tiap Klaster")
                ukuran = df_plot["Klaster"].value_counts().reindex(urutan)
                fig2, ax2 = plt.subplots(figsize=(7, 5.5))
                sns.barplot(x=ukuran.index, y=ukuran.values, hue=ukuran.index,
                            palette=palet, legend=False, ax=ax2)
                ax2.set_xlabel("")
                ax2.set_ylabel("Jumlah sampel")
                sns.despine()
                st.pyplot(fig2)
                plt.close(fig2)
        return

    # ----- Regresi / Klasifikasi -----
    y_test, y_pred = hasil["y_test"], hasil["y_pred"]
    if hasil["tugas"] == "Regresi":
        st.subheader("📈 Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(8 * sk, 5.5 * sk))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax, color="#EC4899", s=70, alpha=0.75, edgecolor="white")
        batas = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(batas, batas, "--", color="#BE185D", lw=2, label="Prediksi Sempurna")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        sns.despine()
    else:
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        label = hasil["kelas"] if hasil["kelas"] else None
        ukuran = (9, 7) if cm.shape[0] > 5 else (7, 5.5)
        fig, ax = plt.subplots(figsize=(ukuran[0] * sk, ukuran[1] * sk))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="RdPu", ax=ax,
            xticklabels=label if label else "auto",
            yticklabels=label if label else "auto",
            linewidths=0.5, cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    if ringkas:
        st.pyplot(fig)
    else:
        _, col_b, _ = st.columns([1, 3, 1])
        with col_b:
            st.pyplot(fig)
    plt.close(fig)  # bebaskan memori


def render_evaluasi(hasil):
    """Metrik evaluasi lengkap."""
    if hasil is None:
        st.info("👈 Latih model terlebih dahulu untuk melihat metrik evaluasi.")
        return

    if hasil["tugas"] == "Clustering":
        st.subheader("📋 Metrik Clustering")

        def fmt(nilai):
            return f"{nilai:.3f}" if nilai is not None else "–"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔵 Silhouette", fmt(hasil["silhouette"]))
        m2.metric("📉 Davies-Bouldin", fmt(hasil["davies_bouldin"]))
        m3.metric("📈 Calinski-Harabasz", fmt(hasil["calinski"]))
        m4.metric("🎯 Adjusted Rand Index", fmt(hasil["ari"]))
        st.caption(
            "Silhouette (−1 s.d. 1) dan Calinski-Harabasz: makin **tinggi** makin baik. "
            "Davies-Bouldin: makin **rendah** makin baik. "
            "Adjusted Rand Index membandingkan klaster dengan label asli (1 = identik, ≈0 = acak)."
        )

        n1, n2 = st.columns(2)
        n1.metric("🔢 Jumlah Klaster", hasil["n_klaster"])
        n2.metric("🚫 Titik Noise", hasil["n_noise"])
        if hasil["silhouette"] is None:
            st.warning("⚠️ Metrik internal tidak dapat dihitung karena klaster yang terbentuk kurang dari 2. "
                       "Untuk DBSCAN, coba ubah `eps` atau `min_samples`.")

        st.markdown("#### 🔀 Klaster vs Label Asli")
        st.dataframe(hasil["tabel_silang"], use_container_width=True)
        return

    y_test, y_pred = hasil["y_test"], hasil["y_pred"]
    st.subheader("📋 Metrik Evaluasi")

    if hasil["tugas"] == "Regresi":
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📏 MAE", f"{mae:.3f}")
        m2.metric("📐 MSE", f"{mse:.3f}")
        m3.metric("📊 RMSE", f"{rmse:.3f}")
        m4.metric("🏆 R²", f"{r2:.3f}")
    else:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Accuracy", f"{acc:.3f}")
        m2.metric("🔍 Precision", f"{prec:.3f}")
        m3.metric("📡 Recall", f"{rec:.3f}")
        m4.metric("⚖️ F1-Score", f"{f1:.3f}")
        st.caption("Precision, Recall, dan F1 dihitung dengan rata-rata *weighted*.")

        st.markdown("#### 📝 Classification Report")
        laporan = classification_report(
            y_test, y_pred, target_names=hasil["kelas"] if hasil["kelas"] else None,
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(laporan).T.round(3), use_container_width=True)


def render_database():
    """MongoDB CRUD data Iris + Retrain Random Forest."""
    if not cek_koneksi():
        return

    # ---------- 📊 Statistik collection ----------
    try:
        df_mongo = ambil_data_mongo()
    except PyMongoError as e:
        st.error(f"⚠️ Gagal mengambil data dari MongoDB: {e}")
        df_mongo = pd.DataFrame(columns=["_id"] + FITUR_IRIS + ["species", "timestamp"])

    s1, s2, s3 = st.columns(3)
    s1.metric("📦 Total Dokumen", f"{len(df_mongo):,}")
    s2.metric("🗃️ Database", NAMA_DB)
    s3.metric("📁 Collection", NAMA_COLLECTION)

    # ---------- ➕ Form insert ----------
    with kartu("db_insert"):
        judul_kartu("➕ Tambah Data Iris")
        with st.form("form_insert", clear_on_submit=True):
            f1, f2, f3, f4 = st.columns(4)
            sepal_length = f1.number_input("🌿 Sepal Length", 0.0, 15.0, 5.0, 0.1)
            sepal_width = f2.number_input("🌿 Sepal Width", 0.0, 15.0, 3.0, 0.1)
            petal_length = f3.number_input("🌸 Petal Length", 0.0, 15.0, 1.5, 0.1)
            petal_width = f4.number_input("🌸 Petal Width", 0.0, 15.0, 0.2, 0.1)
            species = st.selectbox("🏷️ Species", SPECIES_LIST)
            kirim = st.form_submit_button("💾 Simpan ke MongoDB", type="primary")

    if kirim:
        try:
            get_collection().insert_one({
                "sepal_length": float(sepal_length),
                "sepal_width": float(sepal_width),
                "petal_length": float(petal_length),
                "petal_width": float(petal_width),
                "species": species,
                "timestamp": datetime.now(),  # timestamp otomatis
            })
            bersihkan_cache_data()
            st.toast("Data berhasil disimpan! ✅", icon="💾")
            st.rerun()  # muat ulang halaman agar tabel langsung terbarui
        except PyMongoError as e:
            st.error(f"❌ Gagal menyimpan data: {e}")

    # ---------- 📋 Tabel data + 🗑️ hapus ----------
    with kartu("db_tabel"):
        judul_kartu("📋 Data di MongoDB")
        if df_mongo.empty:
            st.info("📭 Collection masih kosong. Tambahkan data lewat form di atas.")
        else:
            st.dataframe(df_mongo, use_container_width=True, hide_index=True)

            st.markdown("**🗑️ Hapus Data**")
            d1, d2 = st.columns([3, 1])
            id_dipilih = d1.selectbox("Pilih `_id` dokumen yang akan dihapus", df_mongo["_id"].tolist())
            d2.write("")  # spasi agar tombol sejajar
            d2.write("")
            hapus = d2.button("🗑️ Hapus", use_container_width=True)

            if hapus:
                try:
                    hasil_hapus = get_collection().delete_one({"_id": ObjectId(id_dipilih)})
                    if hasil_hapus.deleted_count == 1:
                        bersihkan_cache_data()
                        st.toast("Data berhasil dihapus! 🗑️", icon="✅")
                        st.rerun()
                    else:
                        st.warning("⚠️ Dokumen tidak ditemukan (mungkin sudah dihapus).")
                except PyMongoError as e:
                    st.error(f"❌ Gagal menghapus data: {e}")

    # ---------- 🔁 Retrain ----------
    with kartu("db_retrain"):
        judul_kartu("🔁 Retrain Random Forest")
        st.caption(
            "Menggabungkan 150 sampel Iris dari scikit-learn dengan data dari MongoDB, "
            f"lalu melatih ulang Random Forest (random state = {int(random_state)})."
        )

        if st.button("🔁 Retrain Sekarang", type="primary"):
            with st.spinner("⏳ Menggabungkan data dan melatih model..."):
                # Data bawaan sklearn (150 sampel), kolom diseragamkan dengan skema MongoDB
                iris = datasets.load_iris()
                df_sk = pd.DataFrame(iris.data, columns=FITUR_IRIS)
                df_sk["target"] = iris.target  # 0, 1, 2

                # Data MongoDB: konversi species string -> integer
                df_db = df_mongo.copy()
                if not df_db.empty:
                    df_db = df_db[df_db["species"].isin(SPECIES_MAP)].dropna(subset=FITUR_IRIS)
                    df_db["target"] = df_db["species"].map(SPECIES_MAP)
                    df_db = df_db[FITUR_IRIS + ["target"]].astype(
                        {**{f: float for f in FITUR_IRIS}, "target": int}
                    )

                # Gabungkan kedua sumber data
                df_gabung = pd.concat([df_sk, df_db], ignore_index=True) if not df_db.empty else df_sk
                Xg, yg = df_gabung[FITUR_IRIS], df_gabung["target"]

                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xg, yg, test_size=test_size,
                    random_state=int(random_state), stratify=yg,
                )
                rf = RandomForestClassifier(n_estimators=100, random_state=int(random_state))
                rf.fit(X_tr, y_tr)
                y_hat = rf.predict(X_te)

                st.session_state.hasil_retrain = {
                    "n_sklearn": len(df_sk),
                    "n_mongo": len(df_db),
                    "n_total": len(df_gabung),
                    "akurasi": accuracy_score(y_te, y_hat),
                    "laporan": pd.DataFrame(
                        classification_report(
                            y_te, y_hat, target_names=SPECIES_LIST,
                            output_dict=True, zero_division=0,
                        )
                    ).T.round(3),
                    "importance": pd.Series(rf.feature_importances_, index=FITUR_IRIS).sort_values(),
                    "random_state": int(random_state),
                }
            st.toast("Retrain selesai! 🎉", icon="✅")

        # Tampilkan hasil retrain terakhir (tetap ada saat rerun)
        hr = st.session_state.hasil_retrain
        if hr is not None:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("🌺 Data sklearn", hr["n_sklearn"])
            r2.metric("🍃 Data MongoDB", hr["n_mongo"])
            r3.metric("📦 Total Data", hr["n_total"])
            r4.metric("🎯 Akurasi", f"{hr['akurasi']:.3f}")

            col_kiri, col_kanan = st.columns(2)
            with col_kiri:
                st.markdown("#### 📝 Classification Report")
                st.dataframe(hr["laporan"], use_container_width=True)
            with col_kanan:
                st.markdown("#### ⭐ Feature Importance")
                sns.set_theme(style="whitegrid", rc=TEMA_GRAFIK)
                fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
                sns.barplot(
                    x=hr["importance"].values, y=hr["importance"].index,
                    hue=hr["importance"].index,
                    palette=sns.blend_palette(["#F9A8D4", "#BE185D"], n_colors=len(hr["importance"])),
                    legend=False, ax=ax_fi,
                )
                ax_fi.set_xlabel("Importance")
                ax_fi.set_ylabel("")
                sns.despine()
                st.pyplot(fig_fi)
                plt.close(fig_fi)


def render_komentar():
    """Form komentar + daftar komentar dari MongoDB."""
    if not cek_koneksi():
        return

    # ---------- ✍️ Form komentar ----------
    with kartu("komentar_form"):
        judul_kartu("✍️ Tulis Komentar")
        st.caption("Tinggalkan komentar atau saran untuk aplikasi ini. Komentar tersimpan di MongoDB.")
        with st.form("form_komentar", clear_on_submit=True):
            nama_komentar = st.text_input("👤 Nama (opsional)", max_chars=50, placeholder="Anonim")
            isi_komentar = st.text_area("💬 Komentar", max_chars=500, height=120,
                                        placeholder="Tulis komentar Anda di sini...")
            kirim_komentar = st.form_submit_button("📨 Kirim Komentar", type="primary")

    if kirim_komentar:
        if not isi_komentar.strip():
            st.warning("⚠️ Komentar tidak boleh kosong.")
        else:
            try:
                get_collection_komentar().insert_one({
                    "nama": nama_komentar.strip() or "Anonim",
                    "komentar": isi_komentar.strip(),
                    "timestamp": datetime.now(timezone.utc),  # disimpan dalam UTC
                })
                bersihkan_cache_data()
                st.toast("Komentar terkirim! 🎉", icon="💬")
                st.rerun()
            except PyMongoError as e:
                st.error(f"❌ Gagal menyimpan komentar: {e}")

    # ---------- 📜 Daftar komentar ----------
    try:
        daftar_komentar = ambil_komentar()
    except PyMongoError as e:
        st.error(f"⚠️ Gagal mengambil komentar: {e}")
        daftar_komentar = []

    with kartu("komentar_list"):
        judul_kartu("📜 Daftar Komentar")
        st.metric("💬 Jumlah Komentar (maks. 100 terbaru ditampilkan)", len(daftar_komentar))

        if not daftar_komentar:
            st.info("📭 Belum ada komentar. Jadilah yang pertama!")
        else:
            for k in daftar_komentar:
                waktu = k["timestamp"]
                teks_waktu = (waktu + WIB).strftime("%d %b %Y, %H:%M") + " WIB" if waktu else "-"
                with st.container(border=True):
                    st.markdown(f"**👤 {escape_md(k['nama'])}** · 🕒 {teks_waktu}")
                    st.markdown(escape_md(k["komentar"]))


# ------------------------------------------------------------
# 🧭 Routing halaman sesuai menu sidebar
# ------------------------------------------------------------
if halaman == MENU[0]:
    # ===== Dashboard: kolom utama (kiri) + panel skor (kanan) =====
    header_halaman(
        "HALO, SELAMAT DATANG! 👋",
        "Regresi, Klasifikasi, dan Clustering secara interaktif ✨",
        [f"🎯 {tugas}", f"📚 {nama_dataset}", f"🧩 {nama_model}", f"🎲 Seed {int(random_state)}"],
    )
    st.write("")

    utama, panel = st.columns([3, 1.05], gap="large")

    with utama:
        a, b = st.columns(2, gap="medium")
        with a:
            with kartu("ringkas"):
                judul_kartu("📊 Ringkasan Dataset")
                m1, m2 = st.columns(2)
                m1.metric("📏 Sampel", f"{X.shape[0]:,}")
                m2.metric("🧮 Fitur", X.shape[1])
        with b:
            with kartu("status"):
                judul_kartu("🧠 Status Model")
                if hasil is None:
                    st.info("Belum ada model. Klik **🚀 Train Model** di sidebar.")
                else:
                    st.success(f"**{hasil['model']}** pada **{hasil['dataset']}**")
                    st.caption(f"Tugas: {hasil['tugas']} · random state {hasil['random_state']}")

        with kartu("cuplikan"):
            judul_kartu("👀 Cuplikan Dataset")
            st.dataframe(pd.concat([X, y], axis=1).head(10), use_container_width=True)

        with kartu("hasil"):
            judul_kartu("📈 Visualisasi Terbaru")
            render_visualisasi(hasil, ringkas=True)

    with panel:
        with st.container(key="panel_kanan"):
            warna = ["#F9A8D4", "#5EEAD4", "#FCD34D"]
            kartu_skor = "".join(
                gauge_html(j, p, t, warna[i]) for i, (j, p, t) in enumerate(skor_ringkas(hasil))
            )
            st.markdown(kartu_skor, unsafe_allow_html=True)
            st.markdown('<div class="panel-link">Lihat detail di menu 📋 Evaluasi</div>',
                        unsafe_allow_html=True)

elif halaman == MENU[1]:
    header_halaman("📈 VISUALISASI", "Grafik hasil model terakhir yang dilatih.")
    st.write("")
    with kartu("visual"):
        render_visualisasi(hasil)

elif halaman == MENU[2]:
    header_halaman("📋 EVALUASI", "Metrik performa model terakhir yang dilatih.")
    st.write("")
    with kartu("eval"):
        render_evaluasi(hasil)

elif halaman == MENU[3]:
    header_halaman("🗄️ DATABASE & RETRAIN", "Kelola data Iris di MongoDB dan latih ulang Random Forest.")
    st.write("")
    render_database()

else:
    header_halaman("💬 KOMENTAR", "Komentar pengunjung tersimpan di MongoDB.")
    st.write("")
    render_komentar()