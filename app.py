import os
import html
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from datetime import date, timedelta
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TODAY = date.today()
IMAGE_BASE = "https://image.tmdb.org/t/p/"


# ==========================================================
# 2. THEMES
# ==========================================================

THEMES = {
    "Retro": {"bg":"#21140f", "panel":"#382219", "text":"#fff1d6", "muted":"#d9b995", "accent":"#ffb02e", "border":"#795033", "font":"Georgia, serif", "radius":"5px", "bg_image":"https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=1800", "hero":"linear-gradient(90deg,rgba(36,18,10,.94),rgba(95,43,20,.60)),url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=1800')", "texture":"repeating-linear-gradient(0deg,rgba(255,190,80,.025) 0 1px,transparent 1px 4px)"},
    "Traditional": {"bg":"#f4ead8", "panel":"#fff8ed", "text":"#40251c", "muted":"#80614d", "accent":"#a83226", "border":"#d9b98b", "font":"Georgia, serif", "radius":"14px", "hero":"linear-gradient(100deg,rgba(48,20,18,.86),rgba(119,52,29,.52)),url('https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?w=1800')", "texture":"radial-gradient(circle at 90% 10%,rgba(190,132,54,.15),transparent 30%)"},
    "Futuristic": {"bg":"#080b1d", "panel":"#111936", "text":"#eff6ff", "muted":"#a5b4d5", "accent":"#42e8ff", "border":"#354c82", "font":"Arial, sans-serif", "radius":"18px", "hero":"linear-gradient(100deg,rgba(5,9,30,.92),rgba(25,24,68,.48)),url('https://images.unsplash.com/photo-1519608487953-e999c86e7455?w=1800')", "texture":"radial-gradient(ellipse at 80% 20%,rgba(66,232,255,.12),transparent 35%)"},
    "Kids": {"bg":"#fff8e7", "panel":"#ffffff", "text":"#26334d", "muted":"#68738a", "accent":"#ff6b8a", "border":"#ffd0a8", "font":"Trebuchet MS, sans-serif", "radius":"24px", "hero":"linear-gradient(100deg,rgba(29,65,116,.72),rgba(255,116,160,.35)),url('https://images.unsplash.com/photo-1470252649378-9c29740c9fa8?w=1800')", "texture":"radial-gradient(circle at 15% 20%,rgba(255,210,70,.22),transparent 22%),radial-gradient(circle at 85% 70%,rgba(255,107,138,.16),transparent 24%)"},
    "Natural": {"bg":"#e9efe2", "panel":"#f7faf3", "text":"#243b2b", "muted":"#627565", "accent":"#477a4b", "border":"#b8cbb5", "font":"Georgia, serif", "radius":"16px", "hero":"linear-gradient(100deg,rgba(18,48,31,.78),rgba(37,75,45,.32)),url('https://images.unsplash.com/photo-1448375240586-882707db888b?w=1800')", "texture":"radial-gradient(ellipse at 80% 20%,rgba(86,130,73,.15),transparent 38%)"},
    "Anime": {"bg":"#17152c", "panel":"#252044", "text":"#fff3ff", "muted":"#d0bfe3", "accent":"#ff78c8", "border":"#624a86", "font":"Trebuchet MS, sans-serif", "radius":"12px", "hero":"linear-gradient(100deg,rgba(25,17,53,.86),rgba(96,34,104,.40)),url('https://images.unsplash.com/photo-1519608487953-e999c86e7455?w=1800')", "texture":"radial-gradient(circle at 80% 20%,rgba(255,120,200,.18),transparent 30%)"},
    "Romance": {"bg":"#2a1720", "panel":"#412432", "text":"#fff0f2", "muted":"#e0b9c3", "accent":"#ff91ae", "border":"#805065", "font":"Georgia, serif", "radius":"20px", "hero":"linear-gradient(100deg,rgba(52,22,39,.86),rgba(145,58,89,.36)),url('https://images.unsplash.com/photo-1518621736915-f3b1c41bfd00?w=1800')", "texture":"radial-gradient(ellipse at 75% 15%,rgba(255,145,174,.18),transparent 35%)"},
    "Thriller": {"bg":"#101216", "panel":"#1c2025", "text":"#f1f1f1", "muted":"#a5aab0", "accent":"#e5484d", "border":"#44464d", "font":"Arial, sans-serif", "radius":"3px", "hero":"linear-gradient(100deg,rgba(7,10,14,.92),rgba(67,16,23,.58)),url('https://images.unsplash.com/photo-1519608487953-e999c86e7455?w=1800')", "texture":"linear-gradient(135deg,transparent 48%,rgba(229,72,77,.06) 49%,transparent 51%)"},
    "Nostalgia": {"bg":"#2b2923", "panel":"#3b382f", "text":"#f5e8c9", "muted":"#c4b797", "accent":"#d5a65b", "border":"#6e624c", "font":"Courier New, monospace", "radius":"4px", "hero":"linear-gradient(100deg,rgba(40,34,23,.88),rgba(105,76,43,.44)),url('https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=1800')", "texture":"repeating-linear-gradient(0deg,rgba(255,255,255,.025) 0 1px,transparent 1px 3px)"},
    "Classic Light": {"bg":"#f4f6fa", "panel":"#ffffff", "text":"#1a2030", "muted":"#687386", "accent":"#536de8", "border":"#dce1eb", "font":"Arial, sans-serif", "radius":"12px", "hero":"linear-gradient(100deg,rgba(20,32,60,.72),rgba(70,95,145,.28)),url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=1800')", "texture":"none"},
    "Midnight": {"bg":"#090b10", "panel":"#151923", "text":"#f4f5f7", "muted":"#a2a8b5", "accent":"#8b9cff", "border":"#292e3a", "font":"Arial, sans-serif", "radius":"12px", "hero":"linear-gradient(100deg,rgba(9,11,16,.90),rgba(21,25,35,.52)),url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=1800')", "texture":"none"}
}

if "theme" not in st.session_state or st.session_state.theme not in THEMES:
    st.session_state.theme = "Retro"

theme = THEMES[st.session_state.theme]


# ==========================================================
# 3. RESPONSIVE CSS
# ==========================================================

st.markdown(
    f"""
    <style>
    :root {{
        --app-bg: {theme["bg"]};
        --app-panel: {theme["panel"]};
        --app-text: {theme["text"]};
        --app-muted: {theme["muted"]};
        --app-accent: {theme["accent"]};
        --app-border: {theme["border"]};
        --app-radius: {theme["radius"]};
        --app-font: {theme["font"]};
    }}

    .stApp {{
        background: var(--app-bg);
        color: var(--app-text);
        font-family: var(--app-font);
        background-image: {theme["texture"]}, {theme["hero"]};
        background-size: auto, cover;
        background-position: center, center;
        background-attachment: fixed, fixed;
    }}

    [data-testid="stMain"] {{ background: transparent; }}
    [data-testid="stMainBlockContainer"] {{ max-width: 1700px; }}
    [data-testid="stSidebar"] {{ background-image: {theme["texture"]}; }}

    [data-testid="stSidebar"] {{
        background: var(--app-panel);
        border-right: 1px solid var(--app-border);
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    h1, h2, h3, h4, p, label {{
        color: var(--app-text);
    }}

    .hero {{
        background: {theme["hero"]};
        background-size: cover;
        background-position: center;
        border: 1px solid var(--app-border);
        border-radius: var(--app-radius);
        padding: clamp(18px, 4vw, 40px);
        margin: 10px 0 24px;
    }}

    .hero-title {{
        font-size: clamp(25px, 4vw, 42px);
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 10px;
    }}

    .hero-subtitle {{
        color: var(--app-muted);
        font-size: clamp(13px, 1.5vw, 17px);
    }}

    .movie-title {{
        font-size: clamp(12px, 1.1vw, 15px);
        font-weight: 700;
        line-height: 1.3;
        min-height: 2.6em;
        margin-top: 7px;
        overflow-wrap: anywhere;
    }}

    .movie-meta {{
        font-size: 12px;
        color: var(--app-muted);
    }}

    .section-heading {{
        font-size: clamp(19px, 2vw, 25px);
        font-weight: 750;
        margin: 22px 0 12px;
    }}

    div.stButton > button {{
        border-radius: 10px;
        min-height: 38px;
        white-space: normal;
        overflow-wrap: anywhere;
    }}

    [data-testid="stImage"] img {{
        border-radius: 8px;
        object-fit: cover;
    }}

    [data-testid="stHorizontalBlock"] {{
        gap: 0.8rem !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: var(--app-radius) !important;
        border-color: var(--app-border) !important;
        transition: transform .18s ease, box-shadow .18s ease;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(0,0,0,.18);
    }}

    div.stButton > button {{
        transition: all .18s ease;
    }}

    div.stButton > button:hover {{
        border-color: var(--app-accent);
        color: var(--app-accent);
    }}

    @media (max-width: 600px) {{
        .hero {{
            padding: 18px;
            border-radius: 12px;
        }}

        div.stButton > button {{
            font-size: 12px;
            padding: 0.35rem 0.45rem;
        }}

        [data-testid="stMainBlockContainer"] {{
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }}
    }}

    @media (min-width: 1600px) {{
        [data-testid="stMainBlockContainer"] {{
            max-width: 1800px;
            margin: auto;
        }}

        .movie-title {{
            font-size: 17px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# 4. LOAD TRAINED MODEL
# ==========================================================

@st.cache_resource
def load_pickle(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file: {filename}"
        )

    return joblib.load(path)


@st.cache_resource
def load_models():
    names = [
        "tfidf_vectorizer.pkl",
        "tfidf_matrix.pkl",
        "movie_indices.pkl",
        "svd_model.pkl",
        "user_to_idx.pkl",
        "movie_to_idx.pkl",
        "user_factors.pkl",
        "movie_factors.pkl",
        "movies_model.pkl",
        "user_rated_movies.pkl"
    ]

    missing = [n for n in names if not (MODEL_DIR / n).exists()]

    if missing:
        raise FileNotFoundError(
            "Missing files in models/: " + ", ".join(missing)
        )

    return {
        "vectorizer": load_pickle("tfidf_vectorizer.pkl"),
        "tfidf": load_pickle("tfidf_matrix.pkl"),
        "movie_indices": load_pickle("movie_indices.pkl"),
        "svd": load_pickle("svd_model.pkl"),
        "user_to_idx": load_pickle("user_to_idx.pkl"),
        "movie_to_idx": load_pickle("movie_to_idx.pkl"),
        "user_factors": load_pickle("user_factors.pkl"),
        "movie_factors": load_pickle("movie_factors.pkl"),
        "movies": load_pickle("movies_model.pkl"),
        "user_rated_movies": load_pickle("user_rated_movies.pkl")
    }


try:
    MODEL = load_models()
    MODEL_READY = True
    MODEL_ERROR = ""
except Exception as e:
    MODEL = {}
    MODEL_READY = False
    MODEL_ERROR = str(e)


def prepare_movies():
    if not MODEL_READY:
        return pd.DataFrame()

    movies = MODEL["movies"].copy()

    if "id" not in movies.columns:
        return pd.DataFrame()

    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["id"]).copy()
    movies["id"] = movies["id"].astype(int)

    if "title" not in movies.columns:
        movies["title"] = "Unknown title"

    movies["title"] = movies["title"].fillna("Unknown title")

    if "vote_average" not in movies.columns:
        movies["vote_average"] = 0

    if "overview" not in movies.columns:
        movies["overview"] = ""

    if "release_date" not in movies.columns:
        movies["release_date"] = ""

    return movies.reset_index(drop=True)


MOVIES = prepare_movies()


# ==========================================================
# 5. TMDB API + LOCAL FALLBACK
# ==========================================================

def get_api_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.getenv("TMDB_API_KEY", "")


TMDB_KEY = get_api_key()


@st.cache_data(ttl=1800, show_spinner=False)
def tmdb_get(endpoint, params=None):
    if not TMDB_KEY:
        return {}

    query_params = dict(params or {})
    query_params["api_key"] = TMDB_KEY

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/" + endpoint.lstrip("/"),
            params=query_params,
            timeout=12
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return {}


def image_url(path, size="w500"):
    if not path:
        return None

    return f"{IMAGE_BASE}{size}{path}"


GENRES = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37
}


# ==========================================================
# LOCAL FALLBACK HELPERS
# ==========================================================

def local_movie_records(frame):
    """
    Convert local MOVIES rows into the same basic dictionary
    structure used by TMDB movie objects.
    """

    if frame is None or frame.empty:
        return []

    result = []

    for _, row in frame.iterrows():
        movie = {}

        for column, value in row.items():
            if pd.isna(value):
                if column in ["vote_average"]:
                    value = 0
                else:
                    value = ""

            movie[column] = value

        try:
            movie["id"] = int(movie["id"])
        except (ValueError, TypeError):
            continue

        movie["title"] = str(
            movie.get("title", "Untitled")
        )

        movie["overview"] = str(
            movie.get("overview", "") or ""
        )

        movie["release_date"] = str(
            movie.get("release_date", "") or ""
        )

        try:
            movie["vote_average"] = float(
                movie.get("vote_average", 0) or 0
            )
        except (ValueError, TypeError):
            movie["vote_average"] = 0

        result.append(movie)

    return result


def local_search_movies(query):
    """
    Search the trained local movie catalog when TMDB search
    is unavailable.
    """

    if MOVIES.empty or not query.strip():
        return []

    query_lower = query.strip().lower()

    titles = MOVIES["title"].astype(str)

    exact = MOVIES[
        titles.str.lower() == query_lower
    ].copy()

    partial = MOVIES[
        titles.str.lower().str.contains(
            query_lower,
            na=False,
            regex=False
        )
    ].copy()

    if not exact.empty:
        partial = pd.concat(
            [exact, partial],
            ignore_index=True
        ).drop_duplicates("id")

    return local_movie_records(
        partial.head(20)
    )


def local_discover_movies(category, page=1, per_page=20):
    """
    Local replacement for TMDB discovery endpoints.

    The local catalog is used only when TMDB does not provide
    results.
    """

    if MOVIES.empty:
        return []

    movies = MOVIES.copy()

    # ------------------------------------------------------
    # Release date preparation
    # ------------------------------------------------------

    if "release_date" in movies.columns:
        movies["_release_date"] = pd.to_datetime(
            movies["release_date"],
            errors="coerce"
        )
    else:
        movies["_release_date"] = pd.NaT

    # ------------------------------------------------------
    # Popular
    # ------------------------------------------------------

    if category == "Popular":

        if "popularity" in movies.columns:
            movies = movies.sort_values(
                "popularity",
                ascending=False,
                na_position="last"
            )
        else:
            movies = movies.sort_values(
                "vote_average",
                ascending=False,
                na_position="last"
            )

    # ------------------------------------------------------
    # Highest Rated
    # ------------------------------------------------------

    elif category == "Highest Rated":

        movies = movies.sort_values(
            "vote_average",
            ascending=False,
            na_position="last"
        )

    # ------------------------------------------------------
    # Latest
    # ------------------------------------------------------

    elif category == "Latest":

        movies = movies[
            movies["_release_date"].notna()
        ].sort_values(
            "_release_date",
            ascending=False
        )

    # ------------------------------------------------------
    # Upcoming
    # ------------------------------------------------------

    elif category == "Upcoming":

        movies = movies[
            movies["_release_date"] > pd.Timestamp(TODAY)
        ].sort_values(
            "_release_date",
            ascending=True
        )

    # ------------------------------------------------------
    # Now Playing
    # ------------------------------------------------------

    elif category == "Now Playing":

        start_date = pd.Timestamp(
            TODAY - timedelta(days=60)
        )

        movies = movies[
            (movies["_release_date"] >= start_date) &
            (movies["_release_date"] <= pd.Timestamp(TODAY))
        ].sort_values(
            "_release_date",
            ascending=False
        )

    # ------------------------------------------------------
    # Trending
    # ------------------------------------------------------

    elif category == "Trending":

        if "popularity" in movies.columns:
            movies = movies.sort_values(
                "popularity",
                ascending=False,
                na_position="last"
            )
        else:
            movies = movies.sort_values(
                "vote_average",
                ascending=False,
                na_position="last"
            )

    # ------------------------------------------------------
    # Kids & Family
    # ------------------------------------------------------

    elif category == "Kids & Family":

        # If genre information exists, try to use it.
        genre_columns = [
            column for column in movies.columns
            if "genre" in column.lower()
        ]

        if genre_columns:

            genre_column = genre_columns[0]

            mask = movies[genre_column].astype(
                str
            ).str.contains(
                "family|animation|kids|children",
                case=False,
                na=False,
                regex=True
            )

            filtered = movies[mask].copy()

            if not filtered.empty:
                movies = filtered

        # If no genre information is available, retain the
        # catalog rather than returning an empty page.
        movies = movies.sort_values(
            "vote_average",
            ascending=False,
            na_position="last"
        )

    else:
        return []

    movies = movies.drop_duplicates(
        subset=["id"]
    )

    start = max(0, (page - 1) * per_page)
    end = start + per_page

    movies = movies.iloc[start:end]

    return local_movie_records(movies)


def local_movie_details(movie_id):
    """
    Return local metadata for a movie when TMDB details are
    unavailable.
    """

    if MOVIES.empty or movie_id is None:
        return {}

    try:
        movie_id = int(movie_id)
    except (ValueError, TypeError):
        return {}

    matches = MOVIES[
        MOVIES["id"] == movie_id
    ]

    if matches.empty:
        return {}

    records = local_movie_records(
        matches.head(1)
    )

    if not records:
        return {}

    return records[0]


def local_similar_movies(movie_id, count=20):
    """
    Use the trained hybrid model as the local fallback for
    similar/recommended movies.
    """

    if not MODEL_READY or MOVIES.empty:
        return []

    try:
        movie_id = int(movie_id)
    except (ValueError, TypeError):
        return []

    recommendations = hybrid_recommend(
        movie_id,
        count
    )

    if recommendations.empty:
        return []

    return recommendations.to_dict(
        "records"
    )


def local_filtered_movies(
    genre,
    year_range,
    minimum_rating,
    sort_by
):
    """
    Local fallback for the Search/Filters page.
    """

    if MOVIES.empty:
        return []

    movies = MOVIES.copy()

    # ------------------------------------------------------
    # Rating
    # ------------------------------------------------------

    if "vote_average" in movies.columns:
        movies["vote_average"] = pd.to_numeric(
            movies["vote_average"],
            errors="coerce"
        ).fillna(0)

        movies = movies[
            movies["vote_average"] >= minimum_rating
        ]

    # ------------------------------------------------------
    # Release year
    # ------------------------------------------------------

    if "release_date" in movies.columns:

        movies["_release_date"] = pd.to_datetime(
            movies["release_date"],
            errors="coerce"
        )

        movies = movies[
            (
                movies["_release_date"].dt.year >=
                year_range[0]
            )
            &
            (
                movies["_release_date"].dt.year <=
                year_range[1]
            )
        ]

    # ------------------------------------------------------
    # Genre
    # ------------------------------------------------------

    if genre != "All genres":

        genre_columns = [
            column for column in movies.columns
            if "genre" in column.lower()
        ]

        if genre_columns:

            genre_column = genre_columns[0]

            movies = movies[
                movies[genre_column]
                .astype(str)
                .str.contains(
                    genre,
                    case=False,
                    na=False,
                    regex=False
                )
            ]

    # ------------------------------------------------------
    # Sorting
    # ------------------------------------------------------

    if sort_by == "Rating":

        movies = movies.sort_values(
            "vote_average",
            ascending=False,
            na_position="last"
        )

    elif sort_by == "Release date":

        if "_release_date" not in movies.columns:
            movies["_release_date"] = pd.to_datetime(
                movies["release_date"],
                errors="coerce"
            )

        movies = movies.sort_values(
            "_release_date",
            ascending=False,
            na_position="last"
        )

    else:

        if "popularity" in movies.columns:
            movies = movies.sort_values(
                "popularity",
                ascending=False,
                na_position="last"
            )
        else:
            movies = movies.sort_values(
                "vote_average",
                ascending=False,
                na_position="last"
            )

    return local_movie_records(
        movies.head(20)
    )


# ==========================================================
# TMDB DISCOVERY
# ==========================================================

def discover_movies(category, page=1):

    if category == "Trending":
        endpoint = "trending/movie/day"
        params = {"page": page}

    elif category == "Popular":
        endpoint = "movie/popular"
        params = {"page": page}

    elif category == "Now Playing":
        endpoint = "movie/now_playing"
        params = {"page": page}

    elif category == "Upcoming":
        endpoint = "movie/upcoming"
        params = {"page": page}

    elif category == "Highest Rated":
        endpoint = "movie/top_rated"
        params = {"page": page}

    elif category == "Latest":
        endpoint = "discover/movie"
        params = {
            "page": page,
            "sort_by": "primary_release_date.desc",
            "primary_release_date.gte":
                (TODAY - timedelta(days=365)).isoformat(),
            "primary_release_date.lte":
                TODAY.isoformat(),
            "vote_count.gte": 5
        }

    elif category == "Kids & Family":
        endpoint = "discover/movie"
        params = {
            "page": page,
            "with_genres":
                f"{GENRES['Animation']},{GENRES['Family']}",
            "sort_by": "popularity.desc",
            "include_adult": "false"
        }

    else:
        return []

    # ------------------------------------------------------
    # Primary source: TMDB
    # ------------------------------------------------------

    result = tmdb_get(
        endpoint,
        params
    )

    tmdb_results = (
        result.get("results", [])
        if isinstance(result, dict)
        else []
    )

    if tmdb_results:
        return tmdb_results

    # ------------------------------------------------------
    # Fallback: local catalog
    # ------------------------------------------------------

    return local_discover_movies(
        category,
        page=page
    )


# ==========================================================
# SEARCH
# ==========================================================

def search_movies(query):

    if not query.strip():
        return []

    # ------------------------------------------------------
    # Primary source: TMDB
    # ------------------------------------------------------

    result = tmdb_get(
        "search/movie",
        {
            "query": query.strip(),
            "include_adult": "false",
            "page": 1
        }
    )

    tmdb_results = (
        result.get("results", [])
        if isinstance(result, dict)
        else []
    )

    if tmdb_results:
        return tmdb_results

    # ------------------------------------------------------
    # Fallback: local catalog
    # ------------------------------------------------------

    return local_search_movies(query)


# ==========================================================
# MOVIE DETAILS
# ==========================================================

def tmdb_details(movie_id):

    if not movie_id:
        return {}

    result = tmdb_get(
        f"movie/{int(movie_id)}"
    )

    if isinstance(result, dict) and result:
        return result

    # ------------------------------------------------------
    # Fallback: local movie metadata
    # ------------------------------------------------------

    return local_movie_details(movie_id)


# ==========================================================
# SIMILAR MOVIES
# ==========================================================

def tmdb_similar(movie_id):

    if not movie_id:
        return []

    # ------------------------------------------------------
    # Primary source: TMDB recommendations
    # ------------------------------------------------------

    result = tmdb_get(
        f"movie/{int(movie_id)}/recommendations"
    )

    movies = (
        result.get("results", [])
        if isinstance(result, dict)
        else []
    )

    # ------------------------------------------------------
    # TMDB similar endpoint
    # ------------------------------------------------------

    if len(movies) < 10:

        similar = tmdb_get(
            f"movie/{int(movie_id)}/similar"
        )

        if isinstance(similar, dict):

            movies += similar.get(
                "results",
                []
            )

    # ------------------------------------------------------
    # Remove duplicates
    # ------------------------------------------------------

    unique = []
    seen = set()

    for movie in movies:

        mid = movie.get("id")

        if mid and mid not in seen:
            seen.add(mid)
            unique.append(movie)

    if unique:
        return unique

    # ------------------------------------------------------
    # Fallback: local hybrid model
    # ------------------------------------------------------

    return local_similar_movies(
        movie_id,
        count=20
    )


# ==========================================================
# 6. HYBRID RECOMMENDATION MODEL
# ==========================================================

def get_content_scores(movie_id):

    if not MODEL_READY:
        return {}

    indices = MODEL["movie_indices"]

    try:

        if isinstance(
            indices,
            (dict, pd.Series)
        ):

            idx = indices.get(
                movie_id,
                indices.get(str(movie_id))
            )

            index_items = indices.items()

        else:
            return {}

        if idx is None:
            return {}

        idx = int(idx)

        similarities = cosine_similarity(
            MODEL["tfidf"][idx],
            MODEL["tfidf"]
        ).flatten()

        scores = {}

        for mid, index in index_items:

            try:

                scores[int(mid)] = float(
                    similarities[index]
                )

            except (
                ValueError,
                TypeError,
                IndexError
            ):
                continue

        return scores

    except Exception:
        return {}


def get_collaborative_scores(movie_id):

    if not MODEL_READY:
        return {}

    movie_to_idx = MODEL["movie_to_idx"]
    factors = np.asarray(
        MODEL["movie_factors"]
    )

    idx = movie_to_idx.get(
        movie_id,
        movie_to_idx.get(str(movie_id))
    )

    if idx is None:
        return {}

    target = factors[idx]

    norms = (
        np.linalg.norm(
            factors,
            axis=1
        )
        *
        np.linalg.norm(target)
    )

    dots = factors @ target

    similarities = np.divide(
        dots,
        norms,
        out=np.zeros_like(
            dots,
            dtype=float
        ),
        where=norms != 0
    )

    scores = {}

    for mid, index in movie_to_idx.items():

        try:

            scores[int(mid)] = float(
                similarities[index]
            )

        except (
            ValueError,
            TypeError,
            IndexError
        ):
            continue

    return scores


@st.cache_data(
    show_spinner=False,
    max_entries=128
)
def hybrid_recommend(movie_id, count=20):

    if not MODEL_READY or MOVIES.empty:
        return pd.DataFrame()

    content = get_content_scores(
        int(movie_id)
    )

    collaborative = get_collaborative_scores(
        int(movie_id)
    )

    if not content and not collaborative:
        return pd.DataFrame()

    ids = set(content) | set(collaborative)

    rows = []

    for mid in ids:

        if mid == int(movie_id):
            continue

        c_score = content.get(
            mid,
            0.0
        )

        cf_score = collaborative.get(
            mid,
            0.0
        )

        # Rescale cosine similarity from [-1, 1] to [0, 1].
        cf_score = max(
            0.0,
            (cf_score + 1.0) / 2.0
        )

        combined = (
            0.6 * c_score
            +
            0.4 * cf_score
        )

        rows.append(
            (mid, combined)
        )

    if not rows:
        return pd.DataFrame()

    ranking = pd.DataFrame(
        rows,
        columns=[
            "id",
            "hybrid_score"
        ]
    )

    result = ranking.merge(
        MOVIES,
        on="id",
        how="inner"
    )

    return result.sort_values(
        "hybrid_score",
        ascending=False
    ).head(count)


# ==========================================================
# 7. SESSION COLLECTIONS
# ==========================================================

if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = {}

if "watch_later" not in st.session_state:
    st.session_state.watch_later = {}

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "filtered_movies" not in st.session_state:
    st.session_state.filtered_movies = []


def is_saved(collection, movie):

    return str(
        movie.get("id", "")
    ) in collection


def toggle_saved(collection, movie):

    mid = str(
        movie.get("id", "")
    )

    if not mid:
        return

    if mid in collection:
        del collection[mid]

    else:
        collection[mid] = movie


# ==========================================================
# 8. MOVIE CARD
# ==========================================================

def movie_card(
    movie,
    source,
    section="default"
):

    if source == "model":

        mid = int(movie["id"])

        title = str(
            movie.get(
                "title",
                "Untitled"
            )
        )

        poster = movie.get(
            "poster_path"
        )

        rating = movie.get(
            "vote_average",
            0
        )

        release = str(
            movie.get(
                "release_date",
                ""
            )
            or ""
        )

        overview = str(
            movie.get(
                "overview",
                ""
            )
            or ""
        )

        movie_data = {
            "id": mid,
            "title": title,
            "poster_path":
                poster
                if isinstance(
                    poster,
                    str
                )
                else None,
            "vote_average": rating,
            "release_date": release,
            "overview": overview
        }

    else:

        mid = movie.get("id")

        title = movie.get(
            "title"
        ) or "Untitled"

        poster = movie.get(
            "poster_path"
        )

        rating = movie.get(
            "vote_average",
            0
        )

        release = movie.get(
            "release_date",
            ""
        ) or ""

        movie_data = dict(movie)

    with st.container(border=True):

        if poster:

            st.image(
                image_url(poster),
                width="stretch"
            )

        else:

            st.markdown(
                f"""
                <div style="
                    height:220px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    background:{theme["panel"]};
                    border-radius:8px;
                    color:{theme["muted"]};
                ">
                    No poster available
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f'<div class="movie-title">{html.escape(str(title))}</div>',
            unsafe_allow_html=True
        )

        try:

            rating_text = (
                f"⭐ {float(rating):.1f}"
            )

        except (
            ValueError,
            TypeError
        ):

            rating_text = (
                "Rating unavailable"
            )

        year = (
            release[:4]
            if len(release) >= 4
            else "Unknown year"
        )

        st.caption(
            f"{rating_text} · {year}"
        )

        if st.button(
            "Details",
            key=f"details_{section}_{source}_{mid}",
            width="stretch"
        ):

            st.session_state.selected_movie = movie_data
            st.session_state.page = (
                "Movie details"
            )
            st.rerun()

        col1, col2 = st.columns(2)

        with col1:

            saved = is_saved(
                st.session_state.bookmarks,
                movie_data
            )

            if st.button(
                "★ Saved"
                if saved
                else "☆ Save",
                key=f"bookmark_{section}_{source}_{mid}",
                width="stretch"
            ):

                toggle_saved(
                    st.session_state.bookmarks,
                    movie_data
                )

                st.rerun()

        with col2:

            saved = is_saved(
                st.session_state.watch_later,
                movie_data
            )

            if st.button(
                "✓ Added"
                if saved
                else "＋ Later",
                key=f"later_{section}_{source}_{mid}",
                width="stretch"
            ):

                toggle_saved(
                    st.session_state.watch_later,
                    movie_data
                )

                st.rerun()


def render_movies(
    movies,
    source="tmdb",
    columns=5,
    section="default"
):

    if isinstance(
        movies,
        pd.DataFrame
    ):
        movies = movies.to_dict(
            "records"
        )

    if not movies:

        st.info(
            "No movies found."
        )

        return

    try:

        columns = max(
            1,
            min(
                int(columns),
                6
            )
        )

    except (
        TypeError,
        ValueError
    ):

        columns = 5

    for start in range(
        0,
        len(movies),
        columns
    ):

        batch = movies[
            start:start + columns
        ]

        cols = st.columns(
            len(batch),
            gap="small"
        )

        for col, movie in zip(
            cols,
            batch
        ):

            with col:

                movie_card(
                    movie,
                    source,
                    section
                )


@st.cache_data(
    ttl=900,
    show_spinner=False,
    max_entries=256
)
def discover_page(
    category,
    page
):

    result = discover_movies(
        category,
        page=page
    )

    return result


def render_category_feed(
    category,
    heading=None,
    initial_count=10
):

    key = (
        "feed_"
        +
        category
        .lower()
        .replace(" & ", "_")
        .replace(" ", "_")
        .replace(" ", "_")
    )

    if key not in st.session_state:

        st.session_state[key] = {
            "page": 1,
            "movies": [],
            "seen": set(),
            "done": False
        }

    state = st.session_state[key]

    if not state["movies"]:

        first = discover_page(
            category,
            1
        )

        state["movies"] = list(
            first
        )

        state["page"] = 1

        state["done"] = (
            len(first) == 0
        )

        state["seen"] = {
            m.get("id")
            for m in first
            if m.get("id") is not None
        }

    st.markdown(
        f'<div class="section-heading">{html.escape(heading or category)}</div>',
        unsafe_allow_html=True
    )

    shown = state["movies"]

    render_movies(
        shown,
        columns=5,
        section=key
    )

    next_col, info_col = st.columns(
        [1, 4]
    )

    with next_col:

        if (
            not state["done"]
            and
            st.button(
                "＋ Load more",
                key=f"load_{key}",
                width="stretch"
            )
        ):

            next_page = (
                state["page"] + 1
            )

            more = discover_page(
                category,
                next_page
            )

            added = [
                m for m in more
                if m.get("id")
                not in state["seen"]
            ]

            state["movies"].extend(
                added
            )

            state["seen"].update(
                m.get("id")
                for m in added
                if m.get("id") is not None
            )

            state["page"] = next_page

            state["done"] = (
                not more
                or
                next_page >= 500
            )

            st.rerun()

    with info_col:

        st.caption(
            f"Showing {len(shown)} movies · Scroll horizontally to explore"
        )


# ==========================================================
# 9. HOME PAGE
# ==========================================================

def render_home():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Your next movie awaits.
            </div>

            <div class="hero-subtitle">
                Discover films, explore what's trending,
                and find something worth watching.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    sections = [
        ("Latest movies", "Latest"),
        ("Trending today", "Trending"),
        ("Upcoming movies", "Upcoming"),
        ("Popular movies", "Popular"),
        ("Highest rated", "Highest Rated"),
        ("Kids & family", "Kids & Family"),
        ("Now playing", "Now Playing")
    ]

    for heading, category in sections:

        render_category_feed(
            category,
            heading=heading,
            initial_count=10
        )


# ==========================================================
# 10. SEARCH AND FILTERS
# ==========================================================

def render_search():

    st.title("Search movies")

    st.caption(
        "Search using a full or partial movie title."
    )

    query = st.text_input(
        "Movie title",
        placeholder="Try Batman, Interstellar, Spider...",
        key="movie_search"
    )

    if query.strip():

        suggestions = search_movies(
            query
        )

        if suggestions:

            options = {
                f"{m.get('title', 'Untitled')} "
                f"({(m.get('release_date') or '????')[:4]})":
                    m
                for m in suggestions[:10]
            }

            selected = st.selectbox(
                "Matching movies",
                ["Select a movie"]
                +
                list(options.keys()),
                key="search_suggestion"
            )

            if selected != "Select a movie":

                if st.button(
                    "Open selected movie",
                    type="primary"
                ):

                    st.session_state.selected_movie = (
                        options[selected]
                    )

                    st.session_state.page = (
                        "Movie details"
                    )

                    st.rerun()

        else:

            st.info(
                "No matching movies found."
            )

    st.divider()

    st.subheader(
        "Browse by filters"
    )

    genre = st.selectbox(
        "Genre",
        ["All genres"]
        +
        list(GENRES.keys()),
        key="filter_genre"
    )

    year_range = st.slider(
        "Release year",
        min_value=1900,
        max_value=TODAY.year + 5,
        value=(
            2000,
            TODAY.year + 5
        ),
        key="filter_year"
    )

    minimum_rating = st.slider(
        "Minimum rating",
        0.0,
        10.0,
        0.0,
        0.5,
        key="filter_rating"
    )

    sort_by = st.selectbox(
        "Sort by",
        [
            "Popularity",
            "Rating",
            "Release date"
        ],
        key="filter_sort"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Apply filters",
            type="primary",
            width="stretch"
        ):

            params = {
                "include_adult": "false",
                "sort_by": {
                    "Popularity":
                        "popularity.desc",
                    "Rating":
                        "vote_average.desc",
                    "Release date":
                        "primary_release_date.desc"
                }[sort_by],
                "vote_count.gte": 5,
                "primary_release_date.gte":
                    f"{year_range[0]}-01-01",
                "primary_release_date.lte":
                    f"{year_range[1]}-12-31",
                "vote_average.gte":
                    minimum_rating,
                "page": 1
            }

            if genre != "All genres":

                params["with_genres"] = (
                    GENRES[genre]
                )

            # --------------------------------------------------
            # Primary source: TMDB
            # --------------------------------------------------

            data = tmdb_get(
                "discover/movie",
                params
            )

            tmdb_results = (
                data.get("results", [])
                if isinstance(
                    data,
                    dict
                )
                else []
            )

            if tmdb_results:

                st.session_state.filtered_movies = (
                    tmdb_results
                )

            else:

                # ----------------------------------------------
                # Fallback: local catalog
                # ----------------------------------------------

                st.session_state.filtered_movies = (
                    local_filtered_movies(
                        genre,
                        year_range,
                        minimum_rating,
                        sort_by
                    )
                )

    with col2:

        if st.button(
            "Reset filters",
            width="stretch"
        ):

            st.session_state.filtered_movies = []

            for key in (
                "filter_genre",
                "filter_year",
                "filter_rating",
                "filter_sort"
            ):

                if key in st.session_state:

                    del st.session_state[key]

            st.rerun()

    if st.session_state.filtered_movies:

        st.subheader(
            "Filtered movies"
        )

        render_movies(
            st.session_state.filtered_movies,
            columns=5,
            section="filtered_search"
        )


# ==========================================================
# 11. MOVIE DETAILS AND RECOMMENDATIONS
# ==========================================================

def render_details():

    movie = (
        st.session_state.selected_movie
    )

    if not movie:

        st.info(
            "Select a movie first."
        )

        return

    if st.button(
        "← Back to Home"
    ):

        st.session_state.page = (
            "Home"
        )

        st.rerun()

    # ------------------------------------------------------
    # Try TMDB details first
    # ------------------------------------------------------

    details = tmdb_details(
        movie.get("id")
    )

    if details:

        movie = {
            **movie,
            **details
        }

    # ------------------------------------------------------
    # Backdrop
    # ------------------------------------------------------

    backdrop = movie.get(
        "backdrop_path"
    )

    if backdrop:

        st.image(
            image_url(
                backdrop,
                "w1280"
            ),
            width="stretch"
        )

    st.title(
        movie.get(
            "title",
            "Untitled"
        )
    )

    rating = movie.get(
        "vote_average",
        0
    )

    release = movie.get(
        "release_date",
        "Unknown"
    )

    runtime = movie.get(
        "runtime"
    )

    line = (
        f"⭐ {float(rating or 0):.1f} · "
        f"{release}"
    )

    if runtime:

        line += (
            f" · {runtime} min"
        )

    st.caption(line)

    genres = movie.get(
        "genres",
        []
    )

    if isinstance(
        genres,
        list
    ):

        names = [
            g.get("name", "")
            if isinstance(g, dict)
            else str(g)
            for g in genres
        ]

        if names:

            st.write(
                " · ".join(names)
            )

    if movie.get(
        "overview"
    ):

        st.write(
            movie["overview"]
        )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "★ Remove bookmark"
            if is_saved(
                st.session_state.bookmarks,
                movie
            )
            else "☆ Add bookmark",
            width="stretch"
        ):

            toggle_saved(
                st.session_state.bookmarks,
                movie
            )

            st.rerun()

    with c2:

        if st.button(
            "✓ Remove from Watch Later"
            if is_saved(
                st.session_state.watch_later,
                movie
            )
            else "＋ Add to Watch Later",
            width="stretch"
        ):

            toggle_saved(
                st.session_state.watch_later,
                movie
            )

            st.rerun()

    st.divider()

    st.subheader(
        "Similar movies"
    )

    recommendations = pd.DataFrame()

    if MODEL_READY:

        recommendations = hybrid_recommend(
            movie.get("id"),
            20
        )

    if not recommendations.empty:

        render_movies(
            recommendations,
            source="model",
            columns=5,
            section="similar_model"
        )

    else:

        st.caption(
            "This movie is not available in the trained model. "
            "Showing TMDB recommendations instead."
        )

        similar_movies = tmdb_similar(
            movie.get("id")
        )

        render_movies(
            similar_movies[:20],
            columns=5,
            section="similar_tmdb"
        )


# ==========================================================
# 12. BOOKMARKS AND WATCH LATER
# ==========================================================

def render_collection(
    title,
    collection
):

    st.title(title)

    st.caption(
        "Saved for this browser session. "
        "These lists are not synchronized across devices."
    )

    movies = list(
        collection.values()
    )

    if not movies:

        st.info(
            f"Your {title.lower()} list is empty."
        )

        return

    if st.button(
        f"Clear {title}"
    ):

        collection.clear()
        st.rerun()

    render_movies(
        movies,
        columns=5,
        section=title.lower().replace(
            " ",
            "_"
        )
    )


# ==========================================================
# 12. MODEL EVALUATION DASHBOARD
# ==========================================================

def render_evaluation():

    st.title(
        "Model Evaluation"
    )

    st.caption(
        "Diagnostics for the trained recommendation models."
    )

    if not MODEL_READY or MOVIES.empty:

        st.error(
            "The trained model or movie catalog is unavailable."
        )

        return

    content_ids = set()
    collaborative_ids = set()

    try:

        content_ids = {
            int(x)
            for x in
            MODEL["movie_indices"].keys()
        }

    except Exception:
        pass

    try:

        collaborative_ids = {
            int(x)
            for x in
            MODEL["movie_to_idx"].keys()
        }

    except Exception:
        pass

    catalog_ids = set(
        MOVIES["id"]
        .astype(int)
        .tolist()
    )

    hybrid_ids = (
        content_ids
        |
        collaborative_ids
    )

    covered_ids = (
        catalog_ids
        &
        hybrid_ids
    )

    coverage = (
        len(covered_ids)
        /
        len(catalog_ids)
        *
        100
        if catalog_ids
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Movies in catalog",
        f"{len(catalog_ids):,}"
    )

    c2.metric(
        "Content model movies",
        f"{len(content_ids):,}"
    )

    c3.metric(
        "Collaborative model movies",
        f"{len(collaborative_ids):,}"
    )

    c4.metric(
        "Catalog coverage",
        f"{coverage:.1f}%"
    )

    st.divider()

    st.subheader(
        "Model availability"
    )

    status = pd.DataFrame([
        {
            "Component":
                "TF-IDF content model",
            "Status":
                "Ready"
                if MODEL.get("tfidf")
                is not None
                else "Missing"
        },
        {
            "Component":
                "Collaborative factors",
            "Status":
                "Ready"
                if MODEL.get("movie_factors")
                is not None
                else "Missing"
        },
        {
            "Component":
                "Hybrid ranker",
            "Status":
                "Ready"
                if content_ids
                and collaborative_ids
                else "Partially available"
        },
        {
            "Component":
                "Movie metadata",
            "Status":
                "Ready"
                if not MOVIES.empty
                else "Missing"
        }
    ])

    st.dataframe(
        status,
        hide_index=True,
        width="stretch"
    )

    st.subheader(
        "Recommendation comparison"
    )

    st.write(
        "Choose a movie that exists in the trained catalog "
        "to compare the three ranking methods."
    )

    options = (
        MOVIES[
            ["id", "title"]
        ]
        .drop_duplicates("id")
        .sort_values("title")
    )

    title_to_id = {
        f"{row.title} (ID: {int(row.id)})":
            int(row.id)
        for row in options.itertuples()
    }

    selected_label = st.selectbox(
        "Movie",
        list(title_to_id.keys()),
        key="eval_seed_movie"
    )

    top_k = st.slider(
        "Recommendations to compare",
        5,
        30,
        10,
        key="eval_top_k"
    )

    if selected_label:

        movie_id = title_to_id[
            selected_label
        ]

        content_scores = (
            get_content_scores(
                movie_id
            )
        )

        cf_scores = (
            get_collaborative_scores(
                movie_id
            )
        )

        def ranked_frame(
            scores,
            score_name
        ):

            rows = [
                (
                    int(mid),
                    float(score)
                )
                for mid, score
                in scores.items()
                if int(mid) != movie_id
                and np.isfinite(score)
            ]

            frame = pd.DataFrame(
                rows,
                columns=[
                    "id",
                    score_name
                ]
            )

            if frame.empty:
                return frame

            return (
                frame
                .sort_values(
                    score_name,
                    ascending=False
                )
                .head(top_k)
                .merge(
                    MOVIES[
                        [
                            "id",
                            "title",
                            "vote_average"
                        ]
                    ],
                    on="id",
                    how="inner"
                )
            )

        content_frame = ranked_frame(
            content_scores,
            "content_score"
        )

        cf_frame = ranked_frame(
            cf_scores,
            "collaborative_cosine"
        )

        hybrid_frame = hybrid_recommend(
            movie_id,
            top_k
        )

        def normalize_cf(frame):

            if frame.empty:
                return frame

            frame = frame.copy()

            frame[
                "collaborative_score"
            ] = (
                (
                    frame[
                        "collaborative_cosine"
                    ]
                    + 1
                )
                / 2
            ).clip(0, 1)

            return frame

        cf_frame = normalize_cf(
            cf_frame
        )

        left, middle, right = st.columns(3)

        with left:

            st.markdown(
                "**Content-based**"
            )

            if content_frame.empty:

                st.info(
                    "No content recommendations available."
                )

            else:

                st.dataframe(
                    content_frame[
                        [
                            "title",
                            "content_score",
                            "vote_average"
                        ]
                    ],
                    hide_index=True,
                    width="stretch"
                )

        with middle:

            st.markdown(
                "**Collaborative**"
            )

            if cf_frame.empty:

                st.info(
                    "No collaborative recommendations available."
                )

            else:

                st.dataframe(
                    cf_frame[
                        [
                            "title",
                            "collaborative_score",
                            "vote_average"
                        ]
                    ],
                    hide_index=True,
                    width="stretch"
                )

        with right:

            st.markdown(
                "**Hybrid (60% / 40%)**"
            )

            if hybrid_frame.empty:

                st.info(
                    "No hybrid recommendations available."
                )

            else:

                st.dataframe(
                    hybrid_frame[
                        [
                            "title",
                            "hybrid_score",
                            "vote_average"
                        ]
                    ],
                    hide_index=True,
                    width="stretch"
                )

        sets = []

        for frame in (
            content_frame,
            cf_frame,
            hybrid_frame
        ):

            sets.append(
                set(
                    frame["id"]
                    .astype(int)
                )
                if not frame.empty
                else set()
            )

        union = (
            set.union(*sets)
            if sets
            else set()
        )

        overlap = (
            len(sets[0] & sets[1])
            if len(sets) >= 2
            else 0
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Content / collaborative overlap",
            f"{overlap} movies"
        )

        m2.metric(
            "Unique movies across lists",
            f"{len(union)}"
        )

        m3.metric(
            "Hybrid list size",
            f"{len(sets[2]) if len(sets) > 2 else 0}"
        )

        st.subheader(
            "Hybrid score distribution"
        )

        if (
            not hybrid_frame.empty
            and
            "hybrid_score"
            in hybrid_frame
        ):

            st.bar_chart(
                hybrid_frame
                .set_index("title")
                [["hybrid_score"]],
                height=300
            )

        else:

            st.info(
                "No hybrid scores are available for this movie."
            )

    st.divider()

    st.subheader(
        "How to interpret these results"
    )

    st.markdown(
        """
- **Catalog coverage** is the percentage of catalog movies represented in at least one trained model.
- **Overlap** counts movies appearing in both content-based and collaborative top-K lists.
- **Scores** are ranking signals, not probabilities or measures of recommendation accuracy.
- This dashboard is a model diagnostic. Precision@K, Recall@K and NDCG@K require a held-out set of user–movie interactions and are intentionally not fabricated here.
"""
    )


# ==========================================================
# 13. SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.markdown(
        "## 🎬 CineMatch"
    )

    st.caption(
        "A universe of stories"
    )

    st.selectbox(
        "Theme",
        list(THEMES.keys()),
        key="theme"
    )

    st.divider()

    pages = [
        "Home",
        "Search",
        "Trending",
        "Popular",
        "Latest",
        "Upcoming",
        "Highest Rated",
        "Kids & Family",
        "Now Playing",
        "Bookmarks",
        "Watch Later",
        "Model Evaluation"
    ]

    for page in pages:

        label = (
            "⌂  Home"
            if page == "Home"
            else page
        )

        if st.button(
            label,
            key=f"nav_{page}",
            width="stretch",
            type=(
                "primary"
                if st.session_state.page == page
                else "secondary"
            )
        ):

            st.session_state.page = page
            st.rerun()

    st.divider()

    st.caption(
        "Movie data provided by TMDB."
    )


# ==========================================================
# 14. STATUS MESSAGES
# ==========================================================

if not TMDB_KEY:

    st.warning(
        "TMDB API key not found. Add TMDB_API_KEY to "
        ".streamlit/secrets.toml to enable live movie data."
    )

if not MODEL_READY:

    st.warning(
        "The trained recommendation model could not be loaded. "
        "TMDB browsing can still work."
    )

    with st.expander(
        "Model loading details"
    ):

        st.code(
            MODEL_ERROR
        )


# ==========================================================
# 15. PAGE ROUTING
# ==========================================================

page = st.session_state.page


if page == "Home":

    render_home()


elif page == "Search":

    render_search()


elif page in [
    "Trending",
    "Popular",
    "Latest",
    "Upcoming",
    "Highest Rated",
    "Kids & Family",
    "Now Playing"
]:

    st.title(page)

    render_category_feed(
        page,
        heading=page
    )


elif page == "Bookmarks":

    render_collection(
        "Bookmarks",
        st.session_state.bookmarks
    )


elif page == "Watch Later":

    render_collection(
        "Watch Later",
        st.session_state.watch_later
    )


elif page == "Movie details":

    render_details()


elif page == "Model Evaluation":

    render_evaluation()