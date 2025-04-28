# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder
import traceback # エラー詳細表示用 (オプション)

# --- アプリケーション設定 ---
st.set_page_config(
    page_title="Gemma 2 Chatbot",
    page_icon="🤖",  # <<< 変更点: 絵文字アイコンを追加
    layout="wide",
    initial_sidebar_state="expanded" # <<< 変更点: サイドバーを最初から開く
)

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
try:
    metrics.initialize_nltk()
except Exception as e:
    st.warning(f"NLTKデータの初期化中にエラーが発生しました: {e}", icon="⚠️")

# データベースの初期化
try:
    database.init_db()
except Exception as e:
    st.error(f"データベースの初期化に失敗しました: {e}", icon="🚨")
    st.stop() # データベース初期化失敗時はアプリを停止

# データベースが空ならサンプルデータを投入
try:
    data.ensure_initial_data()
except Exception as e:
    st.warning(f"サンプルデータの投入中にエラーが発生しました: {e}", icon="⚠️")


# --- LLMモデルのロード（キャッシュを利用） ---
# モデルをキャッシュして再利用
@st.cache_resource(show_spinner="🤖 Gemmaモデルをロード中...") # <<< 変更点: スピナー表示を追加
def load_model_cached():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # HuggingFace Hubへのログイン状態を確認 (オプション)
        # token = HfFolder.get_token()
        # if not token:
        #     st.warning("Hugging Face Hubにログインしていません。プライベートモデルの読み込みに失敗する可能性があります。", icon="🔒")

        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
            # token=token # 必要に応じてトークンを渡す
        )
        # <<< 変更点: 成功メッセージを st.success で表示し、デバイス情報も加える
        st.success(f"モデル '{MODEL_NAME}' の読み込み完了 (デバイス: {device})", icon="✅")
        return pipe
    except ImportError as e:
         st.error(f"必要なライブラリが見つかりません: {e}", icon="🚨")
         st.info("`pip install torch transformers accelerate bitsandbytes` などを実行してください。")
         return None
    except Exception as e:
        # <<< 変更点: エラーメッセージを st.error で表示し、具体的な原因を示唆
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました。", icon="🚨")
        st.error(f"エラー詳細: {e}")
        st.warning("GPUメモリ不足、モデル名の誤り、ネットワーク接続、HuggingFace Hubの認証などを確認してください。", icon="💡")
        # traceback.print_exc() # デバッグ用にコンソールに詳細なエラーを出力する場合
        return None

# モデルロードを実行
pipe = load_model_cached()

# --- Streamlit アプリケーション ---

# --- サイドバー ---
# <<< 変更点: `with st.sidebar:` を使うとインデントで見やすくなる
with st.sidebar:
    st.title("📄 ナビゲーション")
    st.divider() # <<< 変更点: 区切り線を追加

    # ページ選択方法の改善 (アイコン追加)
    # セッション状態を使用して選択ページを保持
    if 'page' not in st.session_state:
        st.session_state.page = "チャット" # デフォルトページ

    # <<< 変更点: ページ選択肢にアイコンを追加
    pages = {
        "チャット": "💬",
        "履歴閲覧": "📜",
        "サンプルデータ管理": "📊"
    }
    page_options = list(pages.keys())

    # 現在のページのインデックスを取得
    try:
        current_page_index = page_options.index(st.session_state.page)
    except ValueError:
        current_page_index = 0 # 見つからない場合はデフォルト(チャット)
        st.session_state.page = page_options[0]

    # ラジオボタンでページ選択
    selected_page = st.radio(
        "ページを選択", # ラベルを少し変更
        options=page_options,
        format_func=lambda page: f"{pages[page]} {page}", # アイコンとページ名を表示
        index=current_page_index,
        key="page_selector",
        label_visibility="collapsed", # <<< 変更点: ラベル "ページを選択" を非表示にし、アイコン付き選択肢を強調
    )

    # 選択されたページをセッション状態に反映 (st.radio のキー更新に依存)
    # on_change や st.rerun() は必須ではないことが多いが、複雑な状態管理が必要な場合に使う
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun() # ページ遷移を即時反映

    st.divider() # <<< 変更点: 区切り線を追加

    # <<< 変更点: モデルの状態表示を追加
    if pipe:
        st.success("✅ モデル準備完了", icon="👍")
    else:
        st.error("❌ モデル読み込み失敗", icon="😥")

    st.divider() # <<< 変更点: 区切り線を追加

    # <<< 変更点: 開発者情報にアイコンを追加
    st.info("👨‍💻 開発者: [Suwa Haruya]", icon="ℹ️")
    # st.image("your_logo.png", use_column_width=True) # ロゴ画像などを表示する場合


# --- メインコンテンツ ---
# <<< 変更点: タイトルとキャプションで見た目を調整
st.title("🤖 Gemma 2 Chatbot with Feedback")
st.caption("✨ Gemmaモデルを使用したチャットボットです。回答へのフィードバックで精度が向上します！")
st.divider() # <<< 変更点: タイトル下の区切り線

# <<< 変更点: メインコンテンツ領域をコンテナで囲む (オプションだが構造化しやすい)
main_container = st.container()

with main_container:
    if st.session_state.page == "チャット":
        # <<< 変更点: 各ページのタイトルを st.header や st.subheader で明確化
        st.header("💬 チャット", divider="rainbow") # dividerで見出しを装飾
        if pipe:
            # ui.display_chat_page は ui.py で定義されていると仮定
            ui.display_chat_page(pipe)
        else:
            st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。", icon="🚫")
            st.info("サイドバーのエラーメッセージ、またはコンソールのログを確認してください。", icon="👀")

    elif st.session_state.page == "履歴閲覧":
        st.header("📜 履歴閲覧", divider="rainbow")
        # ui.display_history_page は ui.py で定義されていると仮定
        ui.display_history_page()

    elif st.session_state.page == "サンプルデータ管理":
        st.header("📊 サンプルデータ管理", divider="rainbow")
        # ui.display_data_page は ui.py で定義されていると仮定
        ui.display_data_page()

# --- フッターなど（任意） ---
st.divider()
st.caption("© 2025 Gemma Chatbot")