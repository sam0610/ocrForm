import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import datetime
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

st.set_page_config(page_title="散工申報表即時辨識 App", layout="wide")

# ---- 強制相機預覽/相片預覽用全螢幕闊度顯示，方便手機用戶睇清楚 ----
st.markdown("""
<style>
    /* 相機直播預覽 (未影相之前) */
    video {
        width: 100% !important;
        height: auto !important;
        max-height: 80vh !important;
    }
    /* 已影低嘅相片預覽 */
    div[data-testid="stCameraInput"] img,
    div[data-testid="stImage"] img {
        width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
    }
    /* 主內容區盡量用盡闊度，手機睇少啲留白 */
    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    /* camera_input 個外層容器都放寬 */
    div[data-testid="stCameraInput"] {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 0. 表格版面設定(要跟返你實際印刷嘅範本調)
# ==========================================
# 4個角校正標記嘅ID(固定用0-3,唔好同大廈ID撞)
CORNER_IDS = {0: "TL", 1: "TR", 2: "BL", 3: "BR"}

# 大廈識別標記(用另一組ID,例如10開始,同角標分開)
BUILDING_MARKER_MAP = {
    10: "THE ONE 大廈",
    11: "iSQUARE 國際廣場",
}

# 校正之後,將表格拉直去一個固定大小嘅畫布(像素),方便之後用固定比例切格仔
WARP_WIDTH = 3300
WARP_HEIGHT = 1800

# 表格喺畫布入面嘅相對範圍(留返少少邊界畀角標本身占用嘅位置)
TABLE_LEFT_RATIO = 0.03
TABLE_RIGHT_RATIO = 0.97
TABLE_TOP_RATIO = 0.07
TABLE_BOTTOM_RATIO = 0.97

N_HEADER_ROWS = 3      # 日期row + 星期row + 欄位標籤row
N_DATA_ROWS = 21        # C0001 - C0021
N_DAYS = 31

# 左邊資料欄相對闊度比例(散工編號、判頭、散工姓名、工作時間)
LEFT_COL_RATIOS = [0.06, 0.04, 0.06, 0.06]  # 總和 + 日欄部分 = 1.0
# 即日欄部分闊度比例 = 1 - sum(LEFT_COL_RATIOS)

# ==========================================
# 1. Google Drive 自動上載邏輯(不變)
# ==========================================
def upload_to_drive(file_bytes, file_name, mime_type):
    try:
        gcp_secrets = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
        folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]

        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        creds = service_account.Credentials.from_service_account_info(gcp_secrets, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)

        file_metadata = {'name': file_name, 'parents': [folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime_type, resumable=True)

        file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
        return file.get('webViewLink')
    except Exception as e:
        st.error(f"Google Drive 上載失敗: {str(e)}")
        return None


# ==========================================
# 2. ArUco 偵測(針對細標記調校參數)
# ==========================================
def get_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # 針對細標記(1cm級)調校:縮細adaptive threshold窗口步進,搜索更細嘅候選
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.01   # 容許細標記(預設0.03太高,細標記會被過濾)
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5

    return cv2.aruco.ArucoDetector(aruco_dict, parameters)


def order_corner_points(ids, corners):
    """由偵測到嘅角標,取得TL/TR/BL/BR四點嘅中心座標"""
    pts = {}
    for marker_id, marker_corners in zip(ids.flatten(), corners):
        if marker_id in CORNER_IDS:
            c = marker_corners[0]
            center = c.mean(axis=0)
            pts[CORNER_IDS[marker_id]] = center
    if len(pts) < 4:
        return None
    return pts


def warp_table(img, pts):
    src = np.array([pts["TL"], pts["TR"], pts["BR"], pts["BL"]], dtype=np.float32)
    dst = np.array([
        [0, 0],
        [WARP_WIDTH, 0],
        [WARP_WIDTH, WARP_HEIGHT],
        [0, WARP_HEIGHT],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (WARP_WIDTH, WARP_HEIGHT))
    return warped


# ==========================================
# 3. 格仔切割 + 符號分類(真正嘅辨識邏輯)
# ==========================================
def build_grid_lines():
    table_left = WARP_WIDTH * TABLE_LEFT_RATIO
    table_right = WARP_WIDTH * TABLE_RIGHT_RATIO
    table_top = WARP_HEIGHT * TABLE_TOP_RATIO
    table_bottom = WARP_HEIGHT * TABLE_BOTTOM_RATIO

    total_rows = N_HEADER_ROWS + N_DATA_ROWS
    row_height = (table_bottom - table_top) / total_rows
    row_lines = [table_top + i * row_height for i in range(total_rows + 1)]
    data_row_lines = row_lines[N_HEADER_ROWS:]  # 21行資料嘅邊界(22條線)

    left_total_ratio = sum(LEFT_COL_RATIOS)
    day_area_left = table_left + (table_right - table_left) * left_total_ratio
    day_col_width = (table_right - day_area_left) / N_DAYS
    col_lines = [day_area_left + i * day_col_width for i in range(N_DAYS + 1)]

    return data_row_lines, col_lines


def classify_cell(gray_cell, margin_ratio=0.24, min_area_ratio=0.045, min_std=13.0, min_contrast=35.0, aspect_thresh=1.6):
    """
    判斷呢一格係：✓(翻工) / X(冇開工) / 空白(漏填)
    用Otsu做「每格獨立」二值化，唔受成張相入面唔均勻嘅光暗/陰影影響
    """
    h, w = gray_cell.shape
    my, mx = int(h * margin_ratio), int(w * margin_ratio)
    cell = gray_cell[my:h - my, mx:w - mx]
    if cell.size == 0:
        return "空白"

    cell_blur = cv2.GaussianBlur(cell, (3, 3), 0)

    # 第一步關卡：如果成格根本冇乜反差(淨係白紙+相機雜訊)，
    # 直接當空白，唔使跑Otsu(Otsu對接近單一色調嘅圖會強行分裂，容易誤判)
    if cell_blur.std() < min_std:
        return "空白"

    thresh_val, binary = cv2.threshold(cell_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 第二步關卡：檢查Otsu分開嘅「前景」同「背景」反差是否足夠大，
    # 反差太細代表可能只係雜訊/紙紋，唔係真正墨水
    fg_mask = binary > 0
    bg_mask = ~fg_mask
    if fg_mask.sum() == 0 or bg_mask.sum() == 0:
        return "空白"
    fg_mean = cell_blur[fg_mask].mean()
    bg_mean = cell_blur[bg_mask].mean()
    if (bg_mean - fg_mean) < min_contrast:
        return "空白"

    # 開運算：用大少少嘅kernel，將格線滲入嚟嘅幼細殘留線徹底剷走，
    # 淨係留低夠粗嘅真實筆劃(✓/X通常筆劃粗過格線本身)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "空白"

    # 用所有輪廓嘅總面積(唔淨係最大嗰嚿)，因為✓同X都可能斷開做幾截筆劃
    total_area = sum(cv2.contourArea(c) for c in contours)
    cell_area = cell.shape[0] * cell.shape[1]
    if total_area < min_area_ratio * cell_area:
        return "空白"

    # 分辨✓ 同 X：
    # X 通常兩筆交叉，輪廓嘅「凸包缺陷(convexity defects)」數量較多，
    # 整體形狀貼近方形(minAreaRect長寬比接近1)；
    # ✓ 剔號通常兩筆唔對稱(一短一長)，輪廓整體斜向一邊，
    # 外接矩形長寬比通常較大(更加瘦長)。
    all_pts = np.vstack(contours)
    rect = cv2.minAreaRect(all_pts)
    (rw, rh) = rect[1]
    if rw == 0 or rh == 0:
        return "空白"
    aspect = max(rw, rh) / min(rw, rh)

    hull = cv2.convexHull(all_pts)
    hull_area = cv2.contourArea(hull)
    solidity = total_area / hull_area if hull_area > 0 else 0

    # X：長寬比接近方形(<aspect_thresh)，因為兩筆交叉撐開成正方形範圍
    # ✓：長寬比較大(>=aspect_thresh)，因為剔號係斜向一筆長劃
    if aspect < aspect_thresh:
        return "X"
    else:
        return "✓"


def process_image(img, min_std=13.0, min_contrast=35.0, min_area_ratio=0.045, margin_ratio=0.24, aspect_thresh=1.6):
    detector = get_aruco_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, "未搵到任何ArUco標記，請重新拍攝，確保4個角標記同大廈標記都影入鏡頭！", img, None

    ids_flat = ids.flatten()

    # 角標(用嚟做透視校正)
    pts = order_corner_points(ids, corners)
    if pts is None:
        found = [i for i in ids_flat if i in CORNER_IDS]
        return None, f"只搵到{len(found)}/4個角標記，請重新拍攝，確保4個角落都清晰入鏡！", img, None

    # 大廈標記(獨立於角標，用嚟識別邊間大廈)
    building_name = "未知大廈（未偵測到大廈標記）"
    for m_id in ids_flat:
        if m_id in BUILDING_MARKER_MAP:
            building_name = BUILDING_MARKER_MAP[m_id]
            break

    annotated = img.copy()
    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

    # 透視校正：將表格拉直去固定畫布
    warped = warp_table(img, pts)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.medianBlur(warped_gray, 3)
    # 注意：唔再喺呢度做成張相嘅全域二值化，改為留返喺classify_cell入面
    # 用Otsu做「每格獨立」判斷，先唔怕相入面局部陰影/光暗不均

    row_lines, col_lines = build_grid_lines()

    # ---- 除錯用：喺拉直咗嘅相上面畫低格仔切割線，等你可以肉眼核對啱唔啱 ----
    grid_debug = warped.copy()
    for y in row_lines:
        cv2.line(grid_debug, (int(col_lines[0]), int(y)), (int(col_lines[-1]), int(y)), (0, 255, 0), 1)
    for x in col_lines:
        cv2.line(grid_debug, (int(x), int(row_lines[0])), (int(x), int(row_lines[-1])), (0, 0, 255), 1)

    dates = [f"{i}日" for i in range(1, N_DAYS + 1)]
    workers = [f"C{i:04d}" for i in range(1, N_DATA_ROWS + 1)]
    matrix = []
    for r in range(N_DATA_ROWS):
        y0, y1 = int(row_lines[r]), int(row_lines[r + 1])
        row_vals = []
        for c in range(N_DAYS):
            x0, x1 = int(col_lines[c]), int(col_lines[c + 1])
            cell_gray = warped_gray[y0:y1, x0:x1]
            row_vals.append(classify_cell(
                cell_gray,
                margin_ratio=margin_ratio,
                min_area_ratio=min_area_ratio,
                min_std=min_std,
                min_contrast=min_contrast,
                aspect_thresh=aspect_thresh,
            ))
        matrix.append(row_vals)

    df = pd.DataFrame(matrix, columns=dates, index=workers)
    return df, building_name, annotated, grid_debug


# ==========================================
# 4. Streamlit UI 畫面
# ==========================================
st.title("📋 散工申報表 AI 雲端辨識系統")
st.write("手機拍照 ➔ 雲端辨識 ➔ 自動備份相片與 Excel 至 Google Drive")

with st.expander("⚠️ 拍攝要求"):
    st.write("""
    - 表格4個角落嘅ArUco校正標記（ID 0-3）必須全部入鏡、清晰、唔矇
    - 大廈識別標記（另一組ID）都要影到
    - 盡量正面影，避免過大角度傾斜
    - 光線要均勻，避免陰影遮住標記
    """)

with st.sidebar:
    st.header("🛠️ 除錯設定（可即時調整，唔使改code）")
    st.caption("如果誤判太多／太少，用呢幾個滑桿試吓，見到準咗就記低個數值，再叫Claude幫手寫返落code永久生效。")
    debug_min_std = st.slider("最低反差(std) — 太細代表當空白", 0.0, 40.0, 13.0, 0.5)
    debug_min_contrast = st.slider("前景/背景最低對比", 0.0, 80.0, 35.0, 1.0)
    debug_min_area_ratio = st.slider("最低墨水面積比例", 0.0, 0.15, 0.045, 0.005)
    debug_margin_ratio = st.slider("格仔邊界收縮比例", 0.0, 0.4, 0.24, 0.01)
    debug_aspect_thresh = st.slider("✓/X 分界長寬比 (細於呢個=X，大於=✓)", 1.0, 3.0, 1.6, 0.05)

camera_image = st.camera_input("請對準散工申報表拍攝 (需包含 4 角校正標記 + 大廈標記)")
st.markdown("**— 或者 —**")
uploaded_image = st.file_uploader("上載已有嘅圖片（例如相簿入面已影低嘅相）", type=["jpg", "jpeg", "png"])

img_file = camera_image or uploaded_image

if img_file:
    bytes_data = img_file.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("🔄 正在進行雲端 OpenCV 解析中..."):
        df, building, annotated_img, grid_debug_img = process_image(
            img,
            min_std=debug_min_std,
            min_contrast=debug_min_contrast,
            min_area_ratio=debug_min_area_ratio,
            margin_ratio=debug_margin_ratio,
            aspect_thresh=debug_aspect_thresh,
        )

    if df is None:
        st.error(building)
        st.image(img, channels="BGR", caption="原始相片")
    else:
        st.success(f"🏢 成功辨識大廈：**{building}**")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 出勤數據預覽")
            st.dataframe(df, height=300)
            st.caption("符號說明：✓ = 有翻工　X = 冇開工　空白 = 可能漏填，需要跟進")

        with col2:
            st.subheader("🔍 角標與校正定位")
            st.image(annotated_img, channels="BGR", use_container_width=True)

        with st.expander("🩺 除錯用：格仔切割線疊圖（睇吓紅線/綠線同真實表格格線啱唔啱）"):
            st.image(grid_debug_img, channels="BGR", use_container_width=True)
            st.caption("綠線 = 行分界　紅線 = 日曆格分界。如果啲線同真實印刷格線對唔上，代表要返去校正工具重新調比例。")

        # 檢查漏填（真正空白格）
        blank_count = (df == "空白").sum().sum()
        if blank_count > 0:
            st.warning(f"⚠️ 偵測到 {blank_count} 格仍然空白，可能係漏填，建議跟進確認。")

        if st.button("☁️ 確認並一鍵同步至 Google Drive", type="primary"):
            with st.spinner("同步上載中..."):
                today_str = datetime.date.today().strftime("%Y%m%d_%H%M%S")

                towrite = io.BytesIO()
                df.to_excel(towrite, index=True)
                excel_url = upload_to_drive(
                    towrite.getvalue(),
                    f"{building}_{today_str}_出勤表.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                img_ext = "jpg"
                img_mime = "image/jpeg"
                if uploaded_image and img_file is uploaded_image:
                    orig_name = uploaded_image.name.lower()
                    if orig_name.endswith(".png"):
                        img_ext, img_mime = "png", "image/png"

                img_url = upload_to_drive(
                    bytes_data,
                    f"{building}_{today_str}_原始相片.{img_ext}",
                    img_mime
                )

                if excel_url and img_url:
                    st.balloons()
                    st.success("✅ 成功！相片與 Excel 檔案已自動存入公司 Google Drive 資料夾！")
