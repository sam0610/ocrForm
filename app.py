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


TABLE_LEFT_RATIO = 0.03
TABLE_RIGHT_RATIO = 0.97
TABLE_TOP_RATIO = 0.07
TABLE_BOTTOM_RATIO = 0.97

N_HEADER_ROWS = 3
N_DATA_ROWS = 21
N_DAYS = 31

LEFT_COL_RATIOS = [0.06, 0.04, 0.06, 0.06]

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


def classify_cell(binary_img, margin_ratio=0.20, min_area=250, slash_aspect_thresh=2.3, circularity_thresh=0.55):
    h, w = binary_img.shape
    my, mx = int(h * margin_ratio), int(w * margin_ratio)
    cell = binary_img[my:h - my, mx:w - mx]
    if cell.size == 0:
        return "空白"

    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "空白"

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return "空白"

    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        return "空白"
    circularity = 4 * np.pi * area / (perimeter ** 2)

    # 用最小外接旋轉矩形嘅長寬比,分辨「單一斜線╱」同「X / 圓」
    rect = cv2.minAreaRect(c)
    (rw, rh) = rect[1]
    if rw == 0 or rh == 0:
        return "空白"
    aspect = max(rw, rh) / min(rw, rh)

    if aspect > slash_aspect_thresh:
        return "╱"  # 確認冇開工
    if circularity > circularity_thresh:
        return "圓形記號"
    return "X"


def process_image(img):
    detector = get_aruco_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, "未搵到任何ArUco標記，請重新拍攝，確保4個角標記同大廈標記都影入鏡頭！", img

    ids_flat = ids.flatten()

    # 角標(用嚟做透視校正)
    pts = order_corner_points(ids, corners)
    if pts is None:
        found = [i for i in ids_flat if i in CORNER_IDS]
        return None, f"只搵到{len(found)}/4個角標記，請重新拍攝，確保4個角落都清晰入鏡！", img

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
    denoised = cv2.medianBlur(warped_gray, 5)
    _, binary = cv2.threshold(denoised, 180, 255, cv2.THRESH_BINARY_INV)

    row_lines, col_lines = build_grid_lines()

    dates = [f"{i}日" for i in range(1, N_DAYS + 1)]
    workers = [f"C{i:04d}" for i in range(1, N_DATA_ROWS + 1)]
    matrix = []
    for r in range(N_DATA_ROWS):
        y0, y1 = int(row_lines[r]), int(row_lines[r + 1])
        row_vals = []
        for c in range(N_DAYS):
            x0, x1 = int(col_lines[c]), int(col_lines[c + 1])
            cell_bin = binary[y0:y1, x0:x1]
            row_vals.append(classify_cell(cell_bin))
        matrix.append(row_vals)

    df = pd.DataFrame(matrix, columns=dates, index=workers)
    return df, building_name, annotated


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

camera_image = st.camera_input("請對準散工申報表拍攝 (需包含 4 角校正標記 + 大廈標記)")
st.markdown("**— 或者 —**")
uploaded_image = st.file_uploader("上載已有嘅圖片（例如相簿入面已影低嘅相）", type=["jpg", "jpeg", "png"])

img_file = camera_image or uploaded_image

if img_file:
    bytes_data = img_file.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("🔄 正在進行雲端 OpenCV 解析中..."):
        df, building, annotated_img = process_image(img)

    if df is None:
        st.error(building)
        st.image(img, channels="BGR", caption="原始相片")
    else:
        st.success(f"🏢 成功辨識大廈：**{building}**")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 出勤數據預覽")
            st.dataframe(df, height=300)
            st.caption("符號說明：X = 有開工　○圓形記號 = 有開工（另一種）　╱ = 確認當日冇開工　空白 = 可能漏填")

        with col2:
            st.subheader("🔍 角標與校正定位")
            st.image(annotated_img, channels="BGR", use_container_width=True)

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
