from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load dữ liệu ---
# Đọc customer-item matrix với index_col và ép dtype, loại bỏ .0
customer_item_matrix = pd.read_csv(
    "/Users/hoangvupham/Desktop/KHDL_mar/app/data/customer_item_matrix.csv",
    index_col=0,
    low_memory=False,
    dtype={0: str}
)
# Loại bỏ đuôi '.0' nếu có
customer_item_matrix.index = (
    customer_item_matrix.index
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)
# Đảm bảo index và columns đều là chuỗi
customer_item_matrix.index = customer_item_matrix.index.astype(str)
customer_item_matrix.columns = customer_item_matrix.columns.astype(str)

# Load ma trận tương đồng user-user
user_user_sim_matrix = pd.read_csv(
    "/Users/hoangvupham/Desktop/KHDL_mar/app/data/user_user_sim_matrix.csv",
    index_col=0,
    low_memory=False
).astype(float)
user_user_sim_matrix.index = user_user_sim_matrix.index.astype(str)
user_user_sim_matrix.columns = user_user_sim_matrix.columns.astype(str)

# Load ma trận tương đồng item-item
item_item_sim_matrix = pd.read_csv(
    "/Users/hoangvupham/Desktop/KHDL_mar/app/data/item_item_sim_matrix.csv",
    index_col=0,
    low_memory=False
).astype(float)
item_item_sim_matrix.index = item_item_sim_matrix.index.astype(str)
item_item_sim_matrix.columns = item_item_sim_matrix.columns.astype(str)

# Load thông tin mô tả sản phẩm và ép StockCode thành str
df = pd.read_csv(
    "/Users/hoangvupham/Desktop/KHDL_mar/app/data/Data_G5_Cleaned.csv",
    dtype={"StockCode": str}
)
df["StockCode"] = df["StockCode"].astype(str)

# ===== TÍNH USER-BASED & ITEM-BASED SCORE =====
k = 5
user_item_np = customer_item_matrix.to_numpy()
user_ids = customer_item_matrix.index
item_ids = customer_item_matrix.columns

user_sim_np = user_user_sim_matrix.to_numpy()
item_sim_np = item_item_sim_matrix.to_numpy()

# Top-k user similarity
topk_user_sim = np.zeros_like(user_sim_np)
for i in range(user_sim_np.shape[0]):
    idx = np.argsort(-user_sim_np[i])[:k+1]
    idx = idx[idx != i]
    topk_user_sim[i, idx] = user_sim_np[i, idx]

user_based_score_np = np.dot(topk_user_sim, user_item_np)
user_sim_sums = np.sum(topk_user_sim, axis=1).reshape(-1, 1)
user_based_score_np = np.divide(user_based_score_np, user_sim_sums, where=user_sim_sums != 0)

# Top-k item similarity
topk_item_sim = np.zeros_like(item_sim_np)
for i in range(item_sim_np.shape[0]):
    idx = np.argsort(-item_sim_np[i])[:k+1]
    idx = idx[idx != i]
    topk_item_sim[i, idx] = item_sim_np[i, idx]

item_based_score_np = np.dot(user_item_np, topk_item_sim)
item_sim_sums = np.sum(topk_item_sim, axis=1).reshape(1, -1)
item_based_score_np = np.divide(item_based_score_np, item_sim_sums, where=item_sim_sums != 0)

# Chuyển thành DataFrame và loại bỏ sản phẩm đã mua
user_based_score = pd.DataFrame(user_based_score_np, index=user_ids, columns=item_ids)
item_based_score = pd.DataFrame(item_based_score_np, index=user_ids, columns=item_ids)
user_based_score[customer_item_matrix > 0] = 0
item_based_score[customer_item_matrix > 0] = 0

# Hàm gợi ý top 5 không trùng
def recommend_top_5_distinct(user_id, top_n_each=5):
    user_id = str(user_id)
    if user_id not in customer_item_matrix.index:
        return {'user_based': [], 'item_based': []}

    top_user = user_based_score.loc[user_id].nlargest(top_n_each).index.tolist()
    item_scores = item_based_score.loc[user_id].drop(index=top_user, errors='ignore')
    top_item = item_scores.nlargest(top_n_each).index.tolist()

    # Nếu thiếu, thêm filler
    if len(top_item) < top_n_each:
        rem = top_n_each - len(top_item)
        all_items = item_based_score.columns.difference(top_user + top_item)
        filler = item_based_score.loc[user_id, all_items].nlargest(rem).index.tolist()
        top_item += filler

    return {'user_based': top_user, 'item_based': top_item}

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend/user", methods=["GET"])
def recommend_user():
    user_id = request.args.get("customer_id", "").strip()
    if not user_id or user_id not in customer_item_matrix.index:
        return jsonify({"error": "User ID không hợp lệ hoặc không tồn tại."}), 400

    recs = recommend_top_5_distinct(user_id)
    def enrich(codes):
        return [{
            'StockCode': code,
            'Description': df[df['StockCode'] == code]['Description'].dropna().unique().tolist()[:1] or ['Unknown'][0]
        } for code in codes]

    return jsonify({
        'user_based': enrich(recs['user_based']),
        'item_based': enrich(recs['item_based'])
    })

if __name__ == "__main__":
    app.run(debug=True)
