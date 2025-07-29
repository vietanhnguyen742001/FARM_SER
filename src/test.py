import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Hàm ContrastiveLoss của bạn ---
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_emb, audio_emb, labels):
        # 1. Chuẩn hóa embeddings
        text_emb = F.normalize(text_emb, dim=-1)
        audio_emb = F.normalize(audio_emb, dim=-1)

        # 2. Tính toán ma trận độ tương đồng (cosine similarity)
        #    logits[i, j] là độ tương đồng giữa text_emb[i] và audio_emb[j]
        #    Chia cho temperature để scale logits
        logits = torch.matmul(text_emb, audio_emb.T) / self.temperature

        # 3. Tạo mask cho các cặp dương (positive pairs)
        #    Cặp dương là những cặp có cùng nhãn trong batch.
        #    labels.view(-1, 1) chuyển nhãn thành cột vector để so sánh dễ hơn.
        #    torch.eq(labels, labels.T) tạo ma trận boolean True/False nơi nhãn khớp nhau.
        #    .float() chuyển True/False thành 1.0/0.0
        labels_reshaped = labels.view(-1, 1)
        mask = torch.eq(labels_reshaped, labels_reshaped.T).float()
        
        # 4. Loại bỏ các cặp tự so sánh (diagonal)
        #    mask.fill_diagonal_(0) đảm bảo bạn không so sánh một mẫu với chính nó khi tính loss.
        #    Điều này là quan trọng vì độ tương đồng của một embedding với chính nó luôn là 1,
        #    sẽ làm méo mó giá trị loss nếu được tính vào.
        mask.fill_diagonal_(0)

        # 5. Áp dụng log_softmax lên logits
        #    log_softmax chuẩn hóa các logits thành log-xác suất.
        log_probs = F.log_softmax(logits, dim=1)
        
        # 6. Tính loss
        #    -log_probs[mask.bool()] chọn ra các log-xác suất của các cặp dương (được đánh dấu bởi mask).
        #    .mean() tính trung bình của các giá trị này.
        #    Dấu trừ (-) là vì chúng ta muốn tối đa hóa các log_probs này (khi log_prob cao, -log_prob thấp).
        
        # Xử lý trường hợp không có cặp dương nào trong batch (mask toàn 0)
        # Điều này có thể xảy ra nếu batch_size quá nhỏ hoặc tất cả các nhãn đều unique.
        if mask.bool().sum() == 0:
            print("Cảnh báo: Không có cặp dương nào (ngoại trừ cặp tự so sánh) trong batch cho ContrastiveLoss. Trả về loss 0.0.")
            return torch.tensor(0.0, device=logits.device)

        loss = -log_probs[mask.bool()].mean()
        return loss

# --- Cài đặt kiểm tra ---
# Cố định seed để đảm bảo kết quả có thể tái tạo
torch.manual_seed(42)

batch_size = 8
embedding_dim = 256 # Kích thước embedding của bạn
num_classes = 4     # Số lớp của bạn
temperature = 0.07

# Khởi tạo hàm loss
criterion = ContrastiveLoss(temperature=temperature)

print("--- Bắt đầu kiểm tra ContrastiveLoss với 4 lớp ---")
print(f"Sử dụng temperature: {temperature}")
print(f"Batch size: {batch_size}, Embedding dimension: {embedding_dim}, Number of classes: {num_classes}")

# Labels mẫu có 4 lớp, đảm bảo có các nhãn lặp lại để tạo positive pairs
labels_test = torch.tensor([0, 1, 0, 2, 1, 3, 2, 3], dtype=torch.long)
print(f"Nhãn trong batch: {labels_test.tolist()}")

# --- Kịch bản 1: Kịch bản cơ bản (Random + một chút nhiễu) ---
print("\nKịch bản 1: Embeddings ngẫu nhiên có nhiễu nhỏ (Base Case)")
text_emb_base = torch.randn(batch_size, embedding_dim)
audio_emb_base = text_emb_base + 0.01 * torch.randn(batch_size, embedding_dim) # Thêm nhiễu nhỏ
loss_base = criterion(text_emb_base, audio_emb_base, labels_test)
print(f"   Loss: {loss_base.item():.4f} (Giá trị khởi điểm khi embeddings chưa học)")
# Giải thích: Loss sẽ không quá thấp vì các cặp cùng nhãn chưa thực sự "kéo" lại gần nhau do ngẫu nhiên.

# --- Kịch bản 2: Kịch bản lý tưởng (Loss rất thấp) ---
print("\nKịch bản 2: Embeddings lý tưởng (Loss nên rất thấp, gần 0)")
# Tạo các embedding sao cho các cặp cùng nhãn cực kỳ giống nhau (cosine sim = 1)
# và các cặp khác nhãn không liên quan (cosine sim gần 0)
ideal_text_emb = torch.zeros(batch_size, embedding_dim)
ideal_audio_emb = torch.zeros(batch_size, embedding_dim)

# Định nghĩa một số vector embedding độc đáo cho mỗi nhãn
# Mỗi nhãn sẽ có một "vector đại diện" riêng biệt, được chuẩn hóa để cosine sim = 1 khi khớp.
unique_ideal_vectors = [F.normalize(torch.randn(embedding_dim), dim=-1) for _ in range(num_classes)]

for i, label in enumerate(labels_test):
    # Các mẫu cùng nhãn sẽ có embeddings giống hệt nhau
    ideal_text_emb[i] = unique_ideal_vectors[label]
    ideal_audio_emb[i] = unique_ideal_vectors[label]

loss_ideal = criterion(ideal_text_emb, ideal_audio_emb, labels_test)
print(f"   Loss: {loss_ideal.item():.4f} (Mong đợi: Rất gần 0, ví dụ < 0.01)")
# Giải thích: Nếu các cặp dương có độ tương đồng rất cao (nhờ embedding giống hệt),
# thì log_probs của chúng sẽ rất cao (gần 0), dẫn đến loss rất thấp.

# --- Kịch bản 3: Kịch bản tệ nhất (Loss rất cao) ---
print("\nKịch bản 3: Embeddings tệ nhất (Loss nên rất cao)")
# Tạo các embedding sao cho các cặp cùng nhãn lại **rất khác nhau** (cosine sim âm hoặc thấp)
# và các cặp khác nhãn có thể lại gần nhau (để tạo nhiều "hard negatives" cho InfoNCE truyền thống)
worst_text_emb = torch.randn(batch_size, embedding_dim)
worst_audio_emb = torch.randn(batch_size, embedding_dim)

# Để làm loss cao, chúng ta làm cho các cặp dương có similarity thấp
# Ví dụ: gán các vector ngẫu nhiên cho các cặp dương, hoặc thậm chí là vector ngược chiều.
for i in range(batch_size):
    # Với các cặp (text_i, audio_j) mà labels[i] == labels[j] (positive pairs)
    # Chúng ta muốn độ tương đồng của chúng thấp.
    # Đơn giản nhất là để chúng là ngẫu nhiên hoàn toàn, không có sự khớp nối
    pass # Ngẫu nhiên ban đầu đã khá tệ rồi.

# Hoặc có thể cố ý làm chúng "phản đối" nhau để đẩy loss lên cực cao
for i, label_i in enumerate(labels_test):
    for j, label_j in enumerate(labels_test):
        if i != j and label_i == label_j: # Tìm một cặp dương không phải self-pair
            # Làm cho audio_emb[j] ngược hướng với text_emb[i]
            worst_audio_emb[j] = -worst_text_emb[i] + 0.1 * torch.randn(embedding_dim)
            break # Chỉ cần xử lý một vài cặp để thấy hiệu ứng

loss_worst = criterion(worst_text_emb, worst_audio_emb, labels_test)
print(f"   Loss: {loss_worst.item():.4f} (Mong đợi: Rất cao, ví dụ > 5.0)")
# Giải thích: Nếu các cặp dương có độ tương đồng thấp (hoặc âm), thì log_probs của chúng sẽ rất thấp (âm rất lớn),
# dẫn đến loss rất cao.

print("\n--- Kết thúc kiểm tra ContrastiveLoss ---")
print("Nếu kết quả phù hợp với các giá trị 'Mong đợi' cho từng kịch bản,")
print("hàm ContrastiveLoss của bạn đã được triển khai đúng logic.")