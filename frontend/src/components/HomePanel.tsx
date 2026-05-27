export function HomePanel() {
  const processSteps = [
    'Người dùng chọn ảnh mẫu hoặc tải ảnh CT phổi',
    'Hệ thống kiểm tra ảnh đầu vào',
    'YOLO localization khoanh vùng nốt hoặc vùng nghi ngờ',
    'Model multimodal phân tích ảnh ROI và thông tin bệnh nhân',
    'Trả về mức nguy cơ, xác suất và confidence',
    'Lưu kết quả vào MongoDB để xem lịch sử',
  ];

  return (
    <main className="home-page">
      <section className="panel home-intro-card">
        <div className="home-copy">
          <p className="eyebrow">Trang chủ</p>
          <h2>Hệ thống hỗ trợ dự đoán nguy cơ ác tính phổi</h2>
          <p className="home-copy__goal">
            <strong>Mục tiêu:</strong> hỗ trợ demo nghiên cứu theo hướng trực quan, nhất quán và dễ theo dõi.
          </p>
          <p className="home-copy__lead">
            Hệ thống hỗ trợ demo nghiên cứu dự đoán nguy cơ ác tính phổi từ ảnh CT và thông tin bệnh nhân,
            đồng thời cho phép xem vùng nghi ngờ được khoanh và lưu lại lịch sử dự đoán.
          </p>
        </div>
      </section>

      <div className="home-grid">
        <section className="panel home-card">
          <div className="panel__header">
            <div>
              <h2>Quy trình xử lý</h2>
              <p className="panel__subtitle">Luồng xử lý chính của hệ thống từ lúc nạp ảnh đến khi lưu lại kết quả.</p>
            </div>
          </div>

          <div className="home-steps">
            {processSteps.map((step, index) => (
              <div key={step} className="home-step">
                <div className="home-step__index">{index + 1}</div>
                <div className="home-step__content">
                  <strong>{step}</strong>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="panel home-card">
          <div className="panel__header">
            <div>
              <h2>Model được sử dụng</h2>
              <p className="panel__subtitle">Các thành phần chính và vai trò của chúng trong hệ thống hiện tại.</p>
            </div>
          </div>

          <div className="home-model-list">
            <div className="home-model-item">
              <span className="home-model-item__label">Classification model</span>
              <strong>
                Mô hình phân loại đa phương thức, kết hợp ảnh ROI và thông tin bệnh nhân để ước lượng nguy cơ thấp,
                trung gian hoặc cao.
              </strong>
            </div>
            <div className="home-model-item">
              <span className="home-model-item__label">YOLO localization model</span>
              <strong>
                Mô hình YOLO dùng để khoanh vùng nốt hoặc vùng nghi ngờ trên ảnh CT, giúp người dùng đối chiếu vị trí
                cần chú ý.
              </strong>
            </div>
            <div className="home-model-item">
              <span className="home-model-item__label">MongoDB</span>
              <strong>
                Hệ quản trị dữ liệu dùng để lưu lịch sử dự đoán, giúp tra cứu lại các ca đã chạy và xem chi tiết kết quả.
              </strong>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
